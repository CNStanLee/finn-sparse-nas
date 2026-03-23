import os, json, torch, copy
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets.cifar10.dataset import CIFAR10Dataset
from models.brevitas_cnv import CIFAR10Model


class SqrHingeLoss(nn.Module):
    def forward(self, logits, targets_pm1):
        return torch.mean(torch.clamp(1.0 - logits * targets_pm1, min=0.0) ** 2)


@torch.no_grad()
def acc(model, dl, device):
    model.eval()
    correct = 0
    total = 0
    for xb, yb in dl:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb).argmax(1)
        correct += (pred == yb).sum().item()
        total += xb.size(0)
    return correct / max(1, total)


def train(cfg, arch):
    out_dir = f"results/baseline/{arch}"
    os.makedirs(out_dir, exist_ok=True)

    ds_tr = CIFAR10Dataset(root=cfg["data"]["root"], split="train", val_frac=cfg["data"]["val_frac"], seed=cfg["data"]["seed"], download=cfg["data"]["download"])
    ds_te = CIFAR10Dataset(root=cfg["data"]["root"], split="test", val_frac=cfg["data"]["val_frac"], seed=cfg["data"]["seed"], download=cfg["data"]["download"])

    dl_train = DataLoader(ds_tr, batch_size=cfg["train"]["batch_size"], shuffle=True, drop_last=False, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    dl_test = DataLoader(ds_te, batch_size=cfg["train"]["batch_size"], shuffle=False, drop_last=False, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    n_classes = int(cfg["task"]["n_classes"])
    p = cfg["presets"][arch]["model"]

    m = CIFAR10Model(
        n_classes=n_classes,
        in_ch=cfg["task"]["in_channels"],
        weight_bits=p["weight_bits"],
        act_bits=p["act_bits"],
        in_bits=p["input_bits"],
        img_h=cfg["task"]["img_h"],
        img_w=cfg["task"]["img_w"],
    ).to(device)

    opt = torch.optim.Adam(m.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    criterion = SqrHingeLoss().to(device)

    best = {"epoch": -1, "acc_test": 0.0}
    epochs = int(cfg["presets"][arch]["train"]["epochs"])

    for epoch in range(epochs):
        m.train()

        for xb, yb in dl_train:
            xb, yb = xb.to(device), yb.to(device)
            logits = m(xb)

            tgt = torch.full((yb.size(0), n_classes), -1.0, device=device)
            tgt.scatter_(1, yb.unsqueeze(1), 1.0)

            loss = criterion(logits, tgt)
            opt.zero_grad(); loss.backward(); opt.step()

            if hasattr(m, "clip_weights"):
                m.clip_weights(-1, 1)

        if (epoch + 1) % int(cfg["train"]["lr_decay_every"]) == 0:
            opt.param_groups[0]["lr"] *= float(cfg["train"]["lr_decay_factor"])

        acc_test = acc(m, dl_test, device)
        print(f"[epoch {epoch}] test_acc={acc_test:.4f}")

        if acc_test >= best["acc_test"]:
            best = {"epoch": epoch, "acc_test": acc_test}
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in m.state_dict().items()})
            torch.save(best_state, cfg["presets"][arch]["export"]["pytorch"])

    m.load_state_dict(torch.load(cfg["presets"][arch]["export"]["pytorch"], map_location=device))
    acc_test = acc(m, dl_test, device)
    print(f"[test] acc={acc_test:.4f}")
    json.dump({"acc_test": acc_test}, open(f"{out_dir}/metrics.json", "w"))