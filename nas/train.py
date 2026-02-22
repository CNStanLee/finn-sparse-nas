import os, torch
from torch.utils.data import DataLoader
from models.brevitas_mlp import JetSubstructureModel
from datasets.jsc_logicnets.dataset import JetSubstructureDataset


def make_dataloaders(cfg):
    ds_tr = JetSubstructureDataset(cfg["data"]["input_file"], cfg["data"]["config_file"], split="train", val_frac=cfg["data"]["val_frac"], seed=cfg["data"]["seed"])
    ds_va = JetSubstructureDataset(cfg["data"]["input_file"], cfg["data"]["config_file"], split="val",   val_frac=cfg["data"]["val_frac"], seed=cfg["data"]["seed"])
    dl_tr = DataLoader(
        ds_tr, batch_size=cfg["train"]["batch_size"], shuffle=True, drop_last=True,
        num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2
    )
    dl_va = DataLoader(
        ds_va, batch_size=cfg["train"]["val_batch_size"], shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2
    )
    return dl_tr, dl_va


@torch.no_grad()
def val_acc(model, dl, device):
    model.eval(); 
    correct = 0; total = 0
    for xb, yb in dl:
        xb, yb = xb.to(device), yb.to(device)
        correct += (model(xb).argmax(1) == yb).sum().item()
        total += xb.size(0)
    return correct/max(1, total)


def train_quick(cfg, cand, out_weights):
    device = "cuda" if torch.cuda.is_available() and cfg["ea"]["cuda"] else "cpu"
    q = cand["quant"]
    
    m = JetSubstructureModel(
        cfg["task"]["in_features"], 
        cfg["task"]["n_classes"], 
        tuple(cand["hidden"]),
        q["WB"], 
        q["IA"],
        q["HA"],
        q["OA"]
    ).to(device)

    opt = torch.optim.AdamW(m.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    dl_tr, dl_va = make_dataloaders(cfg)

    best = 0.0
    for epoch in range(cfg["train"]["epochs"]):
        m.train()
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            loss = torch.nn.functional.cross_entropy(m(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        acc_val = val_acc(m, dl_va, device)
        print(f"[epoch {epoch}] val_acc={acc_val:.4f}")
        best = max(best, acc_val)

    print(f"Saving model to {out_weights}")

    os.makedirs(os.path.dirname(out_weights), exist_ok=True)
    torch.save(m.state_dict(), out_weights)
    return best