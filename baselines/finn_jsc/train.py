import os, json, torch
from torch.utils.data import DataLoader
from datasets.jsc_logicnets.dataset import JetSubstructureDataset
from models.brevitas_mlp import JetSubstructureModel


@torch.no_grad()
def acc(model, dl, device):
    model.eval(); 
    correct = 0; total = 0
    for xb, yb in dl:
        xb, yb = xb.to(device), yb.to(device)
        correct += (model(xb).argmax(1) == yb).sum().item()
        total += xb.size(0)
    return correct/max(1, total)


def train(cfg, arch):
    os.makedirs(f"results/baseline/{arch}", exist_ok=True)

    # data
    ds_tr = JetSubstructureDataset(cfg["data"]["input_file"], cfg["data"]["config_file"], split="train", val_frac=cfg["data"]["val_frac"], seed=cfg["data"]["seed"])
    ds_va = JetSubstructureDataset(cfg["data"]["input_file"], cfg["data"]["config_file"], split="val",   val_frac=cfg["data"]["val_frac"], seed=cfg["data"]["seed"])
    ds_te = JetSubstructureDataset(cfg["data"]["input_file"], cfg["data"]["config_file"], split="test",  val_frac=cfg["data"]["val_frac"], seed=cfg["data"]["seed"])

    dl_train = DataLoader(ds_tr, batch_size=cfg["train"]["batch_size"], shuffle=True, drop_last=True)
    dl_val   = DataLoader(ds_va, batch_size=1024)
    dl_test  = DataLoader(ds_te, batch_size=1024)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # model
    m = JetSubstructureModel(
        cfg["task"]["in_features"], 
        cfg["task"]["n_classes"], 
        tuple(cfg["presets"][arch]["model"]["hidden"]), 
        cfg["presets"][arch]["model"]["weight_bits"], 
        cfg["presets"][arch]["model"]["input_act_bits"],
        cfg["presets"][arch]["model"]["hidden_act_bits"],
        cfg["presets"][arch]["model"]["output_act_bits"]
    ).to(device)

    opt = torch.optim.AdamW(m.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    best = {"epoch": -1, "acc_val": 0.0}
    patience, bad = cfg["train"].get("early_stop_patience", 0), 0

    for epoch in range(cfg["train"]["epochs"]):
        m.train()
        for xb, yb in dl_train:
            xb, yb = xb.to(device), yb.to(device)
            logits = m(xb)
            loss = torch.nn.functional.cross_entropy(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()

        # val
        acc_val = acc(m, dl_val, device)
        print(f"[epoch {epoch}] val_acc={acc_val:.4f}")

        if acc_val > best["acc_val"]:
            best = {"epoch": epoch, "acc_val": acc_val}; 
            bad = 0
            torch.save(m.state_dict(), cfg["presets"][arch]["export"]["pytorch"])
        else:
            bad += 1
            if patience and bad >= patience: break

    # test
    m.load_state_dict(torch.load(cfg["presets"][arch]["export"]["pytorch"], map_location=device))
    acc_test = acc(m, dl_test, device)
    print(f"[test] acc={acc_test:.4f}")
    json.dump({"acc_test": acc_test}, open(f"results/baseline/{arch}/metrics.json", "w"))
