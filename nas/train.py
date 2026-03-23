import os, torch, copy
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


def make_test_dataloader(cfg):
    ds_te = JetSubstructureDataset(cfg["data"]["input_file"], cfg["data"]["config_file"], split="test", val_frac=cfg["data"]["val_frac"], seed=cfg["data"]["seed"])
    dl_te = DataLoader(
        ds_te, batch_size=cfg["train"]["val_batch_size"], shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2
    )
    return dl_te


def build_jsc_model(cfg, cand, hidden=None):
    q = cand["quant"]
    picked_hidden = cand["hidden"] if hidden is None else hidden
    return JetSubstructureModel(
        cfg["task"]["in_features"],
        cfg["task"]["n_classes"],
        tuple(picked_hidden),
        q["WB"], q["IA"],
        q["HA"], q["OA"]
    )


@torch.no_grad()
def eval_acc(model, dl, device):
    model.eval(); 
    correct = 0; total = 0
    for xb, yb in dl:
        xb, yb = xb.to(device), yb.to(device)
        correct += (model(xb).argmax(1) == yb).sum().item()
        total += xb.size(0)
    return correct/max(1, total)


def train(cfg, out_weights, cand=None, model=None, epochs=1, lr=1e-3, weight_decay=0.0,
          early_stop_patience=None, min_delta=0.0, min_epochs=0,
          dl_tr=None, dl_va=None, weight_masks=None):

    device = "cuda" if torch.cuda.is_available() and cfg["ea"]["cuda"] else "cpu"

    if model is None:
        model = build_jsc_model(cfg, cand)
    model = model.to(device)

    if dl_tr is None or dl_va is None:
        dl_tr, dl_va = make_dataloaders(cfg)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # move masks to device once
    if weight_masks is not None:
        weight_masks = {p: mask.to(device) for p, mask in weight_masks.items()}

    best_acc = -1.0
    best_state = None
    best_epoch = -1
    patience_left = early_stop_patience

    for epoch in range(int(epochs)):
        model.train()

        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            loss = torch.nn.functional.cross_entropy(model(xb), yb)
            opt.zero_grad()
            loss.backward()

            # only used for unstructured finetuning
            if weight_masks is not None:
                for p, mask in weight_masks.items():
                    if p.grad is not None:
                        p.grad.mul_(mask)

            opt.step()

            # keep pruned weights at zero
            if weight_masks is not None:
                with torch.no_grad():
                    for p, mask in weight_masks.items():
                        p.mul_(mask)

        acc_val = eval_acc(model, dl_va, device)
        print(f"[epoch {epoch}] val_acc={acc_val:.4f}")

        improved = acc_val > (best_acc + float(min_delta))
        if improved:
            best_acc = acc_val
            best_epoch = epoch
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            if early_stop_patience is not None:
                patience_left = early_stop_patience
        else:
            if early_stop_patience is not None and epoch >= int(min_epochs):
                patience_left -= 1
                if patience_left <= 0:
                    print(f"[early stop] no improvement for {early_stop_patience} evals. "
                          f"best={best_acc:.4f} at epoch {best_epoch}")
                    break

    os.makedirs(os.path.dirname(out_weights), exist_ok=True)
    torch.save(best_state if best_state is not None else model.state_dict(), out_weights)
    return float(best_acc)


def train_quick(cfg, cand, out_weights):
    return train(
        cfg, out_weights,
        cand=cand,
        epochs=cfg["train"]["epochs"],
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"]
    )


def train_full(cfg, cand, out_weights):
    tf = cfg["train_full"]
    return train(
        cfg, out_weights,
        cand=cand,
        epochs=tf["epochs"],
        lr=tf["lr"],
        weight_decay=tf["weight_decay"],
        early_stop_patience=tf["early_stop_patience"],
        min_delta=tf["min_delta"],
        min_epochs=tf["min_epochs"]
    )