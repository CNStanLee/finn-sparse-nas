import os, torch, copy
from nas.task_factory import build_model, make_dataloaders, make_training_recipe, compute_loss, post_opt_step


@torch.no_grad()
def eval_acc(model, dl, device):
    model.eval(); 
    correct = 0; total = 0
    for xb, yb in dl:
        xb, yb = xb.to(device), yb.to(device)
        correct += (model(xb).argmax(1) == yb).sum().item()
        total += xb.size(0)
    return correct/max(1, total)


def train(cfg, out_weights, phase="quick", cand=None, model=None, dl_tr=None, dl_va=None, weight_masks=None):
    device = "cuda" if torch.cuda.is_available() and cfg["ea"]["cuda"] else "cpu"

    if model is None:
        model = build_model(cfg, cand)
    model = model.to(device)

    if dl_tr is None or dl_va is None:
        dl_tr, dl_va = make_dataloaders(cfg, phase=phase)

    recipe = make_training_recipe(cfg, model, phase=phase)
    opt = recipe["optimizer"]
    scheduler = recipe["scheduler"]
    epochs = recipe["epochs"]
    early_stop_patience = recipe["early_stop_patience"]
    min_delta = recipe["min_delta"]
    min_epochs = recipe["min_epochs"]

    if weight_masks is not None:
        weight_masks = {p: mask.to(device) for p, mask in weight_masks.items()}

    best_acc = -1.0
    best_state = None
    best_epoch = -1
    patience_left = early_stop_patience

    for epoch in range(epochs):
        model.train()

        for xb, yb in dl_tr:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            logits = model(xb)
            loss = compute_loss(cfg, recipe, logits, yb, device)
            opt.zero_grad(); loss.backward()

            # only used for unstructured finetuning
            if weight_masks is not None:
                for p, mask in weight_masks.items():
                    if p.grad is not None:
                        p.grad.mul_(mask)

            opt.step(); post_opt_step(cfg, model)

            # keep pruned weights at zero
            if weight_masks is not None:
                with torch.no_grad():
                    for p, mask in weight_masks.items():
                        p.mul_(mask)

        if scheduler is not None:
            scheduler.step()

        acc_val = eval_acc(model, dl_va, device)
        print(f"[{phase} epoch {epoch}] val_acc={acc_val:.4f}")

        improved = acc_val > (best_acc + min_delta)
        if improved:
            best_acc = acc_val
            best_epoch = epoch
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            if early_stop_patience is not None:
                patience_left = early_stop_patience
        else:
            if early_stop_patience is not None and epoch >= min_epochs:
                patience_left -= 1
                if patience_left <= 0:
                    print(f"[early stop] no improvement for {early_stop_patience} evals. "
                          f"best={best_acc:.4f} at epoch {best_epoch}")
                    break

    os.makedirs(os.path.dirname(out_weights), exist_ok=True)
    torch.save(best_state if best_state is not None else model.state_dict(), out_weights)
    return float(best_acc)


def train_quick(cfg, cand, out_weights):
    return train(cfg, out_weights, phase="quick", cand=cand)


def train_full(cfg, cand, out_weights):
    return train(cfg, out_weights, phase="full", cand=cand)


def train_finetune(cfg, out_weights, cand=None, model=None, dl_tr=None, dl_va=None, weight_masks=None):
    return train(cfg, out_weights, phase="finetune", cand=cand, model=model, dl_tr=dl_tr, dl_va=dl_va, weight_masks=weight_masks)
    