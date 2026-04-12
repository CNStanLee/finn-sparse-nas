import os, torch
from brevitas.nn import QuantLinear, QuantConv2d
from nas.task_factory import make_dataloaders, build_model
from nas.train import eval_acc, train_finetune


def build_unstructured_weight_masks(model):
    masks = {}
    for name, module in model.named_modules():
        if not isinstance(module, (QuantLinear, QuantConv2d)):
            continue
        if not hasattr(module, "weight") or module.weight is None:
            continue
        w = module.weight
        masks[w] = (w.detach() != 0).float()
    return masks


def unstructured_sparsity_stats(model, min_numel=0):
    per_layer = []
    total = 0
    nonzero = 0
    for name, module in model.named_modules():
        if not isinstance(module, (QuantLinear, QuantConv2d)):
            continue
        if not hasattr(module, "weight") or module.weight is None:
            continue
        w = module.weight.detach()
        n = int(w.numel())
        if n < int(min_numel):
            continue
        nz = int(torch.count_nonzero(w).item())
        sp = 0.0 if n == 0 else (1.0 - (nz / n))
        per_layer.append({
            "name": f"{name}.weight",
            "layer_type": module.__class__.__name__,
            "numel": n,
            "nonzero": nz,
            "sparsity": float(sp),
        })
        total += n
        nonzero += nz
    overall = 0.0 if total == 0 else (1.0 - (nonzero / total))
    return float(overall), int(total), int(nonzero), per_layer



@torch.no_grad()
def apply_unstructured_global_magnitude_prune(model, target_sparsity, min_retain_per_layer=64):
    # In-place global magnitude pruning on QuantLinear/QuantConv2d weights
    if target_sparsity <= 0.0:
        return

    layers = []
    all_abs = []
    for name, module in model.named_modules():
        if not isinstance(module, (QuantLinear, QuantConv2d)):
            continue
        if not hasattr(module, "weight") or module.weight is None:
            continue
        w = module.weight
        n = int(w.numel())
        if n <= int(min_retain_per_layer):
            continue
        layers.append((name, w))
        all_abs.append(w.abs().reshape(-1))

    if len(all_abs) == 0:
        return

    all_abs = torch.cat(all_abs)
    total_params = int(all_abs.numel())
    k_prune = int(target_sparsity * total_params)
    if k_prune <= 0:
        return

    sorted_abs = torch.sort(all_abs).values
    thr = sorted_abs[min(k_prune - 1, total_params - 1)]

    for name, w in layers:
        abs_w = w.abs()
        mask = abs_w > thr
        remain = int(mask.sum().item())
        if remain < int(min_retain_per_layer):
            flat = abs_w.reshape(-1)
            kth = torch.sort(flat).values[-int(min_retain_per_layer)]
            mask = abs_w >= kth
        w.mul_(mask)



def prune_unstructured_weights_file(cfg, cand, in_weights, out_weights, target_sparsity, min_retain_per_layer=64, finetune=True, base_acc=None):
    device = "cuda" if torch.cuda.is_available() and cfg["ea"]["cuda"] else "cpu"
    dl_tr, dl_va = make_dataloaders(cfg, phase="finetune")

    model = build_model(cfg, cand)
    model.load_state_dict(torch.load(in_weights, map_location="cpu"))
    model.eval()

    # Prune (in-place)
    apply_unstructured_global_magnitude_prune(model, target_sparsity, min_retain_per_layer)
    acc_raw = eval_acc(model.to(device), dl_va, device)
    print(f"[PRUNING] {target_sparsity*100}% -> RAW val_acc={acc_raw:.4f}")

    # Finetune (optional)
    acc_ft = None; eps = 0.02
    if (finetune and target_sparsity > 0.0) and (acc_raw + eps < base_acc if base_acc is not None else True):
        weight_masks = build_unstructured_weight_masks(model)
        acc_ft = train_finetune(
            cfg, out_weights,
            model=model,
            dl_tr=dl_tr, dl_va=dl_va,
            weight_masks=weight_masks
        )
        print(f"[PRUNING] {target_sparsity*100}% -> FINETUNE val_acc={acc_ft:.4f}")
        model.load_state_dict(torch.load(out_weights, map_location="cpu"))
    else:
        os.makedirs(os.path.dirname(out_weights), exist_ok=True)
        torch.save(model.state_dict(), out_weights)

    sp, total, nz, per_layer = unstructured_sparsity_stats(model)

    summary = {
        "val_acc_after_prune_raw": acc_raw,
        "val_acc_after_finetune": acc_ft,

        "target_sparsity": float(target_sparsity),
        "achieved_sparsity": float(sp),
        "total_params": int(total),
        "total_nonzero": int(nz),
        "per_layer": per_layer,

        "in_weights": str(in_weights),
        "out_weights": str(out_weights),
    }
    return summary