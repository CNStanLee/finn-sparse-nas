import os, torch
import torch.nn as nn
from brevitas.nn import QuantLinear
from nas.train import make_dataloaders, build_jsc_model, eval_acc, train


def expand_keep_ratios(keep_ratio, n_hidden):
    if isinstance(keep_ratio, (list, tuple)):
        keep_ratios = [float(x) for x in keep_ratio]
        if len(keep_ratios) != n_hidden:
            raise ValueError(f"Expected {n_hidden} keep ratios, got {len(keep_ratios)}")
        return keep_ratios
    return [float(keep_ratio)] * n_hidden


def aligned_keep_count(width, keep_ratio, align=4, min_units=4):
    if keep_ratio >= 1.0:
        return int(width)
    raw = int(width * keep_ratio)
    if align > 1:
        raw = (raw // align) * align
    if raw < min_units:
        raw = min_units
    if raw > width:
        raw = width
    if raw <= 0:
        raw = min(1, width)
    return int(raw)


def structured_linear_param_stats(model):
    per_layer = []
    total = 0
    for name, module in model.named_modules():
        if not isinstance(module, QuantLinear):
            continue
        if not hasattr(module, "weight") or module.weight is None:
            continue
        w = module.weight.detach()
        n = int(w.numel())
        per_layer.append({
            "name": f"{name}.weight",
            "numel": n,
            "out_features": int(w.shape[0]),
            "in_features": int(w.shape[1]),
        })
        total += n
    return int(total), per_layer


def make_structured_sweep_vals(hidden, global_vals):
    n_hidden = len(hidden)
    global_vals = [float(x) for x in global_vals]
    sweep = [[g] * n_hidden for g in global_vals] # global variants
    if n_hidden == 0:
        return [[1.0]]
    hi = global_vals[0]
    mid = global_vals[1] if len(global_vals) > 1 else global_vals[0]
    low = global_vals[-1]

    # add a few automatic per-layer variants
    if n_hidden == 1:
        extras = []
    elif n_hidden == 2:
        extras = [
            [hi, low],
            [mid, low],
            [low, hi],
        ]
    else:
        mid_idx = n_hidden // 2
        def one_high(idx):
            return [hi if j == idx else low for j in range(n_hidden)]
        extras = [
            one_high(0),
            one_high(mid_idx),
            one_high(n_hidden - 1),
        ]
    sweep.extend(extras)

    # dedupe while preserving order
    out = []
    seen = set()
    for v in sweep:
        key = tuple(float(x) for x in v)
        if key in seen:
            continue
        seen.add(key)
        out.append(list(key))
    return out



def apply_structured_neuron_prune(cfg, cand, in_weights, keep_ratio, align=4, min_units=4):
    old_hidden = list(cand["hidden"])
    model = build_jsc_model(cfg, cand, old_hidden)
    old_state = torch.load(in_weights, map_location="cpu")
    model.load_state_dict(old_state)
    model.eval()

    n_hidden = len(old_hidden)
    if n_hidden == 0:
        new_cand = {"hidden": old_hidden, "quant": cand["quant"]}
        return model, new_cand, old_hidden, old_hidden, []
    
    keep_ratios = expand_keep_ratios(keep_ratio, n_hidden)

    # No structural pruning requested
    if all(r >= 1.0 for r in keep_ratios):
        new_cand = {"hidden": old_hidden, "quant": cand["quant"]}
        return model, new_cand, old_hidden, old_hidden, keep_ratios

    # Collect QuantLinear layers in order + names
    old_linears = []
    old_linear_names = []
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            old_linears.append(module)
            old_linear_names.append(name)

    # Decide which neurons to keep in each hidden layer
    keep_indices = []
    new_hidden = []
    for i in range(n_hidden):
        cur = old_linears[i]
        nxt = old_linears[i + 1]
        width = int(cur.weight.shape[0])
        keep_k = aligned_keep_count(width, keep_ratios[i], align=align, min_units=min_units)

        # neuron importance = current row L1 + next-layer input-column L1
        cur_score = cur.weight.data.abs().sum(dim=1)
        nxt_score = nxt.weight.data.abs().sum(dim=0)
        score = cur_score + nxt_score

        idx = torch.topk(score, k=keep_k, largest=True).indices
        idx = torch.sort(idx).values
        keep_indices.append(idx)
        new_hidden.append(int(keep_k))

    new_cand = {"hidden": new_hidden, "quant": cand["quant"]}

    # Reuse the original model/state if widths are the same
    if new_hidden == old_hidden:
        return model, new_cand, old_hidden, new_hidden, keep_ratios

    # Build smaller model
    pruned_model = build_jsc_model(cfg, cand, new_hidden)
    pruned_model.eval()

    # Collect new QuantLinear layers in order + names
    new_linears = []
    new_linear_names = []
    for name, module in pruned_model.named_modules():
        if isinstance(module, QuantLinear):
            new_linears.append(module)
            new_linear_names.append(name)

    # Collect BatchNorm1d layers
    old_bns_all = []
    old_bn_names_all = []
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm1d):
            old_bns_all.append(module)
            old_bn_names_all.append(name)

    new_bns_all = []
    new_bn_names_all = []
    for name, module in pruned_model.named_modules():
        if isinstance(module, nn.BatchNorm1d):
            new_bns_all.append(module)
            new_bn_names_all.append(name)

    # Hidden BNs only (skip input BN at index 0)
    old_bns = old_bns_all[1:]
    old_bn_names = old_bn_names_all[1:]
    new_bns = new_bns_all[1:]
    new_bn_names = new_bn_names_all[1:]

    # Start from the new model state_dict and preserve all matching-shape entries
    new_state = pruned_model.state_dict()
    for k, v in old_state.items():
        if k in new_state and new_state[k].shape == v.shape:
            new_state[k] = v.clone()

    # Overwrite QuantLinear weights / biases with sliced tensors
    for i in range(len(old_linears)):
        old_lin = old_linears[i]

        W = old_lin.weight.data.clone()
        B = old_lin.bias.data.clone() if old_lin.bias is not None else None

        # previous hidden layer pruned -> slice input columns
        if i > 0:
            prev_idx = keep_indices[i - 1]
            W = W[:, prev_idx]

        # current hidden layer pruned -> slice output rows
        if i < n_hidden:
            cur_idx = keep_indices[i]
            W = W[cur_idx, :]
            if B is not None:
                B = B[cur_idx]

        weight_key = f"{new_linear_names[i]}.weight"
        new_state[weight_key] = W.clone()

        bias_key = f"{new_linear_names[i]}.bias"
        if bias_key in new_state and B is not None:
            new_state[bias_key] = B.clone()

    # Overwrite hidden BatchNorm state with sliced tensors
    for i in range(n_hidden):
        idx = keep_indices[i]

        weight_key = f"{new_bn_names[i]}.weight"
        if weight_key in new_state and old_bns[i].weight is not None:
            new_state[weight_key] = old_bns[i].weight.data[idx].clone()

        bias_key = f"{new_bn_names[i]}.bias"
        if bias_key in new_state and old_bns[i].bias is not None:
            new_state[bias_key] = old_bns[i].bias.data[idx].clone()

        rm_key = f"{new_bn_names[i]}.running_mean"
        if rm_key in new_state:
            new_state[rm_key] = old_bns[i].running_mean.data[idx].clone()

        rv_key = f"{new_bn_names[i]}.running_var"
        if rv_key in new_state:
            new_state[rv_key] = old_bns[i].running_var.data[idx].clone()

        # scalar buffer, keep as-is from old state if present
        nbt_key = f"{new_bn_names[i]}.num_batches_tracked"
        if nbt_key in new_state and nbt_key in old_state:
            new_state[nbt_key] = old_state[nbt_key].clone()

    pruned_model.load_state_dict(new_state, strict=False)
    return pruned_model, new_cand, old_hidden, new_hidden, keep_ratios



def prune_structured_neurons_weights_file(cfg, cand, in_weights, out_weights, keep_ratio, align=4, min_units=4, finetune=True):
    device = "cuda" if torch.cuda.is_available() and cfg["ea"]["cuda"] else "cpu"
    dl_tr, dl_va = make_dataloaders(cfg)

    pruned_model, new_cand, old_hidden, new_hidden, keep_ratios = apply_structured_neuron_prune(
        cfg, cand, in_weights, keep_ratio, align=align, min_units=min_units
    )

    # Raw pruned accuracy
    acc_raw = float(eval_acc(pruned_model.to(device), dl_va, device))
    print(f"[PRUNING] keep ratios: {keep_ratios} -> RAW val_acc={acc_raw:.4f}")

    # Optional finetune
    acc_ft = None
    if finetune and new_hidden != old_hidden:
        acc_ft = train(
            cfg, out_weights,
            model=pruned_model,
            epochs=cfg["finalists"]["finetune"]["epochs"],
            lr=cfg["finalists"]["finetune"]["lr"],
            weight_decay=cfg["finalists"]["finetune"]["weight_decay"],
            early_stop_patience=cfg["finalists"]["finetune"]["early_stop_patience"],
            dl_tr=dl_tr, dl_va=dl_va
        )
        print(f"[PRUNING] keep ratios: {keep_ratios} -> FINETUNE val_acc={acc_ft:.4f}")
        pruned_model.load_state_dict(torch.load(out_weights, map_location="cpu"))
    else:
        os.makedirs(os.path.dirname(out_weights), exist_ok=True)
        torch.save(pruned_model.state_dict(), out_weights)

    total_params, per_layer = structured_linear_param_stats(pruned_model)

    summary = {
        "keep_ratio": keep_ratio,
        "requested_keep_ratio_per_layer": keep_ratios,
        "align": int(align),
        "min_units": int(min_units),

        "old_hidden": old_hidden,
        "new_hidden": new_hidden,
        "new_cand": new_cand,

        "kept_units_per_layer": new_hidden,
        "pruned_units_per_layer": [int(o - n) for o, n in zip(old_hidden, new_hidden)],
        "actual_keep_ratio_per_layer": [
            float(n / o) if o > 0 else 0.0 for o, n in zip(old_hidden, new_hidden)
        ],

        "val_acc_after_prune_raw": acc_raw,
        "val_acc_after_finetune": acc_ft,

        "total_linear_weight_params": total_params,
        "per_layer": per_layer,

        "in_weights": str(in_weights),
        "out_weights": str(out_weights),
    }
    return summary