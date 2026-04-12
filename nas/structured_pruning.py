import os, torch
import torch.nn as nn
from brevitas.nn import QuantLinear, QuantConv2d
from nas.task_factory import make_dataloaders, build_model
from nas.train import eval_acc, train_finetune


def expand_keep_ratios(keep_ratio, n_layers):
    if isinstance(keep_ratio, (list, tuple)):
        keep_ratios = [float(x) for x in keep_ratio]
        if len(keep_ratios) != n_layers:
            raise ValueError(f"Expected {n_layers} keep ratios, got {len(keep_ratios)}")
        return keep_ratios
    return [float(keep_ratio)] * n_layers


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


def make_structured_meta(new_cand, structure_name, old_structure, new_structure, keep_ratios):
    old_structure = list(old_structure)
    new_structure = list(new_structure)
    keep_ratios = [float(x) for x in keep_ratios]
    return {
        "new_cand": new_cand,
        "structure_name": structure_name,
        "old_structure": old_structure,
        "new_structure": new_structure,
        "requested_keep_ratio_per_layer": keep_ratios,
        "kept_per_layer": new_structure,
        "pruned_per_layer": [int(o - n) for o, n in zip(old_structure, new_structure)],
        "actual_keep_ratio_per_layer": [
            float(n / o) if o > 0 else 0.0 for o, n in zip(old_structure, new_structure)
        ],
        "changed": new_structure != old_structure,
    }


def structured_linear_param_stats(model):
    per_layer = []; total = 0
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


def structured_cnn_param_stats(model):
    per_layer = []; total = 0
    for name, module in model.named_modules():
        if not isinstance(module, (QuantConv2d, QuantLinear)):
            continue
        if not hasattr(module, "weight") or module.weight is None:
            continue
        w = module.weight.detach()
        n = int(w.numel())
        rec = {
            "name": f"{name}.weight",
            "numel": n,
        }
        if isinstance(module, QuantConv2d):
            rec.update({
                "type": "conv",
                "out_channels": int(w.shape[0]),
                "in_channels": int(w.shape[1]),
                "kernel_h": int(w.shape[2]),
                "kernel_w": int(w.shape[3]),
            })
        else:
            rec.update({
                "type": "linear",
                "out_features": int(w.shape[0]),
                "in_features": int(w.shape[1]),
            })
        per_layer.append(rec)
        total += n
    return int(total), per_layer



def make_structured_sweep_vals(cfg, cand, global_vals):
    task_name = cfg["task"]["name"]
    global_vals = [float(x) for x in global_vals]

    if task_name == "jsc_mlp":
        layer_sizes = cand["hidden"]
        n_layers = len(layer_sizes)
        sweep = [[g] * n_layers for g in global_vals]
        if n_layers == 0:
            return [[1.0]]
        hi = global_vals[0]
        mid = global_vals[1] if len(global_vals) > 1 else global_vals[0]
        low = global_vals[-1]
        # add a few automatic per-layer variants
        if n_layers == 1:
            extras = []
        elif n_layers == 2:
            extras = [[hi, low], [mid, low], [low, hi]]
        else:
            mid_idx = n_layers // 2
            def one_high(idx):
                return [hi if j == idx else low for j in range(n_layers)]
            extras = [
                one_high(0),
                one_high(mid_idx),
                one_high(n_layers - 1),
            ]
        sweep.extend(extras)

    elif task_name == "cifar10_cnv":
        hi = global_vals[0]
        mid = global_vals[1] if len(global_vals) > 1 else global_vals[0]
        low = global_vals[-1]
        sweep = [
            [hi,  hi,  hi,  hi,  hi,  hi ],
            [mid, mid, mid, mid, mid, mid],
            [low, low, low, low, low, low],
            [hi,  hi,  mid, mid, low, low],
            [low, low, hi,  hi,  low, low],
            [low, low, mid, mid, hi,  hi ],
        ]
    else:
        raise ValueError(f"Unsupported task for structured sweep: {task_name}")

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
    model = build_model(cfg, cand)
    old_state = torch.load(in_weights, map_location="cpu")
    model.load_state_dict(old_state)
    model.eval()

    n_hidden = len(old_hidden)
    if n_hidden == 0:
        meta = make_structured_meta(cand, "hidden", old_hidden, old_hidden, [])
        return model, meta
    
    keep_ratios = expand_keep_ratios(keep_ratio, n_hidden)

    # no structural pruning requested
    if all(r >= 1.0 for r in keep_ratios):
        meta = make_structured_meta(cand, "hidden", old_hidden, old_hidden, keep_ratios)
        return model, meta

    old_linears = []; old_linear_names = []
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            old_linears.append(module)
            old_linear_names.append(name)

    # decide which neurons to keep in each hidden layer
    keep_indices = []; new_hidden = []
    for i in range(n_hidden):
        cur = old_linears[i]
        nxt = old_linears[i + 1]
        width = int(cur.weight.shape[0])
        keep_k = aligned_keep_count(width, keep_ratios[i], align=align, min_units=min_units)

        # neuron importance = current row L1 + next-layer input-column L1
        cur_score = cur.weight.detach().abs().sum(dim=1)
        nxt_score = nxt.weight.detach().abs().sum(dim=0)
        score = cur_score + nxt_score

        idx = torch.topk(score, k=keep_k, largest=True).indices
        idx = torch.sort(idx).values
        keep_indices.append(idx)
        new_hidden.append(int(keep_k))

    new_cand = dict(cand)
    new_cand["hidden"] = list(new_hidden)

    if new_hidden == old_hidden:
        meta = make_structured_meta(new_cand, "hidden", old_hidden, new_hidden, keep_ratios)
        return model, meta

    # build smaller model
    pruned_model = build_model(cfg, new_cand)
    pruned_model.eval()

    old_bns_all = []; old_bn_names_all = []
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm1d):
            old_bns_all.append(module)
            old_bn_names_all.append(name)

    new_linears = []; new_linear_names = []
    new_bns_all = []; new_bn_names_all = []
    for name, module in pruned_model.named_modules():
        if isinstance(module, QuantLinear):
            new_linears.append(module)
            new_linear_names.append(name)
        elif isinstance(module, nn.BatchNorm1d):
            new_bns_all.append(module)
            new_bn_names_all.append(name)

    # hidden BNs only (skip input BN at index 0)
    old_bns = old_bns_all[1:]
    new_bn_names = new_bn_names_all[1:]

    # keep all matching-shape entries
    new_state = pruned_model.state_dict()
    for k, v in old_state.items():
        if k in new_state and new_state[k].shape == v.shape:
            new_state[k] = v.clone()

    # overwrite QuantLinear weights and biases with sliced tensors
    for i in range(len(old_linears)):
        old_lin = old_linears[i]
        W = old_lin.weight.detach().clone()
        B = old_lin.bias.detach().clone() if old_lin.bias is not None else None

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

    # overwrite hidden BatchNorm state with sliced tensors
    for i in range(n_hidden):
        idx = keep_indices[i]
        weight_key = f"{new_bn_names[i]}.weight"
        if weight_key in new_state and old_bns[i].weight is not None:
            new_state[weight_key] = old_bns[i].weight.detach()[idx].clone()
        bias_key = f"{new_bn_names[i]}.bias"
        if bias_key in new_state and old_bns[i].bias is not None:
            new_state[bias_key] = old_bns[i].bias.detach()[idx].clone()
        rm_key = f"{new_bn_names[i]}.running_mean"
        if rm_key in new_state:
            new_state[rm_key] = old_bns[i].running_mean.detach()[idx].clone()
        rv_key = f"{new_bn_names[i]}.running_var"
        if rv_key in new_state:
            new_state[rv_key] = old_bns[i].running_var.detach()[idx].clone()

        # scalar buffer, keep as-is from old state if present
        nbt_key = f"{new_bn_names[i]}.num_batches_tracked"
        if nbt_key in new_state and nbt_key in old_state:
            new_state[nbt_key] = old_state[nbt_key].clone()

    pruned_model.load_state_dict(new_state, strict=False)
    meta = make_structured_meta(new_cand, "hidden", old_hidden, new_hidden, keep_ratios)
    return pruned_model, meta



def apply_structured_channel_prune(cfg, cand, in_weights, keep_ratio, align=4, min_units=4):
    old_conv_channels = list(cand["conv_channels"])
    model = build_model(cfg, cand)
    old_state = torch.load(in_weights, map_location="cpu")
    model.load_state_dict(old_state)
    model.eval()

    n_conv = len(old_conv_channels)
    if n_conv == 0:
        meta = make_structured_meta(cand, "conv_channels", old_conv_channels, old_conv_channels, [])
        return model, meta

    keep_ratios = expand_keep_ratios(keep_ratio, n_conv)

    # no structural pruning requested
    if all(r >= 1.0 for r in keep_ratios):
        meta = make_structured_meta(cand, "conv_channels", old_conv_channels, old_conv_channels, keep_ratios)
        return model, meta

    old_convs = []; old_conv_names = []
    old_bn2d = []; old_bn2d_names = []
    for name, module in model.named_modules():
        if isinstance(module, QuantConv2d):
            old_convs.append(module)
            old_conv_names.append(name)
        elif isinstance(module, nn.BatchNorm2d):
            old_bn2d.append(module)
            old_bn2d_names.append(name)

    # L1 filter ranking per conv layer
    keep_indices = []; new_conv_channels = []
    for i in range(n_conv):
        conv = old_convs[i]
        width = int(conv.weight.shape[0]) # out_channels
        keep_k = aligned_keep_count(width, keep_ratios[i], align=align, min_units=min_units)

        # score each output channel/filter by L1 norm
        score = conv.weight.detach().abs().sum(dim=(1, 2, 3))
        idx = torch.topk(score, k=keep_k, largest=True).indices
        idx = torch.sort(idx).values
        keep_indices.append(idx)
        new_conv_channels.append(int(keep_k))

    new_cand = dict(cand)
    new_cand["conv_channels"] = list(new_conv_channels)

    if new_conv_channels == old_conv_channels:
        meta = make_structured_meta(new_cand, "conv_channels", old_conv_channels, new_conv_channels, keep_ratios)
        return model, meta

    # build smaller model
    pruned_model = build_model(cfg, new_cand)
    pruned_model.eval()

    new_convs = []; new_conv_names = []
    new_bn2d_names = []
    new_linears = []; new_linear_names = []
    for name, module in pruned_model.named_modules():
        if isinstance(module, QuantConv2d):
            new_convs.append(module)
            new_conv_names.append(name)
        elif isinstance(module, nn.BatchNorm2d):
            new_bn2d_names.append(name)
        elif isinstance(module, QuantLinear):
            new_linears.append(module)
            new_linear_names.append(name)

    # keep all shape-matching entries
    new_state = pruned_model.state_dict()
    for k, v in old_state.items():
        if k in new_state and new_state[k].shape == v.shape:
            new_state[k] = v.clone()

    # overwrite conv weights and biases with sliced tensors
    for i in range(n_conv):
        old_conv = old_convs[i]
        W = old_conv.weight.detach().clone()
        B = old_conv.bias.detach().clone() if old_conv.bias is not None else None

        # previous conv pruned -> slice input channels
        if i > 0:
            prev_idx = keep_indices[i - 1]
            W = W[:, prev_idx, :, :]

        # current conv pruned -> slice output channels
        cur_idx = keep_indices[i]
        W = W[cur_idx, :, :, :]
        if B is not None:
            B = B[cur_idx]

        weight_key = f"{new_conv_names[i]}.weight"
        new_state[weight_key] = W.clone()
        bias_key = f"{new_conv_names[i]}.bias"
        if bias_key in new_state and B is not None:
            new_state[bias_key] = B.clone()

    # overwrite BatchNorm2d parameters following kept conv channels
    for i in range(n_conv):
        idx = keep_indices[i]
        weight_key = f"{new_bn2d_names[i]}.weight"
        if weight_key in new_state and old_bn2d[i].weight is not None:
            new_state[weight_key] = old_bn2d[i].weight.detach()[idx].clone()
        bias_key = f"{new_bn2d_names[i]}.bias"
        if bias_key in new_state and old_bn2d[i].bias is not None:
            new_state[bias_key] = old_bn2d[i].bias.detach()[idx].clone()
        rm_key = f"{new_bn2d_names[i]}.running_mean"
        if rm_key in new_state:
            new_state[rm_key] = old_bn2d[i].running_mean.detach()[idx].clone()
        rv_key = f"{new_bn2d_names[i]}.running_var"
        if rv_key in new_state:
            new_state[rv_key] = old_bn2d[i].running_var.detach()[idx].clone()
        nbt_key = f"{new_bn2d_names[i]}.num_batches_tracked"
        if nbt_key in new_state and nbt_key in old_state:
            new_state[nbt_key] = old_state[nbt_key].clone()

    # the first linear layer input depends on final conv output channels
    if len(new_linears) > 0:
        old_first_linear = None
        old_linear_names = []; old_linears = []
        for name, module in model.named_modules():
            if isinstance(module, QuantLinear):
                old_linears.append(module)
                old_linear_names.append(name)

        old_first_linear = old_linears[0]
        last_old_ch = int(old_conv_channels[-1])
        first_linear_in = int(old_first_linear.weight.shape[1])
        if first_linear_in % last_old_ch != 0:
            raise RuntimeError(f"First linear in_features={first_linear_in} is not divisible by last conv channels={last_old_ch}")

        spatial_mult = first_linear_in // last_old_ch
        last_idx = keep_indices[-1]

        expanded_idx = []
        for ch_idx in last_idx.tolist():
            start = ch_idx * spatial_mult
            expanded_idx.extend(range(start, start + spatial_mult))
        expanded_idx = torch.tensor(expanded_idx, dtype=torch.long)

        W = old_first_linear.weight.detach()[:, expanded_idx].clone()
        B = old_first_linear.bias.detach().clone() if old_first_linear.bias is not None else None

        first_linear_weight_key = f"{new_linear_names[0]}.weight"
        new_state[first_linear_weight_key] = W.clone()
        first_linear_bias_key = f"{new_linear_names[0]}.bias"
        if first_linear_bias_key in new_state and B is not None:
            new_state[first_linear_bias_key] = B.clone()

    pruned_model.load_state_dict(new_state, strict=False)
    meta = make_structured_meta(new_cand, "conv_channels", old_conv_channels, new_conv_channels, keep_ratios)
    return pruned_model, meta



def prune_structured_weights_file(cfg, cand, in_weights, out_weights, keep_ratio, align=4, min_units=4, finetune=True, base_acc=None):
    device = "cuda" if torch.cuda.is_available() and cfg["ea"]["cuda"] else "cpu"
    dl_tr, dl_va = make_dataloaders(cfg, phase="finetune")

    task_name = cfg["task"]["name"]
    if task_name == "jsc_mlp":
        pruned_model, meta = apply_structured_neuron_prune(
            cfg, cand, in_weights, keep_ratio, align=align, min_units=min_units
        )
        stats_fn = structured_linear_param_stats
    elif task_name == "cifar10_cnv":
        pruned_model, meta = apply_structured_channel_prune(
            cfg, cand, in_weights, keep_ratio, align=align, min_units=min_units
        )
        stats_fn = structured_cnn_param_stats
    else:
        raise ValueError(f"Structured pruning not implemented for task: {task_name}")

    # Raw pruned accuracy
    acc_raw = float(eval_acc(pruned_model.to(device), dl_va, device))
    print(f"[PRUNING] keep ratios: {meta['requested_keep_ratio_per_layer']} -> RAW val_acc={acc_raw:.4f}")

    # Optional finetune
    acc_ft = None
    if finetune and meta["changed"]:
        acc_ft = train_finetune(
            cfg, out_weights,
            model=pruned_model,
            dl_tr=dl_tr, dl_va=dl_va
        )
        print(f"[PRUNING] keep ratios: {meta['requested_keep_ratio_per_layer']} -> FINETUNE val_acc={acc_ft:.4f}")
        pruned_model.load_state_dict(torch.load(out_weights, map_location="cpu"))
    else:
        os.makedirs(os.path.dirname(out_weights), exist_ok=True)
        torch.save(pruned_model.state_dict(), out_weights)

    total_params, per_layer = stats_fn(pruned_model)

    summary = {
        "keep_ratio": keep_ratio,
        "align": int(align),
        "min_units": int(min_units),

        **meta,

        "val_acc_after_prune_raw": acc_raw,
        "val_acc_after_finetune": acc_ft,

        "total_structured_weight_params": total_params,
        "per_layer": per_layer,

        "in_weights": str(in_weights),
        "out_weights": str(out_weights),
    }
    return summary