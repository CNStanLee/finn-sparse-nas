import random


def make_ea_ops(cfg):
    task_name = cfg["task"]["name"]
    if task_name == "jsc_mlp":
        return make_mlp_ea_ops(cfg)
    elif task_name == "cifar10_cnv":
        return make_cnv_ea_ops(cfg)
    else:
        raise ValueError(f"Unsupported task name: {task_name}")


def make_mlp_ea_ops(cfg):
    """
    Returns EA helper/operator functions for the MLP config.
    """

    widths = [int(w) for w in cfg["search"]["widths"]]
    qs = cfg["search"]["quant_space"]
    sparsity_cfg = cfg["search"].get("sparsity", {})
    sparsity_targets = [float(x) for x in sparsity_cfg.get("targets", [0.0])]
    minL = int(cfg["search"]["min_layers"])
    maxL = int(cfg["search"]["max_layers"])

    # Search space helpers (candidate generation)

    def random_hidden():
        L = random.randint(minL, maxL)
        return [random.choice(widths) for _ in range(L)]

    def random_quant():
        ia = int(random.choice(qs["act_bits"]))
        ha = int(random.choice(qs["act_bits"]))
        oa_choices = [b for b in qs["out_bits"] if b >= ha]
        oa = int(random.choice(oa_choices)) if oa_choices else ha
        wb = int(random.choice([b for b in qs["weight_bits"] if b < 8] if max(ia, ha, oa) < 8 else qs["weight_bits"]))
        return {"WB": wb, "IA": ia, "HA": ha, "OA": oa}

    def random_sparsity():
        return {"target": float(random.choice(sparsity_targets))}

    def random_cand():
        return {"hidden": random_hidden(), "quant": random_quant(), "sparsity": random_sparsity()}

    def freeze_cand(c):
        # store plain dicts in DF/pickle
        return {
            "hidden": list(c.get("hidden", [])),
            "quant": dict(c.get("quant", {})),
            "sparsity": repair_sparsity(c.get("sparsity", {})),
        }


    # EA operators: repair, crossover, mutation

    def repair_hidden(h):
        if not h:
            h = random_hidden()
        if len(h) < minL:
            h = h + [random.choice(widths) for _ in range(minL - len(h))]
        elif len(h) > maxL:
            h = h[:maxL]
        return h

    def repair_quant(q):
        q = dict(q or {})
        act_bits = [int(b) for b in qs["act_bits"]]
        out_bits = sorted(int(b) for b in qs["out_bits"])
        weight_bits = [int(b) for b in qs["weight_bits"]]

        ia = int(q.get("IA", random.choice(act_bits)))
        ha = int(q.get("HA", random.choice(act_bits)))
        oa = int(q.get("OA", ha))
        wb = int(q.get("WB", random.choice(weight_bits)))

        if oa < ha or oa not in out_bits:
            oa = next((b for b in out_bits if b >= ha), out_bits[-1])
        if max(ia, ha, oa) < 8 and wb >= 8:
            small = [b for b in weight_bits if b < 8]
            if small:
                wb = random.choice(small)
        return {"WB": wb, "IA": ia, "HA": ha, "OA": oa}

    def repair_sparsity(s):
        s = dict(s or {})
        target = float(s.get("target", random.choice(sparsity_targets)))
        if target not in sparsity_targets:
            target = min(sparsity_targets, key=lambda x: abs(x - target))
        return {"target": float(target)}

    def repair_cand(ind):
        ind["hidden"] = repair_hidden(ind.get("hidden", []))
        ind["quant"] = repair_quant(ind.get("quant", {}))
        ind["sparsity"] = repair_sparsity(ind.get("sparsity", {}))
        return ind   

    def cx_cand(ind1, ind2):
        # Hidden crossover: swap suffixes after random cut points
        h1, h2 = ind1.get("hidden", [])[:], ind2.get("hidden", [])[:]
        if len(h1) > 1 and len(h2) > 1:
            c1 = random.randint(1, len(h1) - 1)
            c2 = random.randint(1, len(h2) - 1)
            nh1 = h1[:c1] + h2[c2:]
            nh2 = h2[:c2] + h1[c1:]
        else:
            nh1, nh2 = h1, h2
        ind1["hidden"] = repair_hidden(nh1)
        ind2["hidden"] = repair_hidden(nh2)
        # Quant crossover: uniform swap
        qkeys = ["WB", "IA", "HA", "OA"]
        q1, q2 = dict(ind1.get("quant", {})), dict(ind2.get("quant", {}))
        for k in qkeys:
            if random.random() < 0.5:
                q1[k], q2[k] = q2.get(k, q1.get(k)), q1.get(k, q2.get(k))
        ind1["quant"] = repair_quant(q1)
        ind2["quant"] = repair_quant(q2)
        if random.random() < 0.5:
            s1, s2 = ind1.get("sparsity", {}), ind2.get("sparsity", {})
            ind1["sparsity"], ind2["sparsity"] = repair_sparsity(s2), repair_sparsity(s1)
        return ind1, ind2

    def mut_cand(ind):
        # hidden mutation
        h = ind.get("hidden", [])[:]
        if h and random.random() < 0.6:
            i = random.randrange(len(h))
            h[i] = int(random.choice(widths))
        if random.random() < 0.2 and len(h) < maxL:
            ins = random.randint(0, len(h))
            h.insert(ins, int(random.choice(widths)))
        if random.random() < 0.2 and len(h) > minL:
            del h[random.randrange(len(h))]
        ind["hidden"] = repair_hidden(h)

        # quant mutation
        q = dict(ind.get("quant", {}))
        r = random.random()
        if r < 0.15:
            q = random_quant() # full resample
        elif r < 0.50:
            field = random.choice(["IA", "HA", "OA", "WB"])
            if field == "IA":
                q["IA"] = int(random.choice(qs["act_bits"]))
            elif field == "HA":
                q["HA"] = int(random.choice(qs["act_bits"]))
            elif field == "OA":
                ha = int(q.get("HA", random.choice(qs["act_bits"])))
                valid_out = [b for b in qs["out_bits"] if int(b) >= ha]
                q["OA"] = int(random.choice(valid_out)) if valid_out else ha
            else:  # WB
                ia = int(q.get("IA", random.choice(qs["act_bits"])))
                ha = int(q.get("HA", random.choice(qs["act_bits"])))
                oa = int(q.get("OA", ha))
                if max(ia, ha, oa) < 8:
                    valid_wb = [b for b in qs["weight_bits"] if int(b) < 8] or qs["weight_bits"]
                else:
                    valid_wb = qs["weight_bits"]
                q["WB"] = int(random.choice(valid_wb))
        ind["quant"] = repair_quant(q)

        # sparsity mutation
        if random.random() < 0.35:
            ind["sparsity"] = random_sparsity()
        else:
            ind["sparsity"] = repair_sparsity(ind.get("sparsity", {}))
        return (ind,)
    

    return random_cand, freeze_cand, repair_cand, cx_cand, mut_cand
    


def make_cnv_ea_ops(cfg):
    """
    Returns EA helper/operator functions for the CNV config.
    """

    s = cfg["search"]
    qs = s["quant_space"]
    defaults = cfg["model"]["defaults"]
    stage1_choices = [int(x) for x in s["conv_stage1"]]
    stage2_choices = [int(x) for x in s["conv_stage2"]]
    stage3_choices = [int(x) for x in s["conv_stage3"]]
    fc_choices = [int(x) for x in s["fc_widths"]]
    fixed_in_bits = int(defaults["in_bits"])

    # Search space helpers (candidate generation)

    def stages_to_conv(c1, c2, c3):
        return [int(c1), int(c1), int(c2), int(c2), int(c3), int(c3)]

    def conv_to_stages(conv_channels):
        cc = list(conv_channels or [])
        if len(cc) == 6:
            return [int(cc[0]), int(cc[2]), int(cc[4])]
        if len(cc) == 3:
            return [int(cc[0]), int(cc[1]), int(cc[2])]
        return None

    def random_quant():
        return {
            "WB": int(random.choice(qs["weight_bits"])),
            "AB": int(random.choice(qs["act_bits"])),
            "IB": fixed_in_bits,
        }

    def random_cand():
        c1 = int(random.choice(stage1_choices))
        c2 = int(random.choice(stage2_choices))
        c3 = int(random.choice(stage3_choices))
        f1 = int(random.choice(fc_choices))
        f2 = int(random.choice(fc_choices))
        return {
            "conv_channels": stages_to_conv(c1, c2, c3),
            "fc_features": [f1, f2],
            "quant": random_quant(),
        }

    def freeze_cand(c):
        return {
            "conv_channels": list(c.get("conv_channels", [])),
            "fc_features": list(c.get("fc_features", [])),
            "quant": dict(c.get("quant", {})),
        }
    

    # EA operators: repair, crossover, mutation

    def repair_quant(q):
        q = dict(q or {})
        wb_choices = [int(x) for x in qs["weight_bits"]]
        ab_choices = [int(x) for x in qs["act_bits"]]
        wb = int(q.get("WB", random.choice(wb_choices)))
        ab = int(q.get("AB", random.choice(ab_choices)))
        if wb not in wb_choices:
            wb = int(random.choice(wb_choices))
        if ab not in ab_choices:
            ab = int(random.choice(ab_choices))
        return {"WB": wb, "AB": ab, "IB": fixed_in_bits}

    def repair_conv_channels(conv_channels):
        stages = conv_to_stages(conv_channels)
        if stages is None:
            c1 = int(random.choice(stage1_choices))
            c2 = int(random.choice(stage2_choices))
            c3 = int(random.choice(stage3_choices))
            return stages_to_conv(c1, c2, c3)
        c1, c2, c3 = stages
        if c1 not in stage1_choices:
            c1 = int(random.choice(stage1_choices))
        if c2 not in stage2_choices:
            c2 = int(random.choice(stage2_choices))
        if c3 not in stage3_choices:
            c3 = int(random.choice(stage3_choices))
        return stages_to_conv(c1, c2, c3)

    def repair_fc_features(fc_features):
        ff = list(fc_features or [])
        if len(ff) < 2:
            ff = ff + [int(random.choice(fc_choices)) for _ in range(2 - len(ff))]
        elif len(ff) > 2:
            ff = ff[:2]
        ff = [int(x) if int(x) in fc_choices else int(random.choice(fc_choices)) for x in ff]
        return ff

    def repair_cand(ind):
        ind["conv_channels"] = repair_conv_channels(ind.get("conv_channels", []))
        ind["fc_features"] = repair_fc_features(ind.get("fc_features", []))
        ind["quant"] = repair_quant(ind.get("quant", {}))
        return ind

    def cx_cand(ind1, ind2):
        # conv channels crossover
        s1 = conv_to_stages(ind1.get("conv_channels", []))
        s2 = conv_to_stages(ind2.get("conv_channels", []))
        if s1 is None:
            s1 = conv_to_stages(random_cand()["conv_channels"])
        if s2 is None:
            s2 = conv_to_stages(random_cand()["conv_channels"])
        for i in range(3):
            if random.random() < 0.5:
                s1[i], s2[i] = s2[i], s1[i]
        ind1["conv_channels"] = repair_conv_channels(stages_to_conv(*s1))
        ind2["conv_channels"] = repair_conv_channels(stages_to_conv(*s2))

        # fc features crossover
        f1 = list(ind1.get("fc_features", []))
        f2 = list(ind2.get("fc_features", []))
        f1 = repair_fc_features(f1)
        f2 = repair_fc_features(f2)
        for i in range(2):
            if random.random() < 0.5:
                f1[i], f2[i] = f2[i], f1[i]
        ind1["fc_features"] = repair_fc_features(f1)
        ind2["fc_features"] = repair_fc_features(f2)

        # quant crossover
        q1 = dict(ind1.get("quant", {}))
        q2 = dict(ind2.get("quant", {}))
        for k in ["WB", "AB"]:
            if random.random() < 0.5:
                q1[k], q2[k] = q2.get(k, q1.get(k)), q1.get(k, q2.get(k))
        ind1["quant"] = repair_quant(q1)
        ind2["quant"] = repair_quant(q2)

        return ind1, ind2

    def mut_cand(ind):
        # random mutation
        r = random.random()
        if r < 0.35:
            stages = conv_to_stages(ind.get("conv_channels", []))
            if stages is None:
                stages = conv_to_stages(random_cand()["conv_channels"])
            which = random.choice([0, 1, 2])
            if which == 0:
                stages[0] = int(random.choice(stage1_choices))
            elif which == 1:
                stages[1] = int(random.choice(stage2_choices))
            else:
                stages[2] = int(random.choice(stage3_choices))
            ind["conv_channels"] = repair_conv_channels(stages_to_conv(*stages))
        elif r < 0.65:
            ff = repair_fc_features(ind.get("fc_features", []))
            which = random.choice([0, 1])
            ff[which] = int(random.choice(fc_choices))
            ind["fc_features"] = repair_fc_features(ff)
        elif r < 0.85:
            q = dict(ind.get("quant", {}))
            q["WB"] = int(random.choice(qs["weight_bits"]))
            ind["quant"] = repair_quant(q)
        else:
            q = dict(ind.get("quant", {}))
            q["AB"] = int(random.choice(qs["act_bits"]))
            ind["quant"] = repair_quant(q)
        return (repair_cand(ind),)


    return random_cand, freeze_cand, repair_cand, cx_cand, mut_cand
