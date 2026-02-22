import random

def make_ea_ops(cfg):
    """
    Returns EA helper/operator functions.
    """

    widths = [int(w) for w in cfg["search"]["widths"]]
    qs = cfg["search"]["quant_space"]
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

    def random_cand():
        return {"hidden": random_hidden(), "quant": random_quant()}

    def freeze_cand(c):
        # store plain dicts in DF/pickle
        return {"hidden": list(c.get("hidden", [])), "quant": dict(c.get("quant", {}))}


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

    def repair_cand(ind):
        ind["hidden"] = repair_hidden(ind.get("hidden", []))
        ind["quant"] = repair_quant(ind.get("quant", {}))
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
        return (ind,)
    

    return random_cand, freeze_cand, repair_cand, cx_cand, mut_cand
    
