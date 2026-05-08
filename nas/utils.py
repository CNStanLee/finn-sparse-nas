import hashlib, json, pathlib, shutil


def cand_to_json(cand: dict):
    return json.dumps(cand, sort_keys=True, separators=(",", ":"))


def cand_hash(cand: dict, cfg: dict):
    key = {
        "cand": cand,
        "finn": {
            "clk_mhz": cfg["finn"]["target_clk_mhz"],
            "target_fps": cfg["finn"].get("target_fps"),
            "board": cfg["finn"]["board"],
            "shell_flow": cfg["finn"]["shell_flow"],
            "folding_config_file": cfg["finn"].get("folding_config_file"),
            "mvau_wwidth_max": cfg["finn"].get("mvau_wwidth_max"),
            "folding_two_pass_relaxation": cfg["finn"].get("folding_two_pass_relaxation"),
        },
        "v": 2,
    }
    s = json.dumps(key, sort_keys=True, separators=(",", ":"))
    return hashlib.md5(s.encode()).hexdigest()


def cand_brief(cand):
    q = cand.get("quant", {})
    sp = cand.get("sparsity", {}).get("target")
    sp_txt = "" if sp is None else f" sp={float(sp):.2f}"
    if "hidden" in cand:
        return f"h={cand.get('hidden', [])} q={q}{sp_txt}"
    if "conv_channels" in cand or "fc_features" in cand:
        return f"conv={cand.get('conv_channels', [])} fc={cand.get('fc_features', [])} q={q}{sp_txt}"
    return str(cand)


def ensure_clean_dir(p):
    p = pathlib.Path(p)
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)
    return p
