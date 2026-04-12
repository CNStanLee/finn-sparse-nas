import hashlib, json, pathlib, shutil


def cand_to_json(cand: dict):
    return json.dumps(cand, sort_keys=True, separators=(",", ":"))


def cand_hash(cand: dict, cfg: dict):
    key = {
        "cand": cand,
        "finn": {
            "clk_mhz": cfg["finn"]["target_clk_mhz"],
            "board": cfg["finn"]["board"],
            "shell_flow": cfg["finn"]["shell_flow"],
        },
        "v": 1,
    }
    s = json.dumps(key, sort_keys=True, separators=(",", ":"))
    return hashlib.md5(s.encode()).hexdigest()


def cand_brief(cand):
    q = cand.get("quant", {})
    if "hidden" in cand:
        return f"h={cand.get('hidden', [])} q={q}"
    if "conv_channels" in cand or "fc_features" in cand:
        return f"conv={cand.get('conv_channels', [])} fc={cand.get('fc_features', [])} q={q}"
    return str(cand)


def ensure_clean_dir(p):
    p = pathlib.Path(p)
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)
    return p
