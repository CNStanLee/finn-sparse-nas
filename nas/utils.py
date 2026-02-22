import hashlib, json, pathlib, shutil


def cand_to_json(cand: dict):
    return json.dumps(cand, sort_keys=True, separators=(",", ":"))


def cand_hash(cand: dict, cfg: dict):
    key = {
        "cand": cand,
        "finn": {
            "folding": cfg["finn"]["folding"],
            "clk_mhz": cfg["finn"]["target_clk_mhz"],
            "board": cfg["finn"]["board"],
            "shell_flow": cfg["finn"]["shell_flow"],
        },
        "v": 1,
    }
    s = json.dumps(key, sort_keys=True, separators=(",", ":"))
    return hashlib.md5(s.encode()).hexdigest()


def ensure_clean_dir(p):
    p = pathlib.Path(p)
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)
    return p
