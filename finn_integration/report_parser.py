import json, pathlib, argparse


def _read_json(path: pathlib.Path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _as_int(x):
    try:
        return int(x)
    except Exception:
        try:
            return int(float(x))
        except Exception:
            return None


def parse_build(build_dir) -> dict:
    build_dir = pathlib.Path(build_dir)
    report_dir = build_dir / "report"

    # Network-level numbers
    net = _read_json(report_dir / "estimate_network_performance.json") or {}
    est_lat_ns = net.get("estimated_latency_ns")
    summary = {
        "latency_ns": est_lat_ns,
        "latency_ms": (float(est_lat_ns) / 1e6) if est_lat_ns is not None else None,
        "throughput_fps": net.get("estimated_throughput_fps"),
        "critical_path_cycles": net.get("critical_path_cycles"),
        "max_cycles": net.get("max_cycles"),
        "max_cycles_node": net.get("max_cycles_node_name"),
    }

    # Per-layer resources + totals
    lyr_res = _read_json(report_dir / "estimate_layer_resources.json") or {}
    total_res = lyr_res.get("total", {}) if isinstance(lyr_res, dict) else {}
    summary.update({
        "total_LUT": total_res.get("LUT"),
        "total_BRAM_18K": total_res.get("BRAM_18K"),
        "total_URAM": total_res.get("URAM"),
        "total_DSP": total_res.get("DSP"),
    })

    # Per-layer cycles
    lyr_cycles = _read_json(report_dir / "estimate_layer_cycles.json") or {}

    # Ops/params
    op_param = _read_json(report_dir / "op_and_param_counts.json") or {}
    total_ops = None
    total_params = None
    if isinstance(op_param, dict):
        t = op_param.get("total", {})
        if isinstance(t, dict):
            ops_sum = sum(v for k, v in t.items() if k.startswith("op_"))
            params_sum = sum(v for k, v in t.items() if k.startswith("param_"))
            total_ops = ops_sum
            total_params = params_sum
    summary.update({"total_ops": total_ops, "total_params": total_params})

    # PE/SIMD
    hw_cfg = _read_json(build_dir / "final_hw_config.json") or {}

    # Build per-layer list
    layers = []
    if isinstance(lyr_res, dict):
        for name, vals in lyr_res.items():
            if name == "total" or not isinstance(vals, dict):
                continue
            row = {
                "name": name,
                "LUT": _as_int(vals.get("LUT")),
                "BRAM_18K": _as_int(vals.get("BRAM_18K")),
                "URAM": _as_int(vals.get("URAM")),
                "DSP": _as_int(vals.get("DSP")),
                "estimated_cycles": _as_int(lyr_cycles.get(name)),
            }
            # attach PE/SIMD if present
            if isinstance(hw_cfg, dict) and name in hw_cfg and isinstance(hw_cfg[name], dict):
                row["PE"] = _as_int(hw_cfg[name].get("PE"))
                row["SIMD"] = _as_int(hw_cfg[name].get("SIMD"))
            # attach per-layer ops/params if present
            if isinstance(op_param, dict) and name in op_param and isinstance(op_param[name], dict):
                ops_sum = sum(v for k, v in op_param[name].items() if k.startswith("op_"))
                params_sum = sum(v for k, v in op_param[name].items() if k.startswith("param_"))
                row["ops"] = ops_sum
                row["params"] = params_sum
            layers.append(row)

    return {"summary": summary, "layers": layers}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True, help="Path to FINN build directory")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    res = parse_build(args.build)
    outp = args.out or (str(pathlib.Path(args.build) / "report_summary.json"))
    with open(outp, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Wrote {outp}")
