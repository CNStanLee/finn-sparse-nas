import json, pathlib, argparse, re


def _read_json(path: pathlib.Path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _read_text(path: pathlib.Path):
    try:
        with open(path, "r") as f:
            return f.read()
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


def _as_float(x):
    try:
        return float(x)
    except Exception:
        return None


def parse_estimate(report_dir):
    net = _read_json(report_dir / "estimate_network_performance.json") or {}
    lyr_res = _read_json(report_dir / "estimate_layer_resources.json") or {}
    op_param = _read_json(report_dir / "op_and_param_counts.json") or {}
    est_lat_ns = net.get("estimated_latency_ns")
    total_res = lyr_res.get("total", {}) if isinstance(lyr_res, dict) else {}
    total_ops = None
    total_params = None
    if isinstance(op_param, dict):
        t = op_param.get("total", {})
        if isinstance(t, dict):
            total_ops = sum(v for k, v in t.items() if k.startswith("op_"))
            total_params = sum(v for k, v in t.items() if k.startswith("param_"))
    return {
        "latency_ns": est_lat_ns,
        "latency_ms": (float(est_lat_ns) / 1e6) if est_lat_ns is not None else None,
        "throughput_fps": net.get("estimated_throughput_fps"),
        "critical_path_cycles": net.get("critical_path_cycles"),
        "max_cycles": net.get("max_cycles"),
        "max_cycles_node": net.get("max_cycles_node_name"),
        "LUT": _as_int(total_res.get("LUT")),
        "BRAM_18K": _as_int(total_res.get("BRAM_18K")),
        "URAM": _as_int(total_res.get("URAM")),
        "DSP": _as_int(total_res.get("DSP")),
        "total_ops": total_ops,
        "total_params": total_params,
    }


def parse_ooc_synthesis(report_dir):
    ooc = _read_json(report_dir / "ooc_synth_and_timing.json") or {}
    if not ooc:
        return {}
    return {
        "LUT": _as_int(ooc.get("LUT")),
        "LUTRAM": _as_int(ooc.get("LUTRAM")),
        "FF": _as_int(ooc.get("FF")),
        "DSP": _as_int(ooc.get("DSP")),
        "BRAM": _as_int(ooc.get("BRAM")),
        "BRAM_18K": _as_int(ooc.get("BRAM_18K")),
        "BRAM_36K": _as_int(ooc.get("BRAM_36K")),
        "URAM": _as_int(ooc.get("URAM")),
        "Carry": _as_int(ooc.get("Carry")),
        "WNS": _as_float(ooc.get("WNS")),
        "Delay": _as_float(ooc.get("Delay")),
        "fmax_mhz": _as_float(ooc.get("fmax_mhz")),
        "estimated_throughput_fps": _as_float(ooc.get("estimated_throughput_fps"))
    }


def parse_rtlsim_performance(report_dir):
    data = _read_json(report_dir / "rtlsim_performance.json") or {}
    if not data:
        return {}
    lat_cycles = data.get("latency_cycles")
    fclk_mhz = data.get("fclk[mhz]")
    lat_ns = None
    lat_ms = None
    if lat_cycles is not None and fclk_mhz not in [None, 0]:
        lat_ns = float(lat_cycles) * (1000.0 / float(fclk_mhz))
        lat_ms = lat_ns / 1e6
    return {
        "latency_cycles": _as_int(lat_cycles),
        "latency_ns": lat_ns,
        "latency_ms": lat_ms,
        "cycles_total": _as_int(data.get("cycles")),
        "throughput_fps": data.get("throughput[images/s]"),
        "stable_throughput_fps": data.get("stable_throughput[images/s]"),
        "fclk_mhz": data.get("fclk[mhz]"),
        "batch_size": _as_int(data.get("N")),
        "n_in_txns": _as_int(data.get("N_IN_TXNS")),
        "n_out_txns": _as_int(data.get("N_OUT_TXNS")),
    }


def parse_post_synth_resources(report_dir):
    post = _read_json(report_dir / "post_synth_resources.json") or {}
    if not post:
        return {}
    top = post.get("(top)", {}) if isinstance(post, dict) else {}
    if not isinstance(top, dict):
        top = {}
    return {
        "LUT": _as_int(top.get("LUT")),
        "SRL": _as_int(top.get("SRL")),
        "FF": _as_int(top.get("FF")),
        "BRAM_36K": _as_int(top.get("BRAM_36K")),
        "BRAM_18K": _as_int(top.get("BRAM_18K")),
        "DSP": _as_int(top.get("DSP")),
    }


def parse_post_route_timing(report_dir):
    txt = _read_text(report_dir / "post_route_timing.rpt")
    if not txt:
        return {}
    out = {}
    if "All user specified timing constraints are met." in txt:
        out["timing_met"] = True
    elif "Timing constraints are not met." in txt:
        out["timing_met"] = False
    m = re.search(
        r"Clock Summary.*?"
        r"Clock\s+Waveform\(ns\)\s+Period\(ns\)\s+Frequency\(MHz\).*?"
        r"[-\s]+\n"
        r"\s*(\S+)\s+\{[^}]+\}\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)",
        txt,
        re.DOTALL,
    )
    if m:
        out["clock_period_ns"] = float(m.group(2))
        out["target_freq_mhz"] = float(m.group(3))
    m = re.search(
        r"Design Timing Summary.*?"
        r"WNS\(ns\).*?[-\s]+\n"
        r"\s*(-?\d+(?:\.\d+)?)",
        txt,
        re.DOTALL,
    )
    if m:
        out["WNS"] = float(m.group(1))
    if "clock_period_ns" in out and "WNS" in out:
        crit_period = out["clock_period_ns"] - out["WNS"]
        if crit_period > 0:
            out["fmax_mhz_approx"] = 1000.0 / crit_period
    return out


def parse_build(build_dir) -> dict:
    build_dir = pathlib.Path(build_dir)
    report_dir = build_dir / "report"

    estimate = parse_estimate(report_dir)
    ooc_synthesis = parse_ooc_synthesis(report_dir)
    rtl_sim = parse_rtlsim_performance(report_dir)

    post_resources = parse_post_synth_resources(report_dir)
    post_timing = parse_post_route_timing(report_dir)

    post_synthesis = {}
    post_synthesis.update(post_resources)
    post_synthesis.update(post_timing)

    return {
        "estimate": estimate,
        "ooc_synthesis": ooc_synthesis,
        "rtl_sim": rtl_sim,
        "post_synthesis": post_synthesis,
    }


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
