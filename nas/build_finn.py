import argparse, yaml, pathlib, json, torch, os
from models.brevitas_mlp import JetSubstructureModel
from brevitas.export import export_qonnx

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg


def parse_outputs(s: str):
    outs = []
    for tok in (t.strip() for t in s.split(",") if t.strip()):
        try:
            outs.append(build_cfg.DataflowOutputType(tok))
        except Exception:
            outs.append(build_cfg.DataflowOutputType[tok.upper()])
    return outs


def mhz_to_period_ns(clk_mhz: float) -> float:
    return 1000.0 / float(clk_mhz)


def qonnx_export(cfg, cand, weights, qonnx):
    qonnx = pathlib.Path(qonnx); qonnx.parent.mkdir(parents=True, exist_ok=True)
    q = cand["quant"]
    m = JetSubstructureModel(
        cfg["task"]["in_features"], cfg["task"]["n_classes"], tuple(cand["hidden"]),
        q["WB"], q["IA"], q["HA"], q["OA"]
    )
    m.load_state_dict(torch.load(weights, map_location="cpu"))
    m.eval()
    dummy = torch.randn(1, cfg["task"]["in_features"])
    export_qonnx(
        m, dummy, qonnx,
        opset_version=13, 
        input_names=["input_0"], 
        output_names=["output_0"],
        dynamic_axes=None, 
        do_constant_folding=True
    )
    if (not qonnx.exists()) or (qonnx.stat().st_size == 0):
        raise RuntimeError(f"QONNX export failed: {qonnx} not created or empty")
    print("[NAS] Exported QONNX ->", qonnx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)                     # nas.yaml
    ap.add_argument("--cand", required=True)                    # path to candidate json
    ap.add_argument("--weights", required=True)                 # path to .pt
    ap.add_argument("--qonnx", required=True)                   # path to qonnx
    ap.add_argument("--build_dir", required=True)               # per-candidate build dir
    ap.add_argument("--outputs", default="estimate_reports")    # comma-separated FINN outputs
    ap.add_argument("--folding_cfg", default=None)              # folding config JSON path
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.cfg))
    cand = json.loads(pathlib.Path(args.cand).read_text(encoding="utf-8"))

    qonnx_export(cfg, cand, args.weights, args.qonnx)

    generate_outputs = parse_outputs(args.outputs)
    shell = getattr(build_cfg.ShellFlowType, cfg["finn"]["shell_flow"])
    out = pathlib.Path(args.build_dir); out.mkdir(parents=True, exist_ok=True)

    build_kwargs = dict(
        output_dir=str(out),
        synth_clk_period_ns=mhz_to_period_ns(float(cfg["finn"]["target_clk_mhz"])),
        board=cfg["finn"]["board"],
        shell_flow_type=shell,
        target_fps=1000,
        generate_outputs=generate_outputs,
        save_intermediate_models=True,
        minimize_bit_width=True,
        enable_build_pdb_debug=False
    )

    if args.folding_cfg is not None and str(args.folding_cfg).strip() != "":
        build_kwargs["folding_config_file"] = args.folding_cfg

    bcfg = build_cfg.DataflowBuildConfig(**build_kwargs)

    rc = build.build_dataflow_cfg(args.qonnx, bcfg)
    if rc != 0: raise SystemExit(rc)
    print("[NAS] FINN Build completed successfully.")


if __name__ == "__main__":
    main()
