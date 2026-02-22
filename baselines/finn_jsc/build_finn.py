import yaml, pathlib, argparse, torch
from datetime import datetime
from models.brevitas_mlp import JetSubstructureModel

from brevitas.export import export_qonnx
import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg


def mhz_to_period_ns(clk_mhz: float) -> float:
    return 1000.0 / float(clk_mhz)


def qonnx_export(cfg, arch, qonnx):
    m = JetSubstructureModel(
        cfg["task"]["in_features"], 
        cfg["task"]["n_classes"], 
        tuple(cfg["presets"][arch]["model"]["hidden"]), 
        cfg["presets"][arch]["model"]["weight_bits"], 
        cfg["presets"][arch]["model"]["input_act_bits"],
        cfg["presets"][arch]["model"]["hidden_act_bits"],
        cfg["presets"][arch]["model"]["output_act_bits"]
    )
    m.load_state_dict(torch.load(cfg["presets"][arch]["export"]["pytorch"], map_location="cpu"))
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
    print("[FINN] QONNX exported to:", qonnx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/baseline.yaml")
    ap.add_argument("--arch", choices=["jsc-s","jsc-m","jsc-l"], required=True)
    ap.add_argument("--folding", choices=["latency","resource"], required=True)
    ap.add_argument("--build_dir", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.cfg))
    arch = args.arch ; folding_cfg = args.folding
    qonnx = cfg["presets"][arch]["export"]["qonnx_out"]

    # Export .pt to .qonnx format if not already done
    if not pathlib.Path(qonnx).exists():
        qonnx_export(cfg, arch, qonnx)

    build_dir = pathlib.Path(args.build_dir); build_dir.mkdir(parents=True, exist_ok=True)

    log_path = build_dir / "build_dataflow.log"
    if log_path.exists():
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_path.rename(build_dir / f"build_dataflow.{ts}.log")

    if arch == "jsc-m":
        folding_cfg_file = f"baselines/finn_jsc/folding_cfgs/folding_jsc-s_{folding_cfg}.json"
    else:
        folding_cfg_file = f"baselines/finn_jsc/folding_cfgs/folding_{arch}_{folding_cfg}.json"

    # Build FINN estimate
    build_cfg_obj = build_cfg.DataflowBuildConfig(
        output_dir=str(build_dir),
        synth_clk_period_ns=mhz_to_period_ns(float(cfg["finn"]["target_clk_mhz"])),
        board="Pynq-Z1",
        shell_flow_type=build_cfg.ShellFlowType.VIVADO_ZYNQ,
        folding_config_file=folding_cfg_file,
        generate_outputs=[build_cfg.DataflowOutputType.ESTIMATE_REPORTS],
        save_intermediate_models=True,
        minimize_bit_width=True
    )

    rc = build.build_dataflow_cfg(qonnx, build_cfg_obj)
    if rc != 0: raise SystemExit(rc)
    print("[FINN] Build completed successfully.")


if __name__ == "__main__":
    main()
