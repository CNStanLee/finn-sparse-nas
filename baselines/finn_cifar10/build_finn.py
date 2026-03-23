import yaml, pathlib, argparse, torch
from models.brevitas_cnv import CIFAR10Model

from brevitas.export import export_qonnx
import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg

from finn.util.pytorch import ToTensor
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.datatype import DataType
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.general import (GiveReadableTensorNames, GiveUniqueNodeNames, RemoveStaticGraphInputs)
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.insert_topk import InsertTopK
from qonnx.transformation.merge_onnx_models import MergeONNXModels
from qonnx.util.cleanup import cleanup as qonnx_cleanup


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


def qonnx_export(cfg, arch, qonnx):
    qonnx = pathlib.Path(qonnx); qonnx.parent.mkdir(parents=True, exist_ok=True)
    p = cfg["presets"][arch]["model"]
    m= CIFAR10Model(
        n_classes=cfg["task"]["n_classes"],
        in_ch=cfg["task"]["in_channels"],
        weight_bits=p["weight_bits"],
        act_bits=p["act_bits"],
        in_bits=p["input_bits"],
        img_h=cfg["task"]["img_h"],
        img_w=cfg["task"]["img_w"],
    )
    m.load_state_dict(torch.load(cfg["presets"][arch]["export"]["pytorch"], map_location="cpu"))
    m.eval()
    dummy = torch.rand(1, cfg["task"]["in_channels"], cfg["task"]["img_h"], cfg["task"]["img_w"])
    export_qonnx(
        m, dummy, qonnx,
        opset_version=13,
        input_names=["input_0"],
        output_names=["output_0"],
        dynamic_axes=None,
        do_constant_folding=True,
    )
    if (not qonnx.exists()) or (qonnx.stat().st_size == 0):
        raise RuntimeError(f"QONNX export failed: {qonnx} not created or empty")
    print("[FINN] QONNX exported to:", qonnx)


def add_pre_postproc(qonnx_path):
    qonnx_path = pathlib.Path(qonnx_path)

    # load exported model and do the same initial tidy-up FINN uses
    model = ModelWrapper(str(qonnx_path))
    model = model.transform(ConvertQONNXtoFINN())
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataTypes())
    model = model.transform(RemoveStaticGraphInputs())

    # preprocessing: raw uint8 image -> divide by 255
    global_inp_name = model.graph.input[0].name
    ishape = model.get_tensor_shape(global_inp_name)

    preproc_path = qonnx_path.with_name(qonnx_path.stem + "_preproc.onnx")
    preproc = ToTensor()

    export_qonnx(preproc, torch.randn(ishape), str(preproc_path), opset_version=13)
    qonnx_cleanup(str(preproc_path), out_file=str(preproc_path))

    pre_model = ModelWrapper(str(preproc_path))
    pre_model = pre_model.transform(ConvertQONNXtoFINN())
    pre_model = pre_model.transform(InferShapes())
    pre_model = pre_model.transform(FoldConstants())

    # merge preprocessing in front of the CNV
    model = model.transform(MergeONNXModels(pre_model))

    # mark external input as UINT8
    global_inp_name = model.graph.input[0].name
    model.set_tensor_datatype(global_inp_name, DataType["UINT8"])

    # postprocessing: logits -> top-1 class index
    model = model.transform(InsertTopK(k=1))

    # tidy again after graph edits
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataTypes())
    model = model.transform(RemoveStaticGraphInputs())

    model.save(str(qonnx_path))
    print("[FINN] Added pre/post-processing to:", qonnx_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/baseline_cnv.yaml")
    ap.add_argument("--arch", choices=["cnv_1w1a", "cnv_1w2a", "cnv_2w2a"], required=True)
    ap.add_argument("--build_dir", required=True)
    ap.add_argument("--outputs", default="estimate_reports,stitched_ip,rtlsim_performance,ooc_synth,bitfile")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.cfg))
    arch = args.arch
    qonnx = cfg["presets"][arch]["export"]["qonnx_out"]

    qonnx_export(cfg, arch, qonnx)
    add_pre_postproc(qonnx)

    generate_outputs = parse_outputs(args.outputs)
    shell = getattr(build_cfg.ShellFlowType, cfg["finn"]["shell_flow"])
    build_dir = pathlib.Path(args.build_dir); build_dir.mkdir(parents=True, exist_ok=True)

    build_cfg_obj = build_cfg.DataflowBuildConfig(
        output_dir=str(build_dir),
        synth_clk_period_ns=mhz_to_period_ns(float(cfg["finn"]["target_clk_mhz"])),
        board=cfg["finn"]["board"],
        shell_flow_type=shell,
        folding_config_file=cfg["presets"][arch]["folding"],
        generate_outputs=generate_outputs,
        save_intermediate_models=True,
        specialize_layers_config_file=cfg["finn"]["specialize_layers_config"],
        minimize_bit_width=True,
        enable_build_pdb_debug=False
    )

    rc = build.build_dataflow_cfg(qonnx, build_cfg_obj)
    if rc != 0: raise SystemExit(rc)
    print("[FINN] Build completed successfully.")


if __name__ == "__main__":
    main()