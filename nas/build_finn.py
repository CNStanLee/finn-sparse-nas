import argparse, yaml, pathlib, json, torch
from models.brevitas_mlp import JetSubstructureModel
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


def build_model_get_dummy(cfg, cand):
    task_name = cfg["task"]["name"]
    q = cand["quant"]
    if task_name == "jsc_mlp":
        dummy = torch.randn(1, cfg["task"]["in_features"])
        m = JetSubstructureModel(
            cfg["task"]["in_features"], cfg["task"]["n_classes"],
            tuple(cand["hidden"]), q["WB"], q["IA"], q["HA"], q["OA"]
        )
        return m, dummy
    elif task_name == "cifar10_cnv":
        dummy = torch.randn(1, cfg["task"]["in_channels"], cfg["task"]["img_h"], cfg["task"]["img_w"])
        d = cfg["model"]["defaults"]
        m = CIFAR10Model(
            cfg["task"]["n_classes"], cfg["task"]["in_channels"],
            q["WB"], q["AB"], d["in_bits"], tuple(cand["conv_channels"]),
            tuple(cand["fc_features"]), tuple(d["pool_after"]),
            d["kernel_size"], cfg["task"]["img_h"], cfg["task"]["img_w"]
        )
        return m, dummy
    else:
        raise ValueError(f"Unsupported task name: {task_name}")


def qonnx_export(cfg, cand, weights, qonnx):
    qonnx = pathlib.Path(qonnx); qonnx.parent.mkdir(parents=True, exist_ok=True)
    m, dummy = build_model_get_dummy(cfg, cand)
    m.load_state_dict(torch.load(weights, map_location="cpu"))
    m.eval()
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


def add_pre_postproc(qonnx_path):
    qonnx_path = pathlib.Path(qonnx_path)

    model = ModelWrapper(str(qonnx_path))
    model = model.transform(ConvertQONNXtoFINN())
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataTypes())
    model = model.transform(RemoveStaticGraphInputs())

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

    model = model.transform(MergeONNXModels(pre_model))
    global_inp_name = model.graph.input[0].name
    model.set_tensor_datatype(global_inp_name, DataType["UINT8"])

    model = model.transform(InsertTopK(k=1))
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataTypes())
    model = model.transform(RemoveStaticGraphInputs())
    model.save(str(qonnx_path))
    print("[NAS] Added pre/post-processing to:", qonnx_path)


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
    if cfg["task"]["name"] == "cifar10_cnv":
        add_pre_postproc(args.qonnx)

    generate_outputs = parse_outputs(args.outputs)
    shell = getattr(build_cfg.ShellFlowType, cfg["finn"]["shell_flow"])
    out = pathlib.Path(args.build_dir); out.mkdir(parents=True, exist_ok=True)

    build_kwargs = dict(
        output_dir=str(out),
        synth_clk_period_ns=mhz_to_period_ns(float(cfg["finn"]["target_clk_mhz"])),
        board=cfg["finn"]["board"],
        shell_flow_type=shell,
        target_fps=cfg["finn"]["target_fps"],
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
