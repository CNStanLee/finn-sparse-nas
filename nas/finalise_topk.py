import argparse, json, pathlib, shlex, tempfile, copy, yaml
import pandas as pd
from nas.train import train_full
from nas.utils import cand_to_json, cand_hash, ensure_clean_dir
from finn_integration.finn_client import run_docker


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--finn-cfg", required=True)
    ap.add_argument("--run-dir", required=True, help="results/nas/<run_id>")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--outputs", default="estimate_reports,bitfile,pynq_driver,deployment_package")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.cfg))
    cfg["_cfg_path"] = args.cfg
    finn_cfg = yaml.safe_load(open(args.finn_cfg))

    run_dir = pathlib.Path(args.run_dir)
    pkl = run_dir / pathlib.Path(cfg["ea"]["pickle_path"])
    df = pd.read_pickle(pkl)

    top = (df[df["evaluated"] == True]
           .sort_values(["fitness", "val_acc"], ascending=[True, False])
           .drop_duplicates("hash", keep="first")
           .head(args.top_k)
           .reset_index(drop=True))

    final_dir = run_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    for i, row in top.iterrows():
        cand = row["cand"]
        hsh = row["hash"]
        wrk = final_dir / f"{i:02d}_{hsh}"
        ensure_clean_dir(wrk)

        cand_path = wrk / "cand.json"
        cand_path.write_text(cand_to_json(cand), encoding="utf-8")

        wpath = wrk / "cand_full.pt"
        print(f"[FINAL] Training {i}/{args.top_k} hash={hsh} cand={cand}")
        acc = train_full(cfg, cand, str(wpath))
        print(f"[FINAL] full_train val_acc={acc:.4f}")

        qonnx = wrk / "cand.qonnx"
        bdir = wrk / "build"

        with tempfile.TemporaryDirectory(dir="/dev/shm") as td:
            _finn_cfg = copy.deepcopy(finn_cfg)
            _finn_cfg["path"]["tmp"] = td

            cmd = (
                f"python -m nas.build_finn "
                f"--cfg {shlex.quote(cfg['_cfg_path'])} "
                f"--cand {shlex.quote(str(cand_path))} "
                f"--weights {shlex.quote(str(wpath))} "
                f"--qonnx {shlex.quote(str(qonnx))} "
                f"--build_dir {shlex.quote(str(bdir))} "
                f"--outputs {shlex.quote(args.outputs)}"
            )
            rc = run_docker(cmd, _finn_cfg, str(bdir), name=f"final_{hsh}")
            if rc != 0:
                print(f"[FINAL] FINN build failed for {hsh} (rc={rc})")
            else:
                print(f"[FINAL] FINN build OK for {hsh} -> {bdir}")


if __name__ == "__main__":
    main()