import argparse, yaml, shlex, pathlib, json, tempfile, copy, shutil
from baselines.finn_jsc.train import train
from finn_integration.finn_client import run_docker
from finn_integration.report_parser import parse_build


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/baseline_mlp.yaml")
    ap.add_argument("--finn-cfg", default="configs/finn.yaml")
    ap.add_argument("--arch", choices=["jsc-s","jsc-m","jsc-l"], required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.cfg))
    finn_cfg = yaml.safe_load(open(args.finn_cfg))

    p = pathlib.Path(f"results/baseline/{args.arch}")
    if p.exists():
        shutil.rmtree(p)

    train(cfg, args.arch)

    for folding in ["latency","resource"]:
        bdir = pathlib.Path(cfg["presets"][args.arch]["export"]["build_dir"] + f"_{folding}")
        bdir.mkdir(parents=True, exist_ok=True)
        max_retries = cfg["finn"]["max_retries"]; rc = -1

        for attempt in range(1, max_retries + 1):
            with tempfile.TemporaryDirectory(dir="/dev/shm") as td:
                _finn_cfg = copy.deepcopy(finn_cfg)
                _finn_cfg["path"]["tmp"] = td
                cmd = (
                    f"python -m baselines.finn_jsc.build_finn "
                    f"--cfg {shlex.quote(args.cfg)} "
                    f"--arch {shlex.quote(args.arch)} "
                    f"--folding {shlex.quote(folding)} "
                    f"--build_dir {shlex.quote(str(bdir))}"
                )
                rc = run_docker(cmd, _finn_cfg, str(bdir), name=f"{args.arch}_{folding}_att{attempt}")
                if rc != 0:
                    print(f"[FINN] Attempt {attempt}/{max_retries} failed for {args.arch} {folding}, "
                            f"rc={rc}" + (" - retrying..." if attempt < max_retries else " - giving up."))
                else:
                    print(f"[FINN] FINN build OK for {args.arch} + {folding} -> {bdir}")
                    break

        if rc == 0:
            rep = parse_build(bdir)
            with open(bdir / "report_summary.json", "w") as f:
                json.dump(rep, f, indent=2)
            print("Parsed report summary:", bdir / "report_summary.json")


if __name__ == "__main__":
    main()
