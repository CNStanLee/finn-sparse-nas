import argparse, yaml, shlex, pathlib, json
from baselines.finn_jsc.train import train
from finn_integration.finn_client import run_docker
from finn_integration.report_parser import parse_build


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/baseline.yaml")
    ap.add_argument("--finn-cfg", default="configs/finn.yaml")
    ap.add_argument("--arch", choices=["jsc-s","jsc-m","jsc-l"], required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.cfg))
    finn_cfg = yaml.safe_load(open(args.finn_cfg))

    train(cfg, args.arch)

    for folding in ["latency","resource"]:
        bdir = pathlib.Path(cfg["presets"][args.arch]["export"]["build_dir"] + f"_{folding}")
        bdir.mkdir(parents=True, exist_ok=True)

        cmd = (
            f"python -m baselines.finn_jsc.build_finn "
            f"--cfg {shlex.quote(args.cfg)} "
            f"--arch {shlex.quote(args.arch)} "
            f"--folding {shlex.quote(folding)} "
            f"--build_dir {shlex.quote(str(bdir))}"
        )
        ret = run_docker(cmd, finn_cfg, str(bdir))

        if ret == 0:
            print(f"[FINN] Build completed successfully: {args.arch} + {folding}")

            rep = parse_build(bdir)
            with open(bdir / "report_summary.json", "w") as f:
                json.dump(rep, f, indent=2)
            print("Parsed report summary:", bdir / "report_summary.json")


if __name__ == "__main__":
    main()
