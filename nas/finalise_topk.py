import argparse, json, pathlib, shlex, tempfile, copy, yaml, torch
import pandas as pd
from nas.train import train_full, make_test_dataloader, build_jsc_model, eval_acc
from nas.utils import cand_to_json, ensure_clean_dir
from nas.unstructured_pruning import prune_unstructured_weights_file
from nas.structured_pruning import prune_structured_neurons_weights_file, make_structured_sweep_vals
from finn_integration.finn_client import run_docker
from finn_integration.report_parser import parse_build


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--finn-cfg", required=True)
    ap.add_argument("--run-dir", required=True, help="results/nas/<run_id>")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--pruning-mode", choices=["baseline", "unstructured", "structured"], default="baseline")
    ap.add_argument("--outputs", default="estimate_reports,stitched_ip,rtlsim_performance,ooc_synth,bitfile")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.cfg))
    cfg["_cfg_path"] = args.cfg
    finn_cfg = yaml.safe_load(open(args.finn_cfg))

    device = "cuda" if torch.cuda.is_available() and cfg["ea"]["cuda"] else "cpu"
    dl_te = make_test_dataloader(cfg)

    run_dir = pathlib.Path(args.run_dir)
    pkl = run_dir / pathlib.Path(cfg["ea"]["pickle_path"])
    df = pd.read_pickle(pkl)

    top = (df[df["evaluated"] == True]
           .sort_values(["fitness", "val_acc"], ascending=[True, False])
           .drop_duplicates("hash", keep="first")
           .head(args.top_k)
           .reset_index(drop=True))

    final_root = run_dir / "final"
    final_dir = final_root / args.pruning_mode
    ensure_clean_dir(final_dir)
    results_rows = []

    for i, row in top.iterrows():
        cand = row["cand"]
        hsh = row["hash"]
        wrk = final_dir / f"{i+1:02d}_{hsh}"
        wrk.mkdir(parents=True, exist_ok=True)

        cand_path = wrk / "cand.json"
        cand_path.write_text(cand_to_json(cand), encoding="utf-8")

        base_wpath = wrk / "cand_full.pt"
        print(f"\n[FINAL] Training {i+1}/{args.top_k} hash={hsh} cand={cand}")
        acc = train_full(cfg, cand, str(base_wpath))
        print(f"[FINAL] full_train val_acc={acc:.4f}")

        if args.pruning_mode == "structured":
            sweep_vals = make_structured_sweep_vals(cand["hidden"], cfg["finalists"]["pruning"]["structured"])
        else:
            sweep_vals = cfg["finalists"]["pruning"][args.pruning_mode]

        for val in sweep_vals:
            if args.pruning_mode == "structured":
                tag = "kr_" + "_".join(f"{int(round(float(x) * 100)):03d}" for x in val)
            elif args.pruning_mode == "unstructured":
                tag = f"sp_{int(round(val * 100)):03d}"
            else:
                tag = "baseline"
            
            wrk_sp = wrk if args.pruning_mode == "baseline" else wrk / tag
            wrk_sp.mkdir(parents=True, exist_ok=True)

            # defaults
            wpath = base_wpath
            cand_for_build = cand
            used_folding_cfg = None

            if args.pruning_mode == "baseline":
                prune_stats = {"note": "no pruning applied",}

            elif args.pruning_mode == "unstructured":
                wpath = wrk_sp / "cand_pruned.pt"
                used_folding_cfg = "configs/folding/unstructured_internal_embedded.json"
                prune_stats = prune_unstructured_weights_file(
                    cfg, cand,
                    str(base_wpath), str(wpath),
                    target_sparsity=val,
                    finetune=True
                )

            else:  # structured
                wpath = wrk_sp / "cand_pruned.pt"
                prune_stats = prune_structured_neurons_weights_file(
                    cfg, cand,
                    str(base_wpath), str(wpath),
                    keep_ratio=val,
                    align=4,
                    min_units=4,
                    finetune=True
                )
                cand_for_build = prune_stats["new_cand"]

            cand_path = wrk_sp / "cand.json"
            cand_path.write_text(cand_to_json(cand_for_build), encoding="utf-8")
            
            test_model = build_jsc_model(cfg, cand_for_build)
            test_model.load_state_dict(torch.load(str(wpath), map_location="cpu"))
            test_model = test_model.to(device)
            final_test_acc = float(eval_acc(test_model, dl_te, device))
            print(f"[FINAL] {tag} -> TEST acc={final_test_acc:.4f}")

            finalist_summary = {
                "hash": hsh,
                "orig_cand": cand,
                "val_acc_before": float(acc),
                "pruning_mode": args.pruning_mode,
                "pruning": prune_stats,
                "final_test_acc": final_test_acc,
                "finn": None
            }

            qonnx = wrk_sp / "cand.qonnx"; bdir = wrk_sp / "build"
            max_retries = cfg["finn"]["max_retries"]; rc = -1

            for attempt in range(1, max_retries + 1):
                with tempfile.TemporaryDirectory(dir="/dev/shm") as td:
                    _finn_cfg = copy.deepcopy(finn_cfg)
                    _finn_cfg["path"]["tmp"] = td
                    cmd_parts = [
                        "python -m nas.build_finn",
                        f"--cfg {shlex.quote(cfg['_cfg_path'])}",
                        f"--cand {shlex.quote(str(cand_path))}",
                        f"--weights {shlex.quote(str(wpath))}",
                        f"--qonnx {shlex.quote(str(qonnx))}",
                        f"--build_dir {shlex.quote(str(bdir))}",
                        f"--outputs {shlex.quote(args.outputs)}",
                    ]
                    if used_folding_cfg is not None:
                        cmd_parts.append(f"--folding_cfg {shlex.quote(str(used_folding_cfg))}")
                    cmd = " ".join(cmd_parts)
                    rc = run_docker(cmd, _finn_cfg, str(bdir), name=f"final_{hsh}_{tag}_att{attempt}")
                    if rc != 0:
                        print(f"[FINAL] Attempt {attempt}/{max_retries} failed for {hsh} {tag}, "
                              f"rc={rc}" + (" - retrying..." if attempt < max_retries else " - giving up."))
                    else:
                        print(f"[FINAL] FINN build OK for {hsh} {tag} -> {bdir}")
                        break
            
            post = {}; rtl = {}
            if rc == 0:
                rep = parse_build(bdir)
                finalist_summary["finn"] = rep
                post = rep.get("post_synthesis", {})
                rtl = rep.get("rtl_sim", {})

            with open(wrk_sp / "finalist_summary.json", "w") as f:
                json.dump(finalist_summary, f, indent=2)
            print("[FINAL] Wrote", wrk_sp / "finalist_summary.json")

            bram18 = post.get("BRAM_18K"); bram36 = post.get("BRAM_36K")
            bram_total_18keq = None
            if bram18 is not None or bram36 is not None:
                bram_total_18keq = (0 if bram18 is None else bram18) + 2 * (0 if bram36 is None else bram36)

            row_out = {
                "rank": i + 1,
                "hash": hsh,
                "pruning_mode": args.pruning_mode,
                "tag": tag,
                "val_acc_before": float(acc),
                "final_test_acc": final_test_acc,

                "orig_hidden": str(cand.get("hidden", [])),
                "final_hidden": str(cand_for_build.get("hidden", [])),
                "quant": str(cand_for_build.get("quant", {})),

                "target_sparsity": val if args.pruning_mode == "unstructured" else None,
                "keep_ratio": json.dumps(val) if args.pruning_mode == "structured" else None,

                "LUT": post.get("LUT"),
                "FF": post.get("FF"),
                "DSP": post.get("DSP"),
                "BRAM_18K": bram18,
                "BRAM_36K": bram36,
                "BRAM_total_18Keq" : bram_total_18keq,

                "timing_met": post.get("timing_met"),
                "clock_period_ns": post.get("clock_period_ns"),
                "target_freq_mhz": post.get("target_freq_mhz"),
                "fmax_mhz_approx": post.get("fmax_mhz_approx"),
                
                "latency_cycles_rtlsim": rtl.get("latency_cycles"),
                "latency_ns_rtlsim": rtl.get("latency_ns"),
                "throughput_fps_rtlsim": rtl.get("throughput_fps"),
            }
            results_rows.append(row_out)
            pd.DataFrame(results_rows).to_csv(final_dir / "test_results.csv", index=False)
            print("[FINAL] Updated", final_dir / "test_results.csv")


if __name__ == "__main__":
    main()