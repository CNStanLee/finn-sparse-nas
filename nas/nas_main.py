import argparse, json, time, os, random, yaml, pathlib, shlex, torch, tempfile, copy
import pandas as pd, numpy as np
from threading import Thread, Lock, Semaphore
from deap import base, creator, tools

from nas.train import train_quick
from nas.utils import cand_to_json, cand_hash, cand_brief, ensure_clean_dir
from nas.ea_ops import make_ea_ops
from nas.unstructured_pruning import prune_unstructured_weights_file, summarize_unstructured_weights_file
from finn_integration.finn_client import run_docker
from finn_integration.report_parser import parse_build


def cand_arch_key(cand):
    """Architecture identity without sparsity target, used for dense-vs-sparse grouping."""
    c = copy.deepcopy(dict(cand))
    c.pop("sparsity", None)
    return json.dumps(c, sort_keys=True, separators=(",", ":"))


def sparsity_target(cand):
    return float(cand.get("sparsity", {}).get("target", 0.0))


def resource_delta(after, before):
    out = {}
    for key in ["LUT", "DSP", "BRAM_18K", "URAM", "latency_ns", "throughput_fps"]:
        a = after.get(key)
        b = before.get(key)
        if a is None or b in [None, 0]:
            out[f"{key}_delta"] = None
            out[f"{key}_reduction_frac"] = None
            continue
        out[f"{key}_delta"] = float(a) - float(b)
        out[f"{key}_reduction_frac"] = (float(b) - float(a)) / float(b)
    return out


def run_finn_build(cfg, finn_cfg, cand_path, wpath, qonnx, bdir, hsh, name_suffix=""):
    max_retries = cfg["finn"]["max_retries"]
    ret = -1
    for attempt in range(1, max_retries + 1):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as td:
            _finn_cfg = copy.deepcopy(finn_cfg)
            _finn_cfg["path"]["tmp"] = td
            cmd = (
                f"python -m nas.build_finn "
                f"--cfg {shlex.quote(cfg['_cfg_path'])} "
                f"--cand {shlex.quote(str(cand_path))} "
                f"--weights {shlex.quote(str(wpath))} "
                f"--qonnx {shlex.quote(str(qonnx))} "
                f"--build_dir {shlex.quote(str(bdir))}"
            )
            cname = f"{hsh}{name_suffix}_att{attempt}"
            ret = run_docker(cmd, _finn_cfg, str(bdir), name=cname)
            if ret == 0:
                break
            print(f"  [NAS] Attempt {attempt}/{max_retries} failed for {hsh}{name_suffix}, "
                  f"rc={ret}" + (" - retrying..." if attempt < max_retries else " - giving up."))
    return ret


def add_storage_reduction(summary, cand):
    q = cand.get("quant", {})
    weight_bits = int(q.get("WB", 1))
    total = int(summary.get("total_params", 0))
    nonzero = int(summary.get("total_nonzero", 0))
    dense_bits = total * weight_bits
    nonzero_bits = nonzero * weight_bits
    summary["weight_bits"] = weight_bits
    summary["dense_weight_bits"] = int(dense_bits)
    summary["nonzero_weight_bits"] = int(nonzero_bits)
    summary["zero_weight_bits"] = int(max(0, dense_bits - nonzero_bits))
    summary["weight_storage_reduction_frac_dense_equivalent"] = (
        0.0 if dense_bits == 0 else float(dense_bits - nonzero_bits) / float(dense_bits)
    )
    return summary


def prepare_sparse_weights(cfg, cand, dense_wpath, sparse_wpath, base_acc):
    target = sparsity_target(cand)
    sp_cfg = cfg["search"].get("sparsity", {})
    min_retain = int(sp_cfg.get("min_retain_per_layer", 64))
    finetune = bool(sp_cfg.get("finetune", False))

    if target > 0.0:
        summary = prune_unstructured_weights_file(
            cfg, cand,
            str(dense_wpath), str(sparse_wpath),
            target_sparsity=target,
            min_retain_per_layer=min_retain,
            finetune=finetune,
            base_acc=base_acc,
        )
        acc = summary.get("val_acc_after_finetune")
        if acc is None:
            acc = summary.get("val_acc_after_prune_raw", base_acc)
    else:
        sparse_wpath.write_bytes(dense_wpath.read_bytes())
        summary = summarize_unstructured_weights_file(cfg, cand, str(sparse_wpath))
        summary["val_acc_after_prune_raw"] = float(base_acc)
        summary["val_acc_after_finetune"] = None
        acc = base_acc

    summary["base_val_acc_before_prune"] = float(base_acc)
    summary["selected_val_acc"] = float(acc)
    summary = add_storage_reduction(summary, cand)
    return float(acc), summary


def write_sparse_search_report(df, run_dir):
    rows = df[df["evaluated"] == True].drop_duplicates("hash", keep="first").copy()
    if rows.empty:
        return

    numeric_cols = [
        "target_sparsity", "achieved_sparsity", "sparse_total_params",
        "sparse_total_nonzero", "weight_storage_reduction_frac",
        "val_acc", "LUT", "DSP", "BRAM_18K", "latency_ns",
    ]
    for col in numeric_cols:
        if col in rows.columns:
            rows[col] = pd.to_numeric(rows[col], errors="coerce")

    baseline_rows = (
        rows.sort_values(["arch_key", "target_sparsity", "achieved_sparsity"])
            .drop_duplicates("arch_key", keep="first")
            .set_index("arch_key")
    )

    out_rows = []
    for _, row in rows.iterrows():
        base = baseline_rows.loc[row["arch_key"]]
        rec = row.to_dict()
        rec["baseline_hash"] = base["hash"]
        rec["baseline_target_sparsity"] = base.get("target_sparsity")
        rec["baseline_achieved_sparsity"] = base.get("achieved_sparsity")
        for key in ["LUT", "DSP", "BRAM_18K", "latency_ns"]:
            a = row.get(key)
            b = base.get(key)
            if pd.isna(a) or pd.isna(b) or float(b) == 0.0:
                rec[f"{key}_vs_baseline_delta"] = np.nan
                rec[f"{key}_vs_baseline_reduction_frac"] = np.nan
            else:
                rec[f"{key}_vs_baseline_delta"] = float(a) - float(b)
                rec[f"{key}_vs_baseline_reduction_frac"] = (float(b) - float(a)) / float(b)
        out_rows.append(rec)

    summary = pd.DataFrame(out_rows)
    summary["cand_brief"] = summary["cand"].apply(cand_brief)
    summary.to_csv(run_dir / "sparse_search_summary.csv", index=False)

    agg_cols = [
        "target_sparsity", "achieved_sparsity", "weight_storage_reduction_frac",
        "val_acc", "LUT", "DSP", "BRAM_18K", "latency_ns",
        "LUT_vs_baseline_reduction_frac", "DSP_vs_baseline_reduction_frac",
        "BRAM_18K_vs_baseline_reduction_frac", "latency_ns_vs_baseline_reduction_frac",
    ]
    agg = (
        summary[agg_cols]
        .groupby("target_sparsity", dropna=False)
        .agg(["count", "mean", "min", "max"])
    )
    agg.to_csv(run_dir / "sparse_search_aggregate.csv")

    json_summary = {
        "note": (
            "Resource reduction is computed against the lowest-sparsity evaluated "
            "candidate with the same architecture and quantization. If only sparse "
            "versions exist for an architecture, the baseline is not a true dense run."
        ),
        "num_evaluated": int(len(summary)),
        "num_arch_groups": int(summary["arch_key"].nunique()),
        "targets": sorted(float(x) for x in summary["target_sparsity"].dropna().unique()),
        "best_by_fitness": summary.sort_values(["fitness", "val_acc"], ascending=[True, False]).head(10)[
            [
                "hash", "cand_brief", "fitness", "val_acc", "target_sparsity",
                "achieved_sparsity", "weight_storage_reduction_frac",
                "LUT", "DSP", "BRAM_18K", "latency_ns",
                "LUT_vs_baseline_reduction_frac",
                "BRAM_18K_vs_baseline_reduction_frac",
            ]
        ].to_dict(orient="records"),
    }
    with open(run_dir / "sparse_search_summary.json", "w") as f:
        json.dump(json_summary, f, indent=2)


def fitness_from_report(rep, cfg, acc):
    W = cfg["fitness"]["weights"]; N = cfg["fitness"]["norm"]
    r = rep["estimate"]
    lut = r.get("LUT", 0)         / max(1, N["lut"])
    dsp = r.get("DSP", 0)         / max(1, N["dsp"])
    bram = r.get("BRAM_18K", 0)   / max(1, N["bram_18k"])
    lat = r.get("latency_ns", 0)  / max(1, N["latency_ns"])
    acc_term = max(0.0, cfg["fitness"]["target_accuracy"] - acc)
    # final fitness: lower is better
    return float(W["acc"] * acc_term) + (W["lut"] * lut) + (W["dsp"] * dsp) + (W["bram"] * bram) + (W["latency"] * lat)


def pareto_objectives_from_row(row, cfg):
    if not bool(row.get("evaluated", True)):
        return (0.0, np.inf, np.inf, np.inf, np.inf)
    N = cfg["fitness"]["norm"]
    acc = float(row["val_acc"])
    lut = float(row["LUT"]) / max(1.0, float(N["lut"])) if pd.notna(row["LUT"]) else np.inf
    dsp = float(row["DSP"]) / max(1.0, float(N["dsp"])) if pd.notna(row["DSP"]) else np.inf
    brm = float(row["BRAM_18K"]) / max(1.0, float(N["bram_18k"])) if pd.notna(row["BRAM_18K"]) else np.inf
    lat = float(row["latency_ns"]) / max(1.0, float(N["latency_ns"])) if pd.notna(row["latency_ns"]) else np.inf
    return (acc, lut, dsp, brm, lat)


def build_task(sem, recs_lock, recs, cfg, finn_cfg, qonnx, bdir, cand, cand_path, wpath, dense_wpath, g, hsh, acc, sparse_summary):
    ret = -1
    with sem:
        ret = run_finn_build(cfg, finn_cfg, cand_path, wpath, qonnx, bdir, hsh)

        if ret == 0:
            rep = parse_build(bdir)
            dense_reference = None
            if (
                bool(cfg["search"].get("sparsity", {}).get("evaluate_dense_reference", False))
                and sparsity_target(cand) > 0.0
            ):
                dense_bdir = pathlib.Path(str(bdir) + "_dense_reference")
                dense_qonnx = pathlib.Path(str(qonnx)).with_name(pathlib.Path(str(qonnx)).stem + "_dense_reference.qonnx")
                rc_dense = run_finn_build(
                    cfg, finn_cfg, cand_path, dense_wpath, dense_qonnx,
                    dense_bdir, hsh, name_suffix="_dense_ref"
                )
                if rc_dense == 0:
                    dense_reference = parse_build(dense_bdir)
                    with open(dense_bdir / "report_summary.json", "w") as f:
                        json.dump(dense_reference, f, indent=2)

            if sparse_summary is not None:
                with open(bdir / "sparsity_summary.json", "w") as f:
                    json.dump(sparse_summary, f, indent=2)
                analysis = {
                    "candidate": cand,
                    "architecture_key": cand_arch_key(cand),
                    "note": (
                        "Resource deltas against dense equivalents require a matching "
                        "target_sparsity=0 candidate in sparse_search_summary.csv. "
                        "This file records per-candidate sparsity/storage effects and FINN estimates."
                    ),
                    "sparsity": sparse_summary,
                    "estimate": rep.get("estimate", {}),
                    "dense_reference_estimate": (dense_reference or {}).get("estimate"),
                    "dense_reference_report": (
                        str(pathlib.Path(str(bdir) + "_dense_reference") / "report_summary.json")
                        if dense_reference is not None else None
                    ),
                    "resource_delta_vs_dense_reference": (
                        resource_delta(rep.get("estimate", {}), dense_reference.get("estimate", {}))
                        if dense_reference is not None else None
                    ),
                }
                with open(bdir / "sparsity_resource_analysis.json", "w") as f:
                    json.dump(analysis, f, indent=2)
            with open(bdir / "report_summary.json", "w") as f:
                json.dump(rep, f, indent=2)
            print("[NAS] Parsed FINN report ->", bdir / "report_summary.json")
            s = rep.get("estimate", {})
            lut  = int(s.get("LUT", 0));      dsp  = int(s.get("DSP", 0))
            bram = int(s.get("BRAM_18K", 0)); lat  = float(s.get("latency_ns", 0))
            fit = fitness_from_report(rep, cfg, acc)
            target_sp = sparsity_target(cand)
            achieved_sp = float((sparse_summary or {}).get("achieved_sparsity", 0.0))
            total_params = int((sparse_summary or {}).get("total_params", 0))
            total_nonzero = int((sparse_summary or {}).get("total_nonzero", 0))
            storage_red = float((sparse_summary or {}).get("weight_storage_reduction_frac_dense_equivalent", 0.0))
            with recs_lock:
                recs.append([
                    g, hsh, cand, cand_arch_key(cand), target_sp, achieved_sp,
                    total_params, total_nonzero, storage_red,
                    fit, acc, lut, dsp, bram, lat,
                    str(bdir / "report_summary.json"),
                    str(bdir / "sparsity_summary.json") if sparse_summary is not None else "",
                    str(bdir / "sparsity_resource_analysis.json") if sparse_summary is not None else "",
                    True
                ])
            print(f" {cand_brief(cand)} -> acc={acc:.3f} fit={fit:.4f}  LUT={lut} DSP={dsp} BRAM={bram} lat_ns={lat}")
        else:
            with recs_lock:
                recs.append([
                    g, hsh, cand, cand_arch_key(cand), sparsity_target(cand), np.nan,
                    np.nan, np.nan, np.nan,
                    np.inf, acc, np.nan, np.nan, np.nan, np.nan, "", "", "", False
                ])

        return ret


def ea_loop(cfg, finn_cfg, run_dir):
    random.seed(cfg["ea"]["seed"]); np.random.seed(cfg["ea"]["seed"]); torch.manual_seed(cfg["ea"]["seed"])

    print(f"\nconfig: {cfg['task']['name'].upper()}")

    # resume log
    pkl = run_dir / pathlib.Path(cfg["ea"]["pickle_path"])
    pkl.parent.mkdir(parents=True, exist_ok=True)

    COLS = [
        "gen","hash","cand","arch_key","target_sparsity","achieved_sparsity",
        "sparse_total_params","sparse_total_nonzero","weight_storage_reduction_frac",
        "fitness","val_acc","LUT","DSP","BRAM_18K","latency_ns",
        "report_path","sparsity_summary_path","sparsity_resource_analysis_path","evaluated"
    ]
    df = pd.read_pickle(pkl) if pkl.exists() else pd.DataFrame(columns=COLS)
    for col in COLS:
        if col not in df.columns:
            df[col] = np.nan
    if not df.empty:
        df["arch_key"] = df.apply(
            lambda r: cand_arch_key(r["cand"]) if pd.isna(r.get("arch_key")) else r.get("arch_key"),
            axis=1,
        )
        df["target_sparsity"] = df.apply(
            lambda r: sparsity_target(r["cand"]) if pd.isna(r.get("target_sparsity")) else r.get("target_sparsity"),
            axis=1,
        )

    # EA helpers / operator functions
    random_cand, freeze_cand, repair_cand, cx_cand, mut_cand = make_ea_ops(cfg)

    # DEAP setup: single-objective EA or multi-objective (Pareto)
    mode = (cfg["ea"].get("mode", "single").lower() == "pareto")
    print(f"\nEA mode selected: {cfg['ea'].get('mode', 'single').upper()}")

    if mode: # pareto
        try:
            creator.FitnessMO
        except AttributeError:
            # maximize acc, minimize lut/dsp/bram/lat
            creator.create("FitnessMO", base.Fitness, weights=(1.0, -1.0, -1.0, -1.0, -1.0))
        try:
            creator.IndividualMO
        except AttributeError:
            creator.create("IndividualMO", dict, fitness=creator.FitnessMO)
        IndCls = creator.IndividualMO
    else: # single
        try:
            creator.FitnessMin
        except AttributeError:
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        try:
            creator.Individual
        except AttributeError:
            creator.create("Individual", dict, fitness=creator.FitnessMin)
        IndCls = creator.Individual

    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, IndCls, random_cand)
    toolbox.register("clone", copy.deepcopy)
    toolbox.register("mate", cx_cand)
    toolbox.register("mutate", mut_cand)
    if not mode:
        toolbox.register("select", tools.selTournament, tournsize=cfg["ea"]["tournsize"])

    def as_ind(c):
        if isinstance(c, IndCls):
            return repair_cand(c)
        return repair_cand(IndCls(copy.deepcopy(dict(c))))

    # init population
    if df.empty:
        pop = [repair_cand(toolbox.individual()) for _ in range(int(cfg["ea"]["pop_size"]))]
        start_gen = 0
    else:
        last_gen = int(df["gen"].max())
        pop = [as_ind(c) for c in df[df["gen"] == last_gen]["cand"].tolist()]
        start_gen = last_gen + 1

    # -------------------------------------------------
    # Generational loop
    # -------------------------------------------------
    for g in range(start_gen, int(cfg["ea"]["generations"])):
        print(f"\n[NAS] Generation {g}")
        recs = []
        recs_lock = Lock()
        threads = []
        sem = Semaphore(4)

        for ind in pop:
            repair_cand(ind)
            hsh = cand_hash(ind, cfg)

            # skip if already evaluated, but still use for selection
            old = df[(df["hash"] == hsh) & (df["evaluated"] == True)]
            if not old.empty:
                row = old.iloc[0]
                with recs_lock:
                    recs.append([
                        g, hsh, row["cand"], row.get("arch_key", cand_arch_key(row["cand"])),
                        row.get("target_sparsity", sparsity_target(row["cand"])),
                        row.get("achieved_sparsity", np.nan),
                        row.get("sparse_total_params", np.nan),
                        row.get("sparse_total_nonzero", np.nan),
                        row.get("weight_storage_reduction_frac", np.nan),
                        row["fitness"], row["val_acc"], row["LUT"], row["DSP"], row["BRAM_18K"],
                        row["latency_ns"], row["report_path"],
                        row.get("sparsity_summary_path", ""),
                        row.get("sparsity_resource_analysis_path", ""),
                        True
                    ])
                print(f"  cached {cand_brief(row['cand'])} -> acc={row['val_acc']:.3f} fit={row['fitness']:.4f}")
                continue

            wrk = run_dir / pathlib.Path(cfg["finn"]["tmp_root"]) / hsh
            ensure_clean_dir(wrk)

            cand_plain = freeze_cand(ind)

            cand_path = wrk / "cand.json"
            cand_path.write_text(cand_to_json(cand_plain), encoding="utf-8")

            # quick training
            dense_wpath = wrk / "cand_dense.pt"
            print(f"Training hash={hsh} {cand_brief(cand_plain)}")
            base_acc = train_quick(cfg, cand_plain, str(dense_wpath))
            wpath = wrk / "cand.pt"
            acc, sparse_summary = prepare_sparse_weights(cfg, cand_plain, dense_wpath, wpath, base_acc)
            (wrk / "sparsity_summary.json").write_text(json.dumps(sparse_summary, indent=2), encoding="utf-8")

            qonnx = wrk / "cand.qonnx"
            bdir = wrk / "build"

            # build FINN concurrently inside container
            thread = Thread(target=build_task, args=(sem, recs_lock, recs, cfg, finn_cfg, qonnx, bdir, cand_plain, cand_path, wpath, dense_wpath, g, hsh, acc, sparse_summary))
            thread.start()
            threads.append(thread)

        for t in threads:
            t.join()

        gen_df = pd.DataFrame(recs, columns=COLS)

        # persist unique best-per-architecture (avoid dupes)
        persist = gen_df.copy() if df.empty else pd.concat([df, gen_df], ignore_index=True)
        if not mode:
            persist = persist.sort_values(["fitness", "val_acc"], ascending=[True, False]).drop_duplicates("hash", keep="first")
        else:
            persist = persist.sort_values(["val_acc"], ascending=[False]).drop_duplicates("hash", keep="first")
        persist.to_pickle(pkl)
        persist.to_csv(run_dir / pathlib.Path(cfg["ea"]["csv_out"]), index=False)
        write_sparse_search_report(persist, run_dir)
        df = persist

        # assign DEAP fitness values for selection (lower is better)
        best_per_hash = (gen_df.sort_values(["fitness", "val_acc"], ascending=[True, False]).drop_duplicates("hash", keep="first"))
        if not mode:
            fit_map = dict(zip(best_per_hash["hash"], best_per_hash["fitness"]))
            for ind in pop:
                hsh = cand_hash(ind, cfg)
                ind.fitness.values = (float(fit_map.get(hsh, np.inf)),)
        else:
            # build objective map for this generation
            obj_map = {row["hash"]: pareto_objectives_from_row(row, cfg) for _, row in best_per_hash.iterrows()}
            for ind in pop:
                hsh = cand_hash(ind, cfg)
                ind.fitness.values = obj_map.get(hsh, (0.0, np.inf, np.inf, np.inf, np.inf))
            pop = tools.selNSGA2(pop, k=len(pop)) # crowding distance + sorting for NSGA-II

        # DEAP reproduction (selection / mate / mutate)
        elite_k = int(cfg["ea"]["elitism"])
        rand_k = int(cfg["ea"]["randoms"])
        pop_size = int(cfg["ea"]["pop_size"])
        n_off = max(0, pop_size - elite_k - rand_k)

        if not mode:
            elites = list(map(toolbox.clone, tools.selBest(pop, k=elite_k))) if elite_k > 0 else []
            offspring = list(map(toolbox.clone, toolbox.select(pop, k=n_off))) if n_off > 0 else []
        else:
            # NSGA-II selection pressure
            elites = list(map(toolbox.clone, pop[:elite_k])) if elite_k > 0 else []
            parents = tools.selNSGA2(pop, k=max(n_off, 0)) if n_off > 0 else []
            offspring = list(map(toolbox.clone, parents))

        # crossover in pairs
        for i in range(1, len(offspring), 2):
            if random.random() < float(cfg["ea"]["cxpb"]):
                toolbox.mate(offspring[i - 1], offspring[i])
                if hasattr(offspring[i - 1].fitness, "values"):
                    del offspring[i - 1].fitness.values
                if hasattr(offspring[i].fitness, "values"):
                    del offspring[i].fitness.values

        # mutation
        for i in range(len(offspring)):
            if random.random() < float(cfg["ea"]["mutpb"]):
                toolbox.mutate(offspring[i])
                if hasattr(offspring[i].fitness, "values"):
                    del offspring[i].fitness.values

        # random immigrants
        rands = [repair_cand(toolbox.individual()) for _ in range(rand_k)] if rand_k > 0 else []

        # dedupe + pad
        def uniq_pad(pop_list, target_size):
            seen, out = set(), []
            for ind in pop_list:
                repair_cand(ind)
                h = cand_hash(ind, cfg)
                if h in seen:
                    continue
                seen.add(h)
                out.append(ind)
                if len(out) == target_size:
                    return out
            while len(out) < target_size:
                ind = repair_cand(toolbox.individual())
                h = cand_hash(ind, cfg)
                if h in seen:
                    continue
                seen.add(h)
                out.append(ind)
            return out

        pop = uniq_pad(elites + offspring + rands, pop_size)


    if not mode:
        print("\n[NAS] Top 10:")
        top = (df[df["evaluated"] == True]
                .sort_values(["fitness", "val_acc"], ascending=[True, False])
                .drop_duplicates("hash", keep="first")
                .head(10).copy().reset_index(drop=True))
        top["cand_brief"] = top["cand"].apply(cand_brief)
        print(top[["cand_brief","fitness","val_acc","target_sparsity","achieved_sparsity","LUT","DSP","BRAM_18K","latency_ns"]])
    else:
        print("\n[NAS] Pareto front (first nondominated set):")
        rows = df[df["evaluated"] == True].drop_duplicates("hash", keep="first").copy()
        inds = []
        for _, r in rows.iterrows():
            ind = IndCls(copy.deepcopy(dict(r["cand"])))
            ind.fitness.values = pareto_objectives_from_row(r, cfg)
            inds.append(ind)
        front = tools.sortNondominated(inds, k=len(inds), first_front_only=True)[0]
        front_rows = []
        for ind in front:
            hsh = cand_hash(ind, cfg)
            rr = rows[rows["hash"] == hsh].iloc[0]
            front_rows.append(rr)
        pf = pd.DataFrame(front_rows).drop_duplicates("hash", keep="first")
        pf = pf.sort_values(["val_acc"], ascending=False).reset_index(drop=True)
        pf["cand_brief"] = pf["cand"].apply(cand_brief)
        pf[["hash","cand","val_acc","target_sparsity","achieved_sparsity","LUT","DSP","BRAM_18K","latency_ns"]].to_csv(run_dir / "pareto_front.csv", index=False)
        print(pf[["cand_brief","val_acc","target_sparsity","achieved_sparsity","LUT","DSP","BRAM_18K","latency_ns"]].head(15))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/nas_mlp.yaml")
    ap.add_argument("--finn-cfg", default="configs/finn.yaml")
    ap.add_argument("--run-id", default=None, help="Directory name for this run")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.cfg))
    cfg["_cfg_path"] = args.cfg
    finn_cfg = yaml.safe_load(open(args.finn_cfg))

    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    run_dir = pathlib.Path(cfg["ea"]["root_path"]) / run_id
    os.makedirs(run_dir, exist_ok=True)

    ea_loop(cfg, finn_cfg, run_dir)


if __name__ == "__main__":
    main()
