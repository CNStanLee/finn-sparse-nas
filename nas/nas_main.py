import argparse, json, time, os, random, yaml, pathlib, shlex, torch, tempfile, copy
import pandas as pd, numpy as np
from threading import Thread, Lock, Semaphore
from deap import base, creator, tools

from nas.train import train_quick
from nas.utils import cand_to_json, cand_hash, ensure_clean_dir
from nas.ea_ops import make_ea_ops
from finn_integration.finn_client import run_docker
from finn_integration.report_parser import parse_build


def fitness_from_report(rep, cfg, acc):
    W = cfg["fitness"]["weights"]; N = cfg["fitness"]["norm"]
    r = rep["summary"]
    lut = r.get("total_LUT", 0)         / max(1, N["lut"])
    dsp = r.get("total_DSP", 0)         / max(1, N["dsp"])
    bram = r.get("total_BRAM_18K", 0)   / max(1, N["bram_18k"])
    lat = r.get("latency_ns", 0)        / max(1, N["latency_ns"])
    acc_term = 0.0 if acc >= cfg["fitness"]["min_acc"] else max(0.0, cfg["fitness"]["target_accuracy"] - acc)
    # final fitness: lower is better
    return float(W["acc"] * acc_term) + (W["lut"] * lut) + (W["dsp"] * dsp) + (W["bram"] * bram) + (W["latency"] * lat)


def build_task(sem, recs_lock, recs, cfg, finn_cfg, qonnx, bdir, cand, cand_path, wpath, g, hsh, acc):
    with sem:
        with tempfile.TemporaryDirectory(dir="/dev/shm") as td:
            # export QONNX and build FINN inside container
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
            ret = run_docker(cmd, _finn_cfg, str(bdir), name=str(hsh))

            if ret == 0:
                rep = parse_build(bdir)
                with open(bdir / "report_summary.json", "w") as f:
                    json.dump(rep, f, indent=2)
                print("[NAS] Parsed FINN report ->", bdir / "report_summary.json")

                s = rep.get("summary", {})
                lut  = int(s.get("total_LUT", 0));      dsp  = int(s.get("total_DSP", 0))
                bram = int(s.get("total_BRAM_18K", 0)); lat  = float(s.get("latency_ns", 0))

                fit = fitness_from_report(rep, cfg, acc)
                with recs_lock:
                    recs.append([g, hsh, cand, fit, acc, lut, dsp, bram, lat, str(bdir / "report_summary.json"), True])
                print(f" {cand['hidden']} {cand['quant']} -> acc={acc:.3f} fit={fit:.4f}  LUT={lut} DSP={dsp} BRAM={bram} lat_ns={lat}")
            else:
                with recs_lock:
                    recs.append([g, hsh, cand, np.inf, acc, np.nan, np.nan, np.nan, np.nan, "", False])

            return ret


def ea_loop(cfg, finn_cfg, run_dir):
    random.seed(cfg["ea"]["seed"]); np.random.seed(cfg["ea"]["seed"]); torch.manual_seed(cfg["ea"]["seed"])

    # resume log
    pkl = run_dir / pathlib.Path(cfg["ea"]["pickle_path"])
    pkl.parent.mkdir(parents=True, exist_ok=True)

    COLS = ["gen","hash","cand","fitness","val_acc","LUT","DSP","BRAM_18K","latency_ns","report_path","evaluated"]
    df = pd.read_pickle(pkl) if pkl.exists() else pd.DataFrame(columns=COLS)

    # EA helpers / operator functions
    random_cand, freeze_cand, repair_cand, cx_cand, mut_cand = make_ea_ops(cfg)

    # DEAP setup
    try:
        creator.FitnessMin
    except AttributeError:
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    try:
        creator.Individual
    except AttributeError:
        creator.create("Individual", dict, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.Individual, random_cand)
    toolbox.register("clone", copy.deepcopy)
    toolbox.register("mate", cx_cand)
    toolbox.register("mutate", mut_cand)
    toolbox.register("select", tools.selTournament, tournsize=cfg["ea"]["tournsize"])

    def as_ind(c):
        if isinstance(c, creator.Individual):
            return repair_cand(c)
        return repair_cand(creator.Individual(copy.deepcopy(dict(c))))

    # init population
    if df.empty:
        pop = [repair_cand(toolbox.individual()) for _ in range(int(cfg["ea"]["pop_size"]))]
        gen = 0
    else:
        gen = int(df["gen"].max())
        pop = [as_ind(c) for c in df[df["gen"] == gen]["cand"].tolist()]

    # -------------------------------------------------
    # Generational loop
    # -------------------------------------------------
    for g in range(gen, int(cfg["ea"]["generations"])):
        print(f"\n[NAS] Generation {g}")
        recs = []
        recs_lock = Lock()
        threads = []
        sem = Semaphore(8)

        for ind in pop:
            repair_cand(ind)
            hsh = cand_hash(ind, cfg)

            # skip if already evaluated, but still use for selection
            old = df[(df["hash"] == hsh) & (df["evaluated"] == True)]
            if not old.empty:
                row = old.iloc[0]
                with recs_lock:
                    recs.append([g, hsh, row["cand"], row["fitness"], row["val_acc"], row["LUT"], row["DSP"], row["BRAM_18K"], row["latency_ns"], row["report_path"], True])
                print(f"  cached {row["cand"]['hidden']} q={row["cand"]['quant']} -> acc={row['val_acc']:.3f} fit={row['fitness']:.4f}")
                continue

            wrk = run_dir / pathlib.Path(cfg["finn"]["tmp_root"]) / hsh
            ensure_clean_dir(wrk)

            cand_plain = freeze_cand(ind)

            cand_path = wrk / "cand.json"
            cand_path.write_text(cand_to_json(cand_plain), encoding="utf-8")

            # quick training
            wpath = wrk / "cand.pt"
            print(f"Training hash={hsh} hidden={cand_plain['hidden']} quant={cand_plain['quant']}")
            acc = train_quick(cfg, cand_plain, str(wpath))

            qonnx = wrk / "cand.qonnx"
            bdir = wrk / "build"

            # build FINN concurrently inside container
            thread = Thread(target=build_task, args=(sem, recs_lock, recs, cfg, finn_cfg, qonnx, bdir, cand_plain, cand_path, wpath, g, hsh, acc))
            thread.start()
            threads.append(thread)

        for t in threads:
            t.join()

        gen_df = pd.DataFrame(recs, columns=COLS)

        # persist unique best-per-architecture (avoid dupes)
        persist = gen_df.copy() if df.empty else pd.concat([df, gen_df], ignore_index=True)
        persist = (persist.sort_values(["fitness", "val_acc"], ascending=[True, False]).drop_duplicates("hash", keep="first"))
        persist.to_pickle(pkl)
        persist.to_csv(run_dir / pathlib.Path(cfg["ea"]["csv_out"]), index=False)
        df = persist

        # assign DEAP fitness values for selection (lower is better)
        best_per_hash = (gen_df.sort_values(["fitness", "val_acc"], ascending=[True, False]).drop_duplicates("hash", keep="first"))
        fit_map = dict(zip(best_per_hash["hash"], best_per_hash["fitness"]))
        for ind in pop:
            hsh = cand_hash(ind, cfg)
            ind.fitness.values = (float(fit_map.get(hsh, np.inf)),)

        # DEAP reproduction (selection / mate / mutate)
        elite_k = int(cfg["ea"]["elitism"])
        rand_k = int(cfg["ea"]["randoms"])
        pop_size = int(cfg["ea"]["pop_size"])
        n_off = max(0, pop_size - elite_k - rand_k)

        elites = list(map(toolbox.clone, tools.selBest(pop, k=elite_k))) if elite_k > 0 else []
        offspring = list(map(toolbox.clone, toolbox.select(pop, k=n_off))) if n_off > 0 else []

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

    print("\n[NAS] Top 10:")
    top = (df[df["evaluated"] == True]
            .sort_values(["fitness", "val_acc"], ascending=[True, False])
            .drop_duplicates("hash", keep="first")
            .head(10).copy())
    top = top.reset_index(drop=True)
    top["cand_brief"] = top["cand"].apply(lambda c: f"h={c.get('hidden', [])} q={c.get('quant', {})}")
    print(top[["cand_brief","fitness","val_acc","LUT","DSP","BRAM_18K","latency_ns"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/nas.yaml")
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