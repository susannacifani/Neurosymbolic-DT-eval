import gc
import multiprocessing as mp
import traceback

import torch

from decision_transformer import evaluate, output_paths, train

# =====================================================================
# CONFIGURAZIONE: modifica qui cosa lanciare, poi premi Run
# =====================================================================
STATE_TYPES = ["symbolic", "image"]
REWARD_TYPES = ["three_value", "distance"]
TASKS = range(1,11)
DATASETS = ["medium_expert", "medium", "medium_replay"]
SEEDS = [0, 1, 2]

DO_TRAIN = True
DO_EVAL = True
# se False, le combinazioni con modello / valutazione gia' presenti vengono saltate
# (utile per riprendere una batteria interrotta); se True si rifa' tutto
OVERWRITE = False
# numero di run eseguiti in contemporanea (processi separati che prendono i run da una lista unica).
# Con una sola GPU conviene 1: un run image la usa gia' al 100%, quindi 2 run insieme vanno ciascuno
# a meno della meta' della velocita' (misurato), e con 3 la memoria della GPU si satura. Serve solo
# con piu' GPU o per molti run symbolic, che la usano poco.
NUM_WORKERS = 1
# =====================================================================


def run_job(job):
    """un run (training + valutazione); restituisce (nome, esito)"""
    state_type, reward_type, task, dataset, seed = job
    name = f"{state_type} / {reward_type} / task{task} / {dataset} / seed{seed}"
    print(f"\n{'=' * 70}\n {name}\n{'=' * 70}", flush=True)
    paths = output_paths(state_type, reward_type, task, dataset, seed)
    try:
        if DO_TRAIN and (OVERWRITE or not paths["model"].exists()):
            train(state_type, reward_type, task, dataset, seed=seed)
        if DO_EVAL and (OVERWRITE or not paths["csv"].exists()):
            res = evaluate(state_type, reward_type, task, dataset, seed=seed)
            result = f"{res['success_rate']:.1f}%"
        else:
            result = "saltato (gia' presente)"
    except Exception:
        traceback.print_exc()
        result = "FAILED"
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return name, result


if __name__ == "__main__":
    jobs = [(st, rt, task, ds, seed) for st in STATE_TYPES for rt in REWARD_TYPES for task in TASKS
            for ds in DATASETS for seed in SEEDS]
    # prima i run image (i piu' lunghi), cosi' alla fine non resta un solo processo a lavorare
    jobs.sort(key=lambda job: job[0] != "image")
    if NUM_WORKERS > 1:
        # "spawn": ogni processo inizializza la GPU da zero (necessario con CUDA)
        with mp.get_context("spawn").Pool(NUM_WORKERS) as pool:
            results = dict(pool.imap_unordered(run_job, jobs))
    else:
        results = dict(run_job(job) for job in jobs)

    print(f"\n{'=' * 70}\n SUMMARY - success rate\n{'=' * 70}")
    for job in sorted(jobs):
        name = f"{job[0]} / {job[1]} / task{job[2]} / {job[3]} / seed{job[4]}"
        print(f"{name:<55} {results[name]}")
