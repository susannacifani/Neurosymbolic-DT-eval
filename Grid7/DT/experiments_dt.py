"""Batteria del Decision Transformer standard (senza stato dell'automa).

Per ogni combinazione: training (MAX_EPOCHS epoch fisse) e valutazione del modello finale e di quelli
alle epoch SAVE_EPOCHS. Risultati in DT/<State>/<reward>/task<N>/seed<S>/ (vedi decision_transformer.py).
Uso: imposta la CONFIGURAZIONE e premi Run.
"""
import gc
import multiprocessing as mp
import traceback

import torch

from decision_transformer import SAVE_EPOCHS, evaluate, output_paths, run_dir, train

# =====================================================================
# CONFIGURAZIONE
# =====================================================================
STATE_TYPES = ["symbolic", "image"]
REWARD_TYPES = ["three_value", "distance"]
TASKS = range(1, 11)
DATASETS = ["medium_expert", "medium", "medium_replay"]
SEEDS = [0, 1, 2]
# se False, i run gia' completati vengono saltati (per riprendere una batteria interrotta)
OVERWRITE = False
# run in contemporanea (processi separati); con una GPU sola conviene 1 per image
NUM_WORKERS = 1
# =====================================================================


def run_case(state_type, reward_type, task, dataset, seed, base, automaton=None):
    """training + valutazione del modello finale e di quelli intermedi; restituisce il success rate finale"""
    dirs = [base] + [base / f"epoch{e:03d}" for e in SAVE_EPOCHS]
    if OVERWRITE or not output_paths(state_type, reward_type, task, dataset, seed, base)["model"].exists():
        train(state_type, reward_type, task, dataset, seed=seed, out_dir=base, automaton=automaton)
    results = {}
    for d in dirs:
        summary = output_paths(state_type, reward_type, task, dataset, seed, d)["summary"]
        if OVERWRITE or not summary.exists():
            results[d] = evaluate(state_type, reward_type, task, dataset, seed=seed, out_dir=d)["success_rate"]
    return results.get(base)


def run_job(job):
    state_type, reward_type, task, dataset, seed = job
    name = f"{state_type} / {reward_type} / task{task} / {dataset} / seed{seed}"
    print(f"\n{'=' * 70}\n {name}\n{'=' * 70}", flush=True)
    try:
        sr = run_case(state_type, reward_type, task, dataset, seed, run_dir(state_type, reward_type, task, seed))
        result = "saltato (gia' presente)" if sr is None else f"{sr:.1f}%"
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
    if NUM_WORKERS > 1:
        # "spawn": ogni processo inizializza la GPU da zero (necessario con CUDA)
        with mp.get_context("spawn").Pool(NUM_WORKERS) as pool:
            results = dict(pool.imap_unordered(run_job, jobs))
    else:
        results = dict(run_job(job) for job in jobs)

    print(f"\n{'=' * 70}\n SUMMARY - success rate del modello finale\n{'=' * 70}")
    for job in jobs:
        name = f"{job[0]} / {job[1]} / task{job[2]} / {job[3]} / seed{job[4]}"
        print(f"{name:<55} {results[name]}")
