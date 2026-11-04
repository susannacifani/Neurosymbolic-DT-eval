"""Batteria di training dei grounder (Neural Reward Machine) per la griglia 7x7.

Per ogni combinazione allena il grounder solo dai reward e lo valuta sul test (vedi grounder.py).
Risultati in Grounding/<State>/<reward>/task<N>/seed<S>/{grounder,log}/
Tabella: python3 Grid7/Grounding/print_grounding_table.py
I grounder servono poi al DT con lo stato dell'automa predetto (DT/experiments_dt_automaton.py con
AUTOMATON = "grounder"), che usa il grounder dello stesso task, dataset e seed.
"""
import gc
import multiprocessing as mp
import traceback

import torch

from grounder import grounder_path, log_path, train_grounder

# =====================================================================
# CONFIGURAZIONE: modifica qui cosa lanciare, poi premi Run
# =====================================================================
STATE_TYPES = ["symbolic", "image"]
REWARD_TYPES = ["three_value", "distance"]
TASKS = range(1, 11)
DATASETS = ["medium_expert", "medium", "medium_replay"]
SEEDS = [0, 1, 2]
# se False, i run gia' completati (modello + log con il risultato sul test) vengono saltati
# (utile per riprendere una batteria interrotta); se True si rifa' tutto
OVERWRITE = False
# run symbolic eseguiti in contemporanea (processi separati che prendono i run da una lista unica).
# Con 3 i run symbolic vanno circa 2 volte piu' veloci in tutto (misurato, stessi pesi bit per bit);
# i run image vanno sempre uno alla volta, prima dei symbolic: uno usa gia' tutta la GPU
NUM_WORKERS_SYMBOLIC = 3
# =====================================================================


def run_completed(state_type, reward_type, task, dataset, seed):
    log = log_path(state_type, reward_type, task, dataset, seed)
    return (grounder_path(state_type, reward_type, task, dataset, seed).exists() and log.exists()
            and "RECALL PER CLASSE SUL TEST SET" in log.read_text())


def run_job(job):
    """un run (training + test); restituisce (nome, esito)"""
    state_type, reward_type, task, dataset, seed = job
    name = job_name(job)
    print(f"\n{'=' * 70}\n {name}\n{'=' * 70}", flush=True)
    if not OVERWRITE and run_completed(state_type, reward_type, task, dataset, seed):
        return name, "saltato (gia' presente)"
    try:
        res = train_grounder(state_type, reward_type, task, dataset, seed=seed)
        result = f"balanced {res['balanced']:.2f}% | simboli {res['symbol_formula']:.2f}%"
    except Exception:
        traceback.print_exc()
        result = "FAILED"
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return name, result


def job_name(job):
    return f"{job[0]} / {job[1]} / task{job[2]} / {job[3]} / seed{job[4]}"


if __name__ == "__main__":
    jobs = [(st, rt, task, ds, seed) for st in STATE_TYPES for rt in REWARD_TYPES for task in TASKS
            for ds in DATASETS for seed in SEEDS]
    results = dict(run_job(job) for job in jobs if job[0] == "image")
    symbolic = [job for job in jobs if job[0] == "symbolic"]
    if NUM_WORKERS_SYMBOLIC > 1:
        # "spawn": ogni processo inizializza la GPU da zero (necessario con CUDA); un processo nuovo
        # per ogni run, cosi' la memoria della GPU non si accumula da un run all'altro
        with mp.get_context("spawn").Pool(NUM_WORKERS_SYMBOLIC, maxtasksperchild=1) as pool:
            results |= dict(pool.imap_unordered(run_job, symbolic))
    else:
        results |= dict(run_job(job) for job in symbolic)

    print(f"\n{'=' * 70}\n SUMMARY - risultati sul test (balanced accuracy | symbol accuracy sui simboli della formula)\n{'=' * 70}")
    for job in jobs:
        print(f"{job_name(job):<55} {results[job_name(job)]}")
