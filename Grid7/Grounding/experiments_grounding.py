"""Batteria di training dei grounder (Neural Reward Machine) per la griglia 7x7.

Per ogni combinazione allena il grounder solo dai reward e lo valuta sul test (vedi grounder.py).
Risultati in Grounding/<State>/<reward>/task<N>/seed<S>/{grounder,log}/; li usa DT+NRM
(DT/experiments_dt_automaton.py con AUTOMATON = "grounder"), con lo stesso task, dataset e seed.
Tabella: python3 Grid7/Grounding/print_grounding_table.py
Uso: imposta la CONFIGURAZIONE e premi Run, oppure
    python3 Grounding/experiments_grounding.py [reward1,...] [task1,...] [dataset1,...] [state1,...]
(gli argomenti mancanti prendono i valori della configurazione)
"""
import gc
import multiprocessing as mp
import sys
import traceback

import torch

from grounder import grounder_path, log_path, train_grounder

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
# run symbolic in contemporanea; i run image vanno uno alla volta (uno usa gia' tutta la GPU)
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
    REWARD_TYPES = sys.argv[1].split(",") if len(sys.argv) > 1 else REWARD_TYPES
    TASKS = [int(t) for t in sys.argv[2].split(",")] if len(sys.argv) > 2 else TASKS
    DATASETS = sys.argv[3].split(",") if len(sys.argv) > 3 else DATASETS
    STATE_TYPES = sys.argv[4].split(",") if len(sys.argv) > 4 else STATE_TYPES
    jobs = [(st, rt, task, ds, seed) for st in STATE_TYPES for rt in REWARD_TYPES for task in TASKS
            for ds in DATASETS for seed in SEEDS]
    results = dict(run_job(job) for job in jobs if job[0] == "image")
    symbolic = [job for job in jobs if job[0] == "symbolic"]
    if NUM_WORKERS_SYMBOLIC > 1:
        # "spawn": ogni processo inizializza la GPU da zero; un processo nuovo per ogni run
        with mp.get_context("spawn").Pool(NUM_WORKERS_SYMBOLIC, maxtasksperchild=1) as pool:
            results |= dict(pool.imap_unordered(run_job, symbolic))
    else:
        results |= dict(run_job(job) for job in symbolic)

    print(f"\n{'=' * 70}\n SUMMARY - risultati sul test (balanced accuracy | symbol accuracy sui simboli della formula)\n{'=' * 70}")
    for job in jobs:
        print(f"{job_name(job):<55} {results[job_name(job)]}")
