"""Batteria di training dei grounder (Neural Reward Machine) per la griglia 7x7.

Per ogni combinazione allena il grounder solo dai reward e lo valuta sul test (vedi grounder.py).
Risultati in Grounding/<State>/<reward>/task<N>/seed<S>/{grounder,log}/
Tabella: python3 Grid7/Grounding/print_grounding_table.py
I grounder servono poi al DT con lo stato dell'automa predetto (DT/experiments_dt_automaton.py con
AUTOMATON = "grounder"), che usa il grounder dello stesso task, dataset e seed.
"""
import gc
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
SEEDS = [0]
# se False, i run gia' completati (modello + log con il risultato sul test) vengono saltati
# (utile per riprendere una batteria interrotta); se True si rifa' tutto
OVERWRITE = False
# =====================================================================


def run_completed(state_type, reward_type, task, dataset, seed):
    log = log_path(state_type, reward_type, task, dataset, seed)
    return (grounder_path(state_type, reward_type, task, dataset, seed).exists() and log.exists()
            and "RECALL PER CLASSE SUL TEST SET" in log.read_text())


summary = []
for state_type in STATE_TYPES:
    for reward_type in REWARD_TYPES:
        for task in TASKS:
            for dataset in DATASETS:
                for seed in SEEDS:
                    name = f"{state_type} / {reward_type} / task{task} / {dataset} / seed{seed}"
                    print(f"\n{'=' * 70}\n {name}\n{'=' * 70}")
                    if not OVERWRITE and run_completed(state_type, reward_type, task, dataset, seed):
                        summary.append((name, "saltato (gia' presente)"))
                        continue
                    try:
                        res = train_grounder(state_type, reward_type, task, dataset, seed=seed)
                        summary.append((name, f"balanced {res['balanced']:.2f}% | simboli {res['symbol_formula']:.2f}%"))
                    except Exception:
                        traceback.print_exc()
                        summary.append((name, "FAILED"))
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

print(f"\n{'=' * 70}\n SUMMARY - risultati sul test (balanced accuracy | symbol accuracy sui simboli della formula)\n{'=' * 70}")
for name, res in summary:
    print(f"{name:<55} {res}")
