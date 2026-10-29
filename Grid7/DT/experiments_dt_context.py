"""Ablation sul contesto K del Decision Transformer.

Stesso training e stessa valutazione di experiments_dt.py, cambia solo K (quanti passi di
storia vede il modello). I risultati vanno in una cartella separata, per non toccare quelli
della configurazione principale (K = 50, l'intero episodio):
    DT/context_ablation/K<k>/<State>/<reward>/task<N>/seed<S>/{dt_model,log,eval}/
Tabella: python3 DT/print_dt_table.py [metrica] [seed] K=<k>
"""
import gc
import traceback
from pathlib import Path

import torch

from decision_transformer import evaluate, output_paths, train

# =====================================================================
# CONFIGURAZIONE: modifica qui cosa lanciare, poi premi Run
# =====================================================================
CONTEXT_LENS = [1]
STATE_TYPES = ["image"]
REWARD_TYPES = ["distance", "three_value"]
TASKS = range(1, 11)
DATASETS = ["medium_expert", "medium", "medium_replay"]
SEEDS = [0]

DO_TRAIN = True
DO_EVAL = True
# se False, le combinazioni con modello / valutazione gia' presenti vengono saltate
# (utile per riprendere una batteria interrotta); se True si rifa' tutto
OVERWRITE = False
# =====================================================================

ABLATION_ROOT = Path(__file__).resolve().parent / "context_ablation"


def ablation_dir(k, state_type, reward_type, task, seed):
    return ABLATION_ROOT / f"K{k}" / state_type.capitalize() / reward_type / f"task{task}" / f"seed{seed}"


summary = []
for k in CONTEXT_LENS:
    for state_type in STATE_TYPES:
        for reward_type in REWARD_TYPES:
            for task in TASKS:
                for dataset in DATASETS:
                    for seed in SEEDS:
                        name = f"K{k} / {state_type} / {reward_type} / task{task} / {dataset} / seed{seed}"
                        print(f"\n{'=' * 70}\n {name}\n{'=' * 70}")
                        out_dir = ablation_dir(k, state_type, reward_type, task, seed)
                        paths = output_paths(state_type, reward_type, task, dataset, seed, out_dir)
                        try:
                            if DO_TRAIN and (OVERWRITE or not paths["model"].exists()):
                                train(state_type, reward_type, task, dataset, seed=seed, context_len=k, out_dir=out_dir)
                            if DO_EVAL and (OVERWRITE or not paths["csv"].exists()):
                                # la valutazione legge K dal modello salvato
                                res = evaluate(state_type, reward_type, task, dataset, seed=seed, out_dir=out_dir)
                                summary.append((name, f"{res['success_rate']:.1f}%"))
                            else:
                                summary.append((name, "saltato (gia' presente)"))
                        except Exception:
                            traceback.print_exc()
                            summary.append((name, "FAILED"))

                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

print(f"\n{'=' * 70}\n SUMMARY - success rate\n{'=' * 70}")
for name, res in summary:
    print(f"{name:<70} {res}")
