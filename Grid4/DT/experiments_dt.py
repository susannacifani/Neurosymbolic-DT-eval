import gc
import traceback

import torch

from decision_transformer import evaluate, output_paths, train

# =====================================================================
# CONFIGURAZIONE: modifica qui cosa lanciare, poi premi Run
# =====================================================================
STATE_TYPES = ["symbolic", "image"]
REWARD_TYPES = ["three_value", "distance"]
TASKS = range(1, 11)
DATASETS = ["medium_expert", "medium", "medium_replay"]
SEEDS = [0, 1, 2]

DO_TRAIN = True
DO_EVAL = True
# se False, le combinazioni con modello / valutazione gia' presenti vengono saltate
# (utile per riprendere una batteria interrotta); se True si rifa' tutto
OVERWRITE = False
# =====================================================================

summary = []
for state_type in STATE_TYPES:
    for reward_type in REWARD_TYPES:
        for task in TASKS:
            for dataset in DATASETS:
                for seed in SEEDS:
                    name = f"{state_type} / {reward_type} / task{task} / {dataset} / seed{seed}"
                    print(f"\n{'=' * 70}\n {name}\n{'=' * 70}")
                    paths = output_paths(state_type, reward_type, task, dataset, seed)
                    try:
                        if DO_TRAIN and (OVERWRITE or not paths["model"].exists()):
                            train(state_type, reward_type, task, dataset, seed=seed)
                        if DO_EVAL and (OVERWRITE or not paths["csv"].exists()):
                            res = evaluate(state_type, reward_type, task, dataset, seed=seed)
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
    print(f"{name:<55} {res}")
