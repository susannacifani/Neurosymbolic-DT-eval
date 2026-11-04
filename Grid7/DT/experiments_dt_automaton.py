"""Decision Transformer con lo stato dell'automa della formula in input.

A ogni passo il DT riceve, oltre a osservazione, return-to-go e azione, anche lo stato
dell'automa (quali sotto-obiettivi sono gia' stati raggiunti). Serve a vedere se questa
informazione sostituisce la memoria lunga quando il contesto K e' corto.
AUTOMATON = "true":     stato vero, calcolato dall'ambiente (oracolo, cioe' grounder perfetto).
AUTOMATON = "grounder": l'automa avanza con i simboli predetti dal grounder allenato solo dai reward
                        (Grounding/..., stesso dataset e stesso seed: vanno allenati prima).
Risultati in una cartella separata:
    DT/automaton_state/<automaton>/K<k>/<State>/<reward>/task<N>/seed<S>/{dt_model,log,eval}/
Tabella: in DT/print_dt_table.py imposta AUTOMATON_DEFAULT e K_DEFAULT
"""
import gc
import traceback
from pathlib import Path

import torch

from decision_transformer import evaluate, output_paths, train

# =====================================================================
# CONFIGURAZIONE: modifica qui cosa lanciare, poi premi Run
# =====================================================================
AUTOMATON = "grounder"        # "true" (stato vero) oppure "grounder" (stato dal grounder)
CONTEXT_LENS = [50]
STATE_TYPES = ["symbolic", "image"]
REWARD_TYPES = ["distance", "three_value"]
TASKS = range(1,11)
DATASETS = ["medium_expert", "medium", "medium_replay"]
SEEDS = [0, 1, 2]

DO_TRAIN = True
DO_EVAL = True
# se False, le combinazioni con modello / valutazione gia' presenti vengono saltate
# (utile per riprendere una batteria interrotta); se True si rifa' tutto
OVERWRITE = False
# =====================================================================

ABLATION_ROOT = Path(__file__).resolve().parent / "automaton_state" / AUTOMATON


def ablation_dir(k, state_type, reward_type, task, seed):
    return ABLATION_ROOT / f"K{k}" / state_type.capitalize() / reward_type / f"task{task}" / f"seed{seed}"


summary = []
for k in CONTEXT_LENS:
    for state_type in STATE_TYPES:
        for reward_type in REWARD_TYPES:
            for task in TASKS:
                for dataset in DATASETS:
                    for seed in SEEDS:
                        name = f"automa {AUTOMATON} / K{k} / {state_type} / {reward_type} / task{task} / {dataset} / seed{seed}"
                        print(f"\n{'=' * 70}\n {name}\n{'=' * 70}")
                        out_dir = ablation_dir(k, state_type, reward_type, task, seed)
                        paths = output_paths(state_type, reward_type, task, dataset, seed, out_dir)
                        try:
                            if DO_TRAIN and (OVERWRITE or not paths["model"].exists()):
                                train(state_type, reward_type, task, dataset, seed=seed, context_len=k, out_dir=out_dir,
                                      automaton=AUTOMATON)
                            if DO_EVAL and (OVERWRITE or not paths["csv"].exists()):
                                # la valutazione legge K e l'uso dell'automa dal modello salvato
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
    print(f"{name:<80} {res}")
