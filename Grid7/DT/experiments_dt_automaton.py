"""Batteria del Decision Transformer con lo stato dell'automa della formula in input.

AUTOMATON = "true":     stato vero, calcolato con i simboli veri (DT+Oracle)
AUTOMATON = "grounder": l'automa avanza con i simboli predetti dal grounder allenato solo dai reward
                        (DT+NRM; i grounder vanno allenati prima con Grounding/experiments_grounding.py)
Stesso training e stessa valutazione di experiments_dt.py.
Risultati in DT/automaton_state/<automaton>/<State>/<reward>/task<N>/seed<S>/
Uso: imposta la CONFIGURAZIONE e premi Run, oppure python3 DT/experiments_dt_automaton.py [true|grounder]
"""
import gc
import sys
import traceback
from pathlib import Path

import torch

from experiments_dt import run_case

# =====================================================================
# CONFIGURAZIONE
# =====================================================================
AUTOMATON = "grounder"
STATE_TYPES = ["symbolic", "image"]
REWARD_TYPES = ["three_value", "distance"]
TASKS = range(1, 11)
DATASETS = ["medium_expert", "medium", "medium_replay"]
SEEDS = [0, 1, 2]
# =====================================================================


def automaton_dir(automaton, state_type, reward_type, task, seed):
    return (Path(__file__).resolve().parent / "automaton_state" / automaton / state_type.capitalize()
            / reward_type / f"task{task}" / f"seed{seed}")


def main():
    automaton = sys.argv[1] if len(sys.argv) > 1 else AUTOMATON
    summary = []
    for state_type in STATE_TYPES:
        for reward_type in REWARD_TYPES:
            for task in TASKS:
                for dataset in DATASETS:
                    for seed in SEEDS:
                        name = f"automa {automaton} / {state_type} / {reward_type} / task{task} / {dataset} / seed{seed}"
                        print(f"\n{'=' * 70}\n {name}\n{'=' * 70}", flush=True)
                        try:
                            sr = run_case(state_type, reward_type, task, dataset, seed,
                                          automaton_dir(automaton, state_type, reward_type, task, seed), automaton)
                            summary.append((name, "saltato (gia' presente)" if sr is None else f"{sr:.1f}%"))
                        except Exception:
                            traceback.print_exc()
                            summary.append((name, "FAILED"))
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

    print(f"\n{'=' * 70}\n SUMMARY - success rate del modello finale\n{'=' * 70}")
    for name, res in summary:
        print(f"{name:<80} {res}")


if __name__ == "__main__":
    main()
