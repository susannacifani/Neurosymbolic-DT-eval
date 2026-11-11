"""Quanto lo stato dell'automa calcolato con il grounder dice sullo stato vero (quello che riceve DT+NRM
rispetto a DT+Oracle). Per ogni grounder, su tutte le osservazioni del suo dataset:
- uguale: stato predetto = stato vero (numerazione canonica; non tiene conto delle simmetrie della formula)
- purezza: per ogni stato predetto si prende lo stato vero piu' frequente; frazione di passi spiegata cosi'.
  Misura quanto lo stato vero si ricava da quello predetto, con qualunque rinumerazione
- base: purezza di un predittore costante (lo stato vero piu' frequente), cioe' nessuna informazione
Uso: python3 DT/grounder_state_quality.py [three_value|distance] [task1,...]
"""
import sys
from collections import Counter, defaultdict

import h5py
import numpy as np

from decision_transformer import cell_codes, dataset_path, make_tracker
from grounder import grounder_path

# =====================================================================
# CONFIGURAZIONE
# =====================================================================
REWARD_TYPE = sys.argv[1] if len(sys.argv) > 1 else "three_value"
TASKS = [int(t) for t in sys.argv[2].split(",")] if len(sys.argv) > 2 else range(1, 11)
DATASETS = ["medium_expert", "medium", "medium_replay"]
SEEDS = [0, 1, 2]
STATE_TYPE = "symbolic"
# =====================================================================


def codes_per_traj(h5):
    with h5py.File(h5, "r") as f:
        return [cell_codes(f[k]["grid"][:].astype(np.int64), f[k]["positions"][:-1].astype(np.int64))
                for k in sorted(f.keys(), key=int)]


def states(tracker, trajs):
    sym = tracker.symbols(np.concatenate(trajs))
    ends = np.cumsum([len(t) for t in trajs])
    return np.concatenate([tracker.states_from_symbols(s) for s in np.split(sym, ends[:-1])])


def main():
    for task in TASKS:
        true_tracker = make_tracker("true", STATE_TYPE, REWARD_TYPE, task)
        for ds in DATASETS:
            trajs = codes_per_traj(dataset_path(REWARD_TYPE, task, ds))
            q_true = states(true_tracker, trajs)
            base = Counter(q_true).most_common(1)[0][1] / len(q_true)
            for seed in SEEDS:
                g = grounder_path(STATE_TYPE, REWARD_TYPE, task, ds, seed)
                q_pred = states(make_tracker("grounder", STATE_TYPE, REWARD_TYPE, task, str(g)), trajs)
                by_pred = defaultdict(Counter)
                for p, t in zip(q_pred, q_true):
                    by_pred[p][t] += 1
                purity = sum(c.most_common(1)[0][1] for c in by_pred.values()) / len(q_true)
                print(f"task{task} {ds:<14} seed{seed}: uguale {100 * np.mean(q_pred == q_true):5.1f}% | "
                      f"purezza {100 * purity:5.1f}% | base {100 * base:5.1f}% | "
                      f"stati usati {len(by_pred)}/{true_tracker.num_states}", flush=True)


if __name__ == "__main__":
    main()
