"""Confronto tra DT, DT+NRM e DT+Oracle.

Righe: per ogni task una riga per approccio, in fondo la media sui task. Colonne: Symbolic e Image,
ognuno con three_value e distance; per ciascuno Reward, CDR e SR. Una tabella con la media sui dataset
e (opzionale) una per dataset; con piu' seed, media +- deviazione standard tra i seed (delle medie di
ogni seed).
Uso: imposta la CONFIGURAZIONE e premi Run.
"""
from pathlib import Path

import numpy as np
import pandas as pd

# =====================================================================
# CONFIGURAZIONE
# =====================================================================
SEEDS = None             # es. [0, 1, 2]; None = tutti i seed trovati
EPOCH = 200              # 200 = modelli finali; 50, 100 o 150 = modelli intermedi
SHOW_PER_DATASET = True  # dopo la tabella su tutti i dataset, una tabella per ogni dataset
# =====================================================================

DT_ROOT = Path(__file__).resolve().parent
STATE_TYPES = ["Symbolic", "Image"]
REWARD_TYPES = ["three_value", "distance"]
DATASETS = ["medium_expert", "medium", "medium_replay"]
NUM_TASKS = 10
COL_GAP = 2
GROUP_GAP = 4
EVAL_DIR = "eval" if EPOCH == 200 else f"epoch{EPOCH:03d}/eval"

APPROACHES = [
    ("DT", DT_ROOT),
    ("DT+NRM", DT_ROOT / "automaton_state" / "grounder"),
    ("DT+Oracle", DT_ROOT / "automaton_state" / "true"),
]
METRICS = [("Reward", lambda d: d["reward_puro"].mean()),
           ("CDR", lambda d: d["discounted_reward"].mean()),
           ("SR", lambda d: 100.0 * d["success"].mean())]


def find_seeds():
    seeds = set()
    for _, root in APPROACHES:
        seeds |= {int(p.name[4:]) for p in root.glob("*/*/task*/seed*") if p.name[4:].isdigit()}
    return sorted(seeds)


SEEDS = SEEDS if SEEDS is not None else find_seeds()


def load(root, state_type, reward_type, tasks, datasets):
    """per ogni seed, la media della metrica sui task e dataset scelti"""
    per_seed = {}
    for seed in SEEDS:
        frames = [pd.read_csv(f) for task in tasks for ds in datasets
                  if (f := root / state_type / reward_type / f"task{task}" / f"seed{seed}" / EVAL_DIR / f"eval_{ds}.csv").exists()]
        if frames:
            per_seed[seed] = {name: np.mean([fn(d) for d in frames]) for name, fn in METRICS} | {"n": len(frames)}
    return per_seed


def fmt(values):
    if not values:
        return "N/A"
    if len(values) == 1:
        return f"{values[0]:.1f}"
    return f"{np.mean(values):.1f} ± {np.std(values, ddof=1):.1f}"


def table(datasets, title):
    """una riga per task e approccio (prima colonna = task solo sulla prima riga del gruppo);
    l'ultimo gruppo e' la media su tutti i task"""
    columns = [(st, rt, m) for st in STATE_TYPES for rt in REWARD_TYPES for m, _ in METRICS]
    groups = [(f"task{t}", [t]) for t in range(1, NUM_TASKS + 1)] + [("media", list(range(1, NUM_TASKS + 1)))]
    rows, notes = [], set()
    for group, tasks in groups:
        for i, (label, root) in enumerate(APPROACHES):
            row = {}
            for st in STATE_TYPES:
                for rt in REWARD_TYPES:
                    per_seed = load(root, st, rt, tasks, datasets)
                    runs = sum(v["n"] for v in per_seed.values())
                    expected = len(SEEDS) * len(tasks) * len(datasets)
                    for m, _ in METRICS:
                        row[(st, rt, m)] = fmt([v[m] for v in per_seed.values()])
                    if per_seed and runs < expected:
                        row[(st, rt, METRICS[-1][0])] += "!"
                        notes.add("!")
            rows.append((group if i == 0 else "", label, row))

    task_width = max(len("Task"), *(len(g) for g, _, _ in rows))
    label_width = max(len("Approccio"), *(len(label) for _, label, _ in rows))
    first_width = task_width + 2 + label_width
    widths = {c: max(len(c[2]), *(len(r[c]) for _, _, r in rows)) for c in columns}
    positions, pos = {}, first_width
    for i, c in enumerate(columns):
        pos += GROUP_GAP if i % len(METRICS) == 0 else COL_GAP
        positions[c] = pos
        pos += widths[c]
    total = pos

    def centered(line, text, start, end):
        center = (start + end) // 2
        return line.ljust(max(center - len(text) // 2, len(line) + 1 if line.strip() else 0)) + text

    state_line, reward_line = "", ""
    for st in STATE_TYPES:
        cols = [c for c in columns if c[0] == st]
        state_line = centered(state_line, st, positions[cols[0]], positions[cols[-1]] + widths[cols[-1]])
        for rt in REWARD_TYPES:
            cols_rt = [c for c in cols if c[1] == rt]
            reward_line = centered(reward_line, rt, positions[cols_rt[0]], positions[cols_rt[-1]] + widths[cols_rt[-1]])
    metric_line = ("Task".ljust(task_width) + "  " + "Approccio").ljust(first_width)
    for c in columns:
        metric_line = metric_line.ljust(positions[c]) + c[2].rjust(widths[c])

    print(f"\n=== {title} ===")
    print(state_line)
    print(reward_line)
    print(metric_line)
    print("-" * total)
    for group, label, row in rows:
        if group == "media":
            print("-" * total)
        elif group and group != "task1":
            print()
        line = group.ljust(task_width) + "  " + label.ljust(label_width)
        for c in columns:
            line = line.ljust(positions[c]) + row[c].rjust(widths[c])
        print(line)
    if "!" in notes:
        print("! = non tutti i run sono presenti per i seed scelti (la media usa quelli presenti)")


def main():
    print(f"DECISION TRANSFORMER - confronto tra approcci - epoch {EPOCH} - seed {', '.join(map(str, SEEDS))}")
    print("Reward = reward media dell'episodio | CDR = cumulative discounted reward (gamma 0.99) | SR = success rate (0-100)")
    print("NRM (grounder) = stato dell'automa calcolato con i simboli del grounder allenato solo dai reward")
    table(DATASETS, "Tutti i dataset (media sui 3 dataset)")
    if SHOW_PER_DATASET:
        for ds in DATASETS:
            table([ds], f"Solo {ds}")


if __name__ == "__main__":
    main()
