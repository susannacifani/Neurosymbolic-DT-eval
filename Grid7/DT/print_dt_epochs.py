"""Epoch di training del DT standard e del DT + stato dell'automa (NRM), per vedere se con l'automa
il DT converge prima.

Per ogni run si legge dal log di training la best epoch (quella con la validation loss minima: il
modello salvato e' quello) e le epoch fatte in tutto (best epoch + patience, se c'e' stato early stopping).
Righe: Symbolic e Image, ognuno con three_value e distance; colonne: un gruppo per dataset e la media
sui dataset, con la media sui task di ogni approccio. In fondo, per ogni riga, in quanti run il DT + NRM
ha fatto meno epoch del DT standard. Con piu' seed, media +- deviazione standard tra i seed (calcolata
sulle medie di ogni seed).

Uso: imposta la CONFIGURAZIONE qui sotto e premi Run.
"""
import re
from pathlib import Path

import numpy as np

# =====================================================================
# CONFIGURAZIONE: modifica qui cosa stampare, poi premi Run
# =====================================================================
K = 50                   # contesto usato negli esperimenti con l'automa (cartelle K<k>)
SEEDS = [0, 1, 2]        # es. [0] oppure [0, 1, 2]
EPOCHS = "best"          # "best" = best epoch, "total" = epoch fatte in tutto
# =====================================================================

DT_ROOT = Path(__file__).resolve().parent
STATE_TYPES = ["Symbolic", "Image"]
REWARD_TYPES = ["three_value", "distance"]
DATASETS = ["medium_expert", "medium", "medium_replay"]
TASKS = range(1, 11)

# (etichetta, cartella dei risultati); il DT standard e' in DT/<State>
APPROACHES = [
    ("DT", DT_ROOT),
    ("DT + NRM", DT_ROOT / "automaton_state" / "grounder" / f"K{K}"),
    ("DT + automa vero", DT_ROOT / "automaton_state" / "true" / f"K{K}"),
]


def read_epochs(root, state_type, reward_type, task, seed, dataset):
    """best epoch o epoch totali del run; None se il log non c'e' o il training non e' finito"""
    log = root / state_type / reward_type / f"task{task}" / f"seed{seed}" / "log" / f"train_log_{dataset}.txt"
    if not log.exists():
        return None
    text = log.read_text()
    best = re.search(r"Best epoch: (\d+)", text)
    if best is None:
        return None
    if EPOCHS == "best":
        return int(best[1])
    return int(re.findall(r"^Epoch (\d+) \|", text, re.M)[-1])


def per_seed_means(root, state_type, reward_type, datasets):
    """per ogni seed, la media sui task e dataset scelti (solo se ci sono tutti i run)"""
    means = []
    for seed in SEEDS:
        values = [read_epochs(root, state_type, reward_type, task, seed, ds) for task in TASKS for ds in datasets]
        if values and None not in values:
            means.append(np.mean(values))
    return means


def fmt(values):
    if not values:
        return "N/A"
    if len(values) == 1:
        return f"{values[0]:.1f}"
    return f"{np.mean(values):.1f} ± {np.std(values, ddof=1):.1f}"


groups = DATASETS + ["media"]
label_w = max(len(label) for label, _ in APPROACHES)
row_w = len("Symbolic / three_value")
cell_w = max(13 if len(SEEDS) > 1 else 6, label_w)
group_w = len(APPROACHES) * cell_w + (len(APPROACHES) - 1) * 2

print(f"\n{'BEST EPOCH' if EPOCHS == 'best' else 'EPOCH TOTALI'} (media sui {len(TASKS)} task), seed {SEEDS}\n")
print(" " * row_w + "    " + "    ".join(f"{g:^{group_w}}" for g in groups))
print(" " * row_w + "    " + "    ".join("  ".join(f"{label:>{cell_w}}" for label, _ in APPROACHES) for _ in groups))
for state_type in STATE_TYPES:
    for reward_type in REWARD_TYPES:
        cells = []
        for g in groups:
            datasets = DATASETS if g == "media" else [g]
            cells.append("  ".join(f"{fmt(per_seed_means(root, state_type, reward_type, datasets)):>{cell_w}}"
                                   for _, root in APPROACHES))
        print(f"{state_type + ' / ' + reward_type:<{row_w}}    " + "    ".join(cells))

# in quanti run (task x dataset x seed) ogni approccio con l'automa ha fatto meno epoch del DT standard
(base_label, base_root), others = APPROACHES[0], APPROACHES[1:]
for label, root in others:
    print(f"\nRun in cui {label} fa meno epoch / uguali / piu' di {base_label}:")
    for state_type in STATE_TYPES:
        for reward_type in REWARD_TYPES:
            fewer = equal = more = 0
            for task in TASKS:
                for ds in DATASETS:
                    for seed in SEEDS:
                        a = read_epochs(root, state_type, reward_type, task, seed, ds)
                        b = read_epochs(base_root, state_type, reward_type, task, seed, ds)
                        if a is None or b is None:
                            continue
                        fewer, equal, more = fewer + (a < b), equal + (a == b), more + (a > b)
            print(f"  {state_type + ' / ' + reward_type:<{row_w}}  {fewer:>3} / {equal:>3} / {more:>3}")
