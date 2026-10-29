import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Uso: imposta la CONFIGURAZIONE qui sotto e premi Run.
# (In alternativa da terminale, gli argomenti sostituiscono la configurazione:
#  python3 DT/print_dt_table.py [metrica] [seed] [K=<k>])
#   metrica:  all        (default) - Reward, CDR e SR insieme, un blocco per configurazione
#             reward                - Reward: reward media dell'episodio (non scontata)
#             discounted            - CDR: cumulative discounted reward media (gamma = 0.99)
#             success               - SR: success rate in ambiente (%)
#             length                - lunghezza media dell'episodio (passi)
#   seed:     es. 0,1,2 (default: tutti i seed trovati)
#   K=<k>     opzionale, in qualsiasi posizione: risultati dell'ablation sul contesto
#             (DT/context_ablation/K<k>/..., da experiments_dt_context.py) invece di quelli
#             principali (K = 50). Es.: python3 DT/print_dt_table.py all 0 K=10
#
# Per ogni combinazione la metrica si calcola sul CSV di valutazione di ogni seed
# (DT/<State>/<reward>/task<N>/seed<S>/eval/eval_<dataset>.csv, 500 episodi su disposizioni di test diverse)
# e si riporta media +- deviazione standard tra i seed. Con un solo seed c'e' solo la media.

# =====================================================================
# CONFIGURAZIONE: modifica qui cosa stampare, poi premi Run
# =====================================================================
METRIC_DEFAULT = "all"   # "all", "success", "discounted", "reward" o "length"
SEEDS_DEFAULT = None     # es. [0] oppure [0, 1, 2]; None = tutti i seed trovati
K_DEFAULT = None         # None = risultati principali (K = 50); es. 5 = ablation con K = 5
AUTOMATON_DEFAULT = None # None = DT normale; "true" = DT con lo stato vero dell'automa; "grounder" = dal grounder
                         # (DT/automaton_state/true/K<k>/..., da experiments_dt_automaton.py)
# =====================================================================

METRICS = {
    "reward": ("reward media dell'episodio", lambda df: df["reward_puro"].mean()),
    "success": ("success rate (%)", lambda df: 100.0 * df["success"].mean()),
    "length": ("lunghezza media dell'episodio (passi)", lambda df: df["episode_length"].mean()),
    "discounted": ("reward scontata media (gamma 0.99)", lambda df: df["discounted_reward"].mean()),
}
# argomenti da terminale (se ci sono) al posto della configurazione; K=<k> in qualsiasi posizione
K_ARG = next((a for a in sys.argv[1:] if a.startswith("K=")), None)
ARGS = [a for a in sys.argv[1:] if not a.startswith("K=")]
METRIC = ARGS[0] if len(ARGS) > 0 else METRIC_DEFAULT
K = int(K_ARG[2:]) if K_ARG else K_DEFAULT
# metriche della vista "all", con l'etichetta della colonna
ALL_METRICS = [("reward", "Reward"), ("discounted", "CDR"), ("success", "SR")]
COL_GAP = 2     # spazi tra le colonne di uno stesso dataset
GROUP_GAP = 4   # spazi tra un dataset e l'altro (expert / medium / replay)

DT_ROOT = Path(__file__).resolve().parent
AUTOMATON = AUTOMATON_DEFAULT
if AUTOMATON is not None:
    DT_ROOT = DT_ROOT / "automaton_state" / AUTOMATON / f"K{K if K is not None else 50}"
elif K is not None:
    DT_ROOT = DT_ROOT / "context_ablation" / f"K{K}"
STATE_TYPES = ["Symbolic", "Image"]
REWARD_TYPES = ["three_value", "distance"]
DATASETS = [("medium_expert", "expert"), ("medium", "medium"), ("medium_replay", "replay")]
NUM_TASKS = 10


def find_seeds():
    return sorted({int(p.name[4:]) for p in DT_ROOT.glob("*/*/task*/seed*") if p.name[4:].isdigit()})


if len(ARGS) > 1:
    SEEDS = [int(s) for s in ARGS[1].split(",")]
else:
    SEEDS = list(SEEDS_DEFAULT) if SEEDS_DEFAULT is not None else find_seeds()


def cell(state_type, reward_type, task, dataset, metric=None):
    """'media +- std' tra i seed disponibili per una combinazione, oppure N/A."""
    metric = metric or METRIC
    values = []
    for seed in SEEDS:
        csv = DT_ROOT / state_type / reward_type / f"task{task}" / f"seed{seed}" / "eval" / f"eval_{dataset}.csv"
        if csv.exists():
            values.append(METRICS[metric][1](pd.read_csv(csv)))
    if not values:
        return "N/A", None
    mean = float(np.mean(values))
    if len(values) == 1:
        return f"{mean:.1f}", mean
    # deviazione standard campionaria tra i seed
    return f"{mean:.1f} ± {float(np.std(values, ddof=1)):.1f}", mean


def build_table(state_type, reward_type):
    rows, means = [], {label: [] for _, label in DATASETS}
    for task in range(1, NUM_TASKS + 1):
        row = {"Task": f"task{task}"}
        for file_name, label in DATASETS:
            text, mean = cell(state_type, reward_type, task, file_name)
            row[label] = text
            if mean is not None:
                means[label].append(mean)
        rows.append(row)
    # ultima riga: media sui task (delle medie tra seed)
    rows.append({"Task": "media", **{label: f"{np.mean(v):.1f}" if v else "N/A" for label, v in means.items()}})
    return pd.DataFrame(rows).set_index("Task")


def format_all_table(state_type, reward_type):
    """Reward, CDR e SR per ogni dataset, come testo: il nome del dataset e' centrato
    sopra la colonna CDR (quella in mezzo) e i dataset sono separati da piu' spazio."""
    row_labels = [f"task{t}" for t in range(1, NUM_TASKS + 1)] + ["media"]
    columns = [(file_name, label, metric, short) for file_name, label in DATASETS for metric, short in ALL_METRICS]
    cells = {r: {} for r in row_labels}
    for file_name, label, metric, short in columns:
        means = []
        for task in range(1, NUM_TASKS + 1):
            text, mean = cell(state_type, reward_type, task, file_name, metric)
            cells[f"task{task}"][(label, short)] = text
            if mean is not None:
                means.append(mean)
        # ultima riga: media sui task (delle medie tra seed)
        cells["media"][(label, short)] = f"{np.mean(means):.1f}" if means else "N/A"

    widths = {(label, short): max(len(short), *(len(cells[r][(label, short)]) for r in row_labels))
              for _, label, _, short in columns}
    first_width = max(len("Task"), *(len(r) for r in row_labels))

    # riga dei nomi delle metriche, tenendo traccia di dove sta ogni colonna
    metric_line, positions = "Task".ljust(first_width), {}
    for i, (_, label, _, short) in enumerate(columns):
        metric_line += " " * (GROUP_GAP if i % len(ALL_METRICS) == 0 else COL_GAP)
        positions[(label, short)] = len(metric_line)
        metric_line += short.rjust(widths[(label, short)])

    # riga dei nomi dei dataset, centrati sopra la colonna CDR
    dataset_line = ""
    for _, label in DATASETS:
        middle = ALL_METRICS[len(ALL_METRICS) // 2][1]
        center = positions[(label, middle)] + widths[(label, middle)] // 2
        start = max(center - len(label) // 2, len(dataset_line) + 1)
        dataset_line = dataset_line.ljust(start) + label

    lines = [dataset_line, metric_line, "-" * len(metric_line)]
    for r in row_labels:
        if r == "media":
            lines.append("-" * len(metric_line))
        line = r.ljust(first_width)
        for i, (_, label, _, short) in enumerate(columns):
            line += " " * (GROUP_GAP if i % len(ALL_METRICS) == 0 else COL_GAP)
            line += cells[r][(label, short)].rjust(widths[(label, short)])
        lines.append(line)
    return "\n".join(lines)


def count_runs():
    """quanti seed hanno la valutazione per ogni combinazione (per segnalare seed incompleti)"""
    counts = {}
    for st in STATE_TYPES:
        for rt in REWARD_TYPES:
            for task in range(1, NUM_TASKS + 1):
                for file_name, _ in DATASETS:
                    counts[(st, rt, task, file_name)] = sum(
                        (DT_ROOT / st / rt / f"task{task}" / f"seed{s}" / "eval" / f"eval_{file_name}.csv").exists()
                        for s in SEEDS)
    return counts


def main():
    if METRIC != "all" and METRIC not in METRICS:
        print(f"Metrica '{METRIC}' non valida. Scegli tra: all, {', '.join(METRICS)}")
        return

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 400)
    pd.set_option("display.expand_frame_repr", False)

    title = "Reward / CDR / SR" if METRIC == "all" else METRICS[METRIC][0]
    context = f"K = {K} (ablation)" if K is not None else "K = 50"
    if AUTOMATON is not None:
        context = f"K = {K if K is not None else 50} + stato dell'automa ({AUTOMATON})"
    print(f"DECISION TRANSFORMER - {title} - {context} - seed {', '.join(map(str, SEEDS))}")
    print("In ogni cella: media ± deviazione standard tra i seed (500 episodi per seed, ognuno su una disposizione di test diversa)")
    if METRIC == "all":
        print("Reward = reward media dell'episodio | CDR = cumulative discounted reward (gamma 0.99) | SR = success rate (0-100)")
    incomplete = [k for k, n in count_runs().items() if 0 < n < len(SEEDS)]
    if incomplete:
        print(f"ATTENZIONE: {len(incomplete)} combinazioni non hanno ancora tutti i seed (la media usa quelli presenti)")

    if METRIC == "all":
        # un blocco per configurazione, uno sotto l'altro
        for state_type in STATE_TYPES:
            for reward_type in REWARD_TYPES:
                print(f"\n=== {state_type} / {reward_type} ===")
                print(format_all_table(state_type, reward_type))
        return

    # un blocco per stato (prima Symbolic, poi Image sotto), con dentro three_value e distance affiancati
    for state_type in STATE_TYPES:
        configs = [(state_type, rt) for rt in REWARD_TYPES]
        combined = pd.concat([build_table(st, rt) for st, rt in configs], axis=1, keys=configs)
        print()
        print(combined.to_string())


if __name__ == "__main__":
    main()
