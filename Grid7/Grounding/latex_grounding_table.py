"""Tabella LaTeX dei risultati del grounding (gli stessi numeri di print_grounding_table.py).

Una sola tabella: righe = task, prima il blocco Symbolic (task 1-10, poi le medie sui task 1-6 e 7-10) e poi
il blocco Image (solo i task con risultati); colonne = three-value e distance, per ognuno medium-expert, medium
e medium-replay. In ogni cella media +- deviazione standard tra i seed; nelle medie, tra i seed delle medie sui
task del gruppo.
La tabella viene scritta in un file .txt (in Grounding/latex/) da cui si incolla nel documento;
servono \\usepackage{booktabs,graphicx}.

Uso: imposta la CONFIGURAZIONE qui sotto e premi Run.
"""
import re
from pathlib import Path

import numpy as np

# =====================================================================
# CONFIGURAZIONE: modifica qui cosa generare, poi premi Run
# =====================================================================
SEEDS = None             # es. [0] oppure [0, 1, 2]; None = tutti i seed trovati
METRIC = "balanced"      # "balanced" = balanced accuracy del reward; "symbol" = symbol accuracy
OUTPUT = None            # None = Grounding/latex/grounder_<metric>.txt
# =====================================================================

STATE_TYPES = [("Symbolic", "Symbolic States"), ("Image", "Image States")]
REWARD_TYPES = [("three_value", "Three-Value Reward"), ("distance", "Distance Reward")]
DATASETS = [("medium_expert", "medium-expert"), ("medium", "medium"), ("medium_replay", "medium-replay")]
TASKS = range(1, 11)
# righe di media (solo nei blocchi che hanno tutti i task del gruppo)
GROUPS = [("\\shortstack[l]{Mean\\\\(tasks 1--6)}", [1, 2, 3, 4, 5, 6]),
          ("\\shortstack[l]{Mean\\\\(tasks 7--10)}", [7, 8, 9, 10])]

# risultati del grounding: Grounding/<State>/<reward>/task<N>/seed<S>/log/log_<dataset>.txt
GROUNDING_ROOT = Path(__file__).resolve().parent
PATTERNS = {
    "balanced": re.compile(r"BALANCED ACCURACY FINALE GLOBALE SUL TEST SET:\s*([0-9.]+)%"),
    "symbol": re.compile(r"SYMBOL ACCURACY SUL TEST SET \(simboli della formula.*?:\s*([0-9.]+)%"),
}
CAPTIONS = {
    "balanced": (r"\textbf{Grounder Evaluation: Reward-Class Balanced Accuracy (\%).} Per-task performance on the "
                 r"held-out test trajectories, across state representations (Symbolic vs.\ Image), reward functions "
                 r"(Three-value vs.\ Distance) and datasets (medium-expert, medium, medium-replay). The reward class "
                 r"of each step is predicted by running the probabilistic automaton on the symbol distributions "
                 r"output by the grounder; values report the balanced accuracy, i.e.\ the recall averaged over the "
                 r"reward classes present in the test set, which accounts for the strong class imbalance (e.g.\ the "
                 r"success class covers only a few percent of the steps)."),
    "symbol": (r"\textbf{Grounder Evaluation: Symbol Accuracy (\%).} Per-task performance on the held-out test "
               r"trajectories, across state representations (Symbolic vs.\ Image), reward functions (Three-value "
               r"vs.\ Distance) and datasets (medium-expert, medium, medium-replay). Values report the fraction of "
               r"steps in which the most probable symbol output by the grounder matches the true content of the "
               r"agent's cell, measured only on what the rewards can identify: objects not used by the formula "
               r"count as empty, and for symmetric formulas the most favourable permutation of the symbols that "
               r"leaves the reward of every sequence unchanged is taken."),
}
LABELS = {"balanced": "tab:grounder_reward_prediction", "symbol": "tab:grounder_symbol_accuracy"}


def find_seeds():
    return sorted({int(p.name[4:]) for p in GROUNDING_ROOT.glob("*/*/task*/seed*") if p.name[4:].isdigit()})


SEEDS = SEEDS if SEEDS is not None else find_seeds()


def value(state_type, reward_type, task, seed, dataset):
    """valore della metrica sul test set, oppure None se il run non e' (ancora) completo"""
    log = GROUNDING_ROOT / state_type / reward_type / f"task{task}" / f"seed{seed}" / "log" / f"log_{dataset}.txt"
    if not log.exists():
        return None
    match = PATTERNS[METRIC].search(log.read_text(encoding="utf-8"))
    return float(match.group(1)) if match else None


def fmt(values):
    if not values:
        return "--"
    if len(values) == 1:
        return f"${values[0]:.1f}$"
    return f"${np.mean(values):.1f}${{\\scriptsize$\\pm {np.std(values, ddof=1):.1f}$}}"


def per_seed(state_type, reward_type, tasks, dataset):
    """per ogni seed con tutti i task, la media sui task"""
    out = []
    for s in SEEDS:
        values = [value(state_type, reward_type, t, s, dataset) for t in tasks]
        if None not in values:
            out.append(float(np.mean(values)))
    return out


def block(state_type, title):
    """righe LaTeX di un tipo di stato (task con risultati, poi le medie dei GROUPS se ci sono tutti i task)"""
    lines = [f"\\multicolumn{{7}}{{c}}{{\\textbf{{Configuration: {title}}}}} \\\\", r"\midrule"]
    tasks = [t for t in TASKS if any(value(state_type, rt, t, s, ds) is not None
                                     for rt, _ in REWARD_TYPES for ds, _ in DATASETS for s in SEEDS)]
    for task in tasks:
        cells = [fmt(per_seed(state_type, rt, [task], ds)) for rt, _ in REWARD_TYPES for ds, _ in DATASETS]
        lines.append(f"{task} & " + " & ".join(cells) + r" \\")
    groups = [(label, g) for label, g in GROUPS if set(g) <= set(tasks)]
    for i, (label, group) in enumerate(groups):
        cells = [fmt(per_seed(state_type, rt, group, ds)) for rt, _ in REWARD_TYPES for ds, _ in DATASETS]
        lines += ([r"\midrule"] if i == 0 else []) + [f"{label} & " + " & ".join(cells) + r" \\"]
    return lines, tasks


def table():
    seeds_text = ", ".join(map(str, SEEDS))
    blocks = [(st, title, *block(st, title)) for st, title in STATE_TYPES]
    blocks = [b for b in blocks if b[3]]
    image = next((tasks for st, _, _, tasks in blocks if st == "Image"), [])
    seed_part = ((f" Mean $\\pm$ std over {len(SEEDS)} seeds ({seeds_text})" if len(SEEDS) > 1
                  else f" Seed {seeds_text}")
                 + "; the last rows of the symbolic block average over tasks 1--6 and over tasks 7--10, with the "
                   "std over seeds of the per-seed average."
                 + (f" Image states: only tasks {' and '.join(map(str, image))} (--: not run)." if image else ""))
    lines = [
        r"\begin{table}[p]",
        r"\centering",
        r"\small",
        f"\\caption{{{CAPTIONS[METRIC]}{seed_part}}}",
        f"\\label{{{LABELS[METRIC]}}}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{r ccc ccc}",
        r"\toprule",
        "& " + " & ".join(f"\\multicolumn{{3}}{{c}}{{\\textbf{{{name}}}}}" for _, name in REWARD_TYPES) + r" \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r"\textbf{Task} & " + " & ".join(f"\\textbf{{{name}}}" for _ in REWARD_TYPES for _, name in DATASETS) + r" \\",
        r"\midrule",
    ]
    for i, (_, _, rows, _) in enumerate(blocks):
        if i > 0:
            lines.append(r"\midrule[\heavyrulewidth]")
        lines += rows
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lines)


if __name__ == "__main__":
    out = Path(OUTPUT) if OUTPUT else GROUNDING_ROOT / "latex" / f"grounder_{METRIC}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    tex = table()
    header = (f"% Grounding, {METRIC} accuracy, seed {', '.join(map(str, SEEDS))} - generato da "
              f"Grounding/latex_grounding_table.py\n% servono \\usepackage{{booktabs,graphicx}}\n")
    out.write_text(header + tex + "\n")
    print(f"Grounding {METRIC}, seed {SEEDS}: tabella scritta in {out}")
