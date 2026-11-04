"""Tabella LaTeX dei risultati del Decision Transformer (gli stessi numeri di print_dt_table.py).

Una sola tabella con un blocco per configurazione (Symbolic / Image x three_value / distance): righe =
task, poi una riga di media per ogni gruppo di task (1-6: solo visite; 7-10: anche vincoli di evitare,
che si possono violare); colonne = per ogni dataset Reward, CDR e SR.
In ogni cella media +- deviazione standard tra i seed; nelle righe di media, media +- deviazione
standard tra i seed delle medie sui task del gruppo di ogni seed.
La tabella viene scritta in un file .txt (in DT/latex/) da cui si incolla nel documento;
servono \\usepackage{booktabs,graphicx}.

Uso: imposta la CONFIGURAZIONE qui sotto e premi Run.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

# =====================================================================
# CONFIGURAZIONE: modifica qui cosa generare, poi premi Run
# =====================================================================
SEEDS = None             # es. [0] oppure [0, 1, 2]; None = tutti i seed trovati
K = None                 # None = risultati principali (K = 50); es. 5 = ablation sul contesto
AUTOMATON = None         # None = DT standard; "grounder" = DT + NRM; "true" = DT + stato vero dell'automa
OUTPUT = None            # None = DT/latex/dt_<approccio>_K<k>.txt
# =====================================================================

K_FULL = 50              # contesto del DT standard (CONTEXT_LEN in decision_transformer.py)
STATE_TYPES = [("Symbolic", "Symbolic States"), ("Image", "Image States")]
REWARD_TYPES = [("three_value", "Three-Value Reward"), ("distance", "Distance Reward")]
DATASETS = [("medium_expert", "medium-expert"), ("medium", "medium"), ("medium_replay", "medium-replay")]
TASKS = range(1, 11)
# righe di media: task con sole visite (1-6) e task con anche vincoli G(!c) (7-10)
TASK_GROUPS = [("Mean 1--6", range(1, 7)), ("Mean 7--10", range(7, 11))]
METRICS = [("Reward", lambda d: d["reward_puro"].mean()),
           ("CDR", lambda d: d["discounted_reward"].mean()),
           ("SR (\\%)", lambda d: 100.0 * d["success"].mean())]

DT_DIR = Path(__file__).resolve().parent
k = K if K is not None else K_FULL
if AUTOMATON is not None:
    ROOT = DT_DIR / "automaton_state" / AUTOMATON / f"K{k}"
elif K is not None:
    ROOT = DT_DIR / "context_ablation" / f"K{K}"
else:
    ROOT = DT_DIR
APPROACH = {None: "DT", "grounder": "DT+NRM", "true": "DT+Oracle"}[AUTOMATON]
APPROACH_TEXT = {None: "",
                 "grounder": " The DT also receives the automaton state predicted by the NRM grounder (learned "
                             "from rewards only; same task, dataset and seed as the DT).",
                 "true": " The DT also receives the ground-truth automaton state (oracle: the automaton advances with the true symbols, i.e. a perfect grounder)."}[AUTOMATON]
SEEDS = SEEDS if SEEDS is not None else sorted(
    {int(p.name[4:]) for p in ROOT.glob("*/*/task*/seed*") if p.name[4:].isdigit()})
SLUG = APPROACH.lower().replace("+", "_").replace(" ", "_")


def value(state_type, reward_type, task, seed, dataset, metric):
    csv = ROOT / state_type / reward_type / f"task{task}" / f"seed{seed}" / "eval" / f"eval_{dataset}.csv"
    return dict(METRICS)[metric](pd.read_csv(csv)) if csv.exists() else None


def fmt(values):
    if not values:
        return "--"
    if len(values) == 1:
        return f"${values[0]:.1f}$"
    return f"${np.mean(values):.1f}${{\\scriptsize$\\pm {np.std(values, ddof=1):.1f}$}}"


def test_episodes():
    """numero di episodi di test (dal primo eval_summary trovato)"""
    for summary in sorted(ROOT.glob("*/*/task*/seed*/eval/eval_summary_*.txt")):
        match = re.search(r"Episodes: (\d+)", summary.read_text())
        if match:
            return match[1]
    return "?"


def block(state_type, reward_type, title):
    """righe LaTeX di una configurazione (task e medie dei gruppi) e numero di valutazioni mancanti"""
    lines = [f"\\multicolumn{{10}}{{c}}{{\\textbf{{Configuration: {title}}}}} \\\\", r"\midrule"]
    missing = 0
    for task in TASKS:
        cells = []
        for ds, _ in DATASETS:
            for metric, _ in METRICS:
                values = [v for s in SEEDS if (v := value(state_type, reward_type, task, s, ds, metric)) is not None]
                missing += len(SEEDS) - len(values)
                cells.append(fmt(values))
        lines.append(f"{task} & " + " & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    # media sui task del gruppo per ogni seed (solo i seed con tutti i task), poi media +- std tra i seed
    for label, tasks in TASK_GROUPS:
        cells = []
        for ds, _ in DATASETS:
            for metric, _ in METRICS:
                per_seed = []
                for s in SEEDS:
                    values = [value(state_type, reward_type, t, s, ds, metric) for t in tasks]
                    if None not in values:
                        per_seed.append(np.mean(values))
                cells.append(fmt(per_seed))
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    return lines, missing


def table():
    seeds_text = ", ".join(map(str, SEEDS))
    n_seeds = f"{len(SEEDS)} seed{'s' if len(SEEDS) > 1 else ''}"
    seed_part = (f" Mean $\\pm$ std over {len(SEEDS)} seeds ({seeds_text}); in the mean rows, std across seeds of "
                 f"the per-seed average over the tasks of the group." if len(SEEDS) > 1 else f" Seed {seeds_text}.")
    caption = (f"\\textbf{{{APPROACH} Complete Evaluation ($K={k}$, {test_episodes()} test layouts, {n_seeds}).}} "
               f"Per-task results across all combinations of state representations (Symbolic vs.\\ Image) and "
               f"reward functions (Three-value vs.\\ Distance).{APPROACH_TEXT} Metrics reported: mean undiscounted "
               f"return (\\emph{{Reward}}), cumulative discounted reward ($\\gamma=0.99$, \\emph{{CDR}}), and success "
               f"rate (\\emph{{SR}}, \\%). Tasks 1--6 only require visiting objects (possibly in order); tasks 7--10 "
               f"also require globally avoiding some objects, so an episode can end in a violation (reward $-100$): "
               f"the two groups are averaged separately.{seed_part}")
    lines = [
        r"\begin{table}[p]",
        r"\centering",
        r"\small",
        f"\\caption{{{caption}}}",
        f"\\label{{tab:{SLUG}_k{k}_all_complete}}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{r ccc ccc ccc}",
        r"\toprule",
        "& " + " & ".join(f"\\multicolumn{{3}}{{c}}{{\\textbf{{{name}}}}}" for _, name in DATASETS) + r" \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
        r"\textbf{Task} & " + " & ".join(m for _ in DATASETS for m, _ in METRICS) + r" \\",
        r"\midrule",
    ]
    missing = {}
    first = True
    for state_type, state_title in STATE_TYPES:
        for reward_type, reward_title in REWARD_TYPES:
            if not first:
                lines.append(r"\midrule[\heavyrulewidth]")
            first = False
            rows, missing[(state_type, reward_type)] = block(state_type, reward_type,
                                                             f"{state_title} --- {reward_title}")
            lines += rows
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lines), missing


if __name__ == "__main__":
    out = Path(OUTPUT) if OUTPUT else DT_DIR / "latex" / f"{SLUG}_K{k}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    tex, missing = table()
    header = (f"% {APPROACH}, K={k}, seed {', '.join(map(str, SEEDS))} - generato da DT/latex_dt_table.py\n"
              f"% servono \\usepackage{{booktabs,graphicx}}\n")
    out.write_text(header + tex + "\n")
    for (state_type, reward_type), n in missing.items():
        if n:
            print(f"ATTENZIONE {state_type}/{reward_type}: mancano {n} valutazioni (task x dataset x seed x metrica)")
    print(f"{APPROACH}, K={k}, seed {SEEDS}: tabella scritta in {out}")
