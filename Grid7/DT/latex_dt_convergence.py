"""Tabelle LaTeX della convergenza per task: success rate di DT, DT+NRM e DT+Oracle alle epoch 50, 100, 150 e
200, una tabella per reward.

Righe: blocco Symbolic (task 1-10, poi le medie sui task 1-6 e 7-10) e blocco Image (solo i task con risultati),
per ogni riga i tre metodi; colonne: per ogni dataset le 4 epoch. In ogni cella media +- deviazione standard tra i
seed; nelle medie, tra i seed delle medie sui task del gruppo.
Output: DT/latex/convergence_per_task_<reward>.txt; servono \\usepackage{booktabs,multirow,adjustbox,rotating}.
Uso: python3 DT/latex_dt_convergence.py
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

# =====================================================================
# CONFIGURAZIONE
# =====================================================================
SEEDS = [0, 1, 2]
# =====================================================================

EPOCHS = [50, 100, 150, 200]
DT_DIR = Path(__file__).resolve().parent
APPROACHES = [("DT", DT_DIR), ("DT+NRM", DT_DIR / "automaton_state" / "grounder"),
              ("DT+Oracle", DT_DIR / "automaton_state" / "true")]
REWARD_TYPES = [("three_value", "Three-Value Reward"), ("distance", "Distance Reward")]
STATE_TYPES = [("Symbolic", "Symbolic States"), ("Image", "Image States")]
DATASETS = [("medium_expert", "medium-expert"), ("medium", "medium"), ("medium_replay", "medium-replay")]
# righe di media del blocco Symbolic
GROUPS = [("\\shortstack[l]{Mean\\\\(tasks 1--6)}", [1, 2, 3, 4, 5, 6]),
          ("\\shortstack[l]{Mean\\\\(tasks 7--10)}", [7, 8, 9, 10])]


@lru_cache(maxsize=None)
def sr(root, state_type, reward_type, task, seed, dataset, epoch):
    sub = "eval" if epoch == 200 else f"epoch{epoch:03d}/eval"
    csv = root / state_type / reward_type / f"task{task}" / f"seed{seed}" / sub / f"eval_{dataset}.csv"
    return 100.0 * pd.read_csv(csv)["success"].mean() if csv.exists() else None


def per_seed(root, state_type, reward_type, tasks, dataset, epoch):
    """per ogni seed con tutti i task, la media sui task (con un task solo: il valore del seed)"""
    out = []
    for s in SEEDS:
        values = [sr(root, state_type, reward_type, t, s, dataset, epoch) for t in tasks]
        if None not in values:
            out.append(float(np.mean(values)))
    return out


def row_cells(state_type, reward_type, tasks):
    rows = {name: [] for name, _ in APPROACHES}
    for ds, _ in DATASETS:
        for e in EPOCHS:
            for name, root in APPROACHES:
                v = per_seed(root, state_type, reward_type, tasks, ds, e)
                if not v:
                    rows[name].append("--")
                    continue
                text = f"${np.mean(v):.1f}$"
                if len(v) > 1:
                    text += f"{{\\scriptsize$\\pm {np.std(v, ddof=1):.1f}$}}"
                rows[name].append(text)
    return rows


def tasks_with_results(state_type, reward_type):
    return [t for t in range(1, 11) if any(sr(root, state_type, reward_type, t, s, ds, 200) is not None
                                           for _, root in APPROACHES for ds, _ in DATASETS for s in SEEDS)]


def table(reward_type, reward_title):
    n_cols = 2 + len(DATASETS) * len(EPOCHS)
    blocks = [(st, title, tasks_with_results(st, reward_type)) for st, title in STATE_TYPES]
    blocks = [b for b in blocks if b[2]]
    image = next((tasks for st, _, tasks in blocks if st == "Image"), [])
    caption = (f"\\textbf{{Convergence, {reward_title} ($K=20$, 500 test episodes, {len(SEEDS)} seeds).}} Success rate "
               f"(\\%) of the models saved at epochs {', '.join(map(str, EPOCHS))} of training. Mean $\\pm$ std over "
               f"seeds {', '.join(map(str, SEEDS))}; the last rows of the symbolic block average over tasks 1--6 and "
               f"over tasks 7--10, with the std over seeds of the per-seed average."
               + (f" Image states: only tasks {' and '.join(map(str, image))} on medium and medium-replay (--: not run)."
                  if image else ""))
    lines = [
        r"\begin{sidewaystable}", r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{tab:convergence_{reward_type}}}",
        r"\begin{adjustbox}{max width=\textwidth, max totalheight=0.62\paperwidth}",
        r"\begin{tabular}{ll " + " ".join("c" * len(EPOCHS) for _ in DATASETS) + "}",
        r"\toprule",
        "& & " + " & ".join(f"\\multicolumn{{{len(EPOCHS)}}}{{c}}{{\\textbf{{{name}}}}}" for _, name in DATASETS) + r" \\",
        "".join(f"\\cmidrule(lr){{{3 + k * len(EPOCHS)}-{2 + (k + 1) * len(EPOCHS)}}}" for k in range(len(DATASETS))),
        r"\textbf{Task} & \textbf{Method} & " + " & ".join(f"ep.\\ {e}" for _ in DATASETS for e in EPOCHS) + r" \\",
        r"\midrule",
    ]
    for b, (state_type, title, tasks) in enumerate(blocks):
        if b:
            lines.append(r"\midrule[\heavyrulewidth]")
        lines += [f"\\multicolumn{{{n_cols}}}{{c}}{{\\textbf{{{title}}}}} \\\\", r"\midrule"]
        rows_spec = [(str(t), [t]) for t in tasks] + (GROUPS if state_type == "Symbolic" else [])
        for i, (label, group) in enumerate(rows_spec):
            if i:
                lines.append(r"\midrule" if i == len(tasks) else f"\\cmidrule(lr){{1-{n_cols}}}")
            rows = row_cells(state_type, reward_type, group)
            for a, (name, _) in enumerate(APPROACHES):
                first = f"\\multirow{{{len(APPROACHES)}}}{{*}}{{{label}}}" if a == 0 else ""
                lines.append(f"{first} & {name} & " + " & ".join(rows[name]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{sidewaystable}"]
    return "\n".join(lines)


if __name__ == "__main__":
    (DT_DIR / "latex").mkdir(exist_ok=True)
    preamble = "% generato da DT/latex_dt_convergence.py; servono \\usepackage{booktabs,multirow,adjustbox,rotating}\n"
    for rt, title in REWARD_TYPES:
        out = DT_DIR / "latex" / f"convergence_per_task_{rt}.txt"
        out.write_text(preamble + table(rt, title) + "\n")
        print(f"{rt}: tabella scritta in {out}")
