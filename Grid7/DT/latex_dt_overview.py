"""Tabella LaTeX riassuntiva: DT, DT+NRM e DT+Oracle su tutti i task, in una pagina (orizzontale).

Colonne: three_value e distance, per ognuno i 3 dataset, per ognuno Reward, CDR e SR. Righe: blocco
Symbolic (task 1-10, poi le medie sui task 1-6 e 7-10) e blocco Image (solo i task con
risultati). In ogni cella media +- deviazione standard tra i seed; nelle medie, tra i seed delle medie sui
task del gruppo. Il migliore dei tre metodi e' in \\best, il secondo in \\second (a parita' al primo posto
nessun secondo).
La tabella viene scritta in DT/latex/overview_epoch<E>.txt; servono
\\usepackage{booktabs,multirow,adjustbox,rotating}.
Uso: imposta la CONFIGURAZIONE e premi Run, oppure python3 DT/latex_dt_overview.py [epoch]
"""
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

# =====================================================================
# CONFIGURAZIONE
# =====================================================================
SEEDS = [0, 1, 2]
EPOCH = int(sys.argv[1]) if len(sys.argv) > 1 else 200   # 200 = modelli finali; 50, 100, 150 = intermedi
# =====================================================================

K = 20
DT_DIR = Path(__file__).resolve().parent
APPROACHES = [("DT", DT_DIR), ("DT+NRM", DT_DIR / "automaton_state" / "grounder"),
              ("DT+Oracle", DT_DIR / "automaton_state" / "true")]
REWARD_TYPES = [("three_value", "Three-Value Reward"), ("distance", "Distance Reward")]
DATASETS = [("medium_expert", "medium-expert"), ("medium", "medium"), ("medium_replay", "medium-replay")]
METRICS = [("Reward", lambda d: d["reward_puro"].mean()), ("CDR", lambda d: d["discounted_reward"].mean()),
           ("SR (\\%)", lambda d: 100.0 * d["success"].mean())]
EVAL_DIR = "eval" if EPOCH == 200 else f"epoch{EPOCH:03d}/eval"
# righe di media del blocco Symbolic
GROUPS = [("\\shortstack[l]{Mean\\\\(tasks 1--6)}", [1, 2, 3, 4, 5, 6]),
          ("\\shortstack[l]{Mean\\\\(tasks 7--10)}", [7, 8, 9, 10])]


@lru_cache(maxsize=None)
def value(root, state_type, reward_type, task, seed, dataset, metric):
    csv = root / state_type / reward_type / f"task{task}" / f"seed{seed}" / EVAL_DIR / f"eval_{dataset}.csv"
    return dict(METRICS)[metric](pd.read_csv(csv)) if csv.exists() else None


def per_seed(root, state_type, reward_type, tasks, dataset, metric):
    """per ogni seed con tutti i task, la media sui task (con un task solo: il valore del seed)"""
    out = []
    for s in SEEDS:
        values = [value(root, state_type, reward_type, t, s, dataset, metric) for t in tasks]
        if None not in values:
            out.append(float(np.mean(values)))
    return out


def cells(state_type, tasks):
    """per ogni metodo la lista delle celle LaTeX della riga, con l'evidenziazione"""
    rows = {name: [] for name, _ in APPROACHES}
    for rt, _ in REWARD_TYPES:
        for ds, _ in DATASETS:
            for metric, _ in METRICS:
                stats = {name: per_seed(root, state_type, rt, tasks, ds, metric) for name, root in APPROACHES}
                means = {n: round(float(np.mean(v)), 1) for n, v in stats.items() if v}
                best = max(means.values()) if means else None
                n_best = sum(m == best for m in means.values())
                rest = [m for m in means.values() if m != best]
                second = max(rest) if rest and n_best == 1 else None
                for name, v in stats.items():
                    if not v:
                        rows[name].append("--")
                        continue
                    text = f"${means[name]:.1f}$"
                    if len(v) > 1:
                        text += f"{{\\scriptsize$\\pm {np.std(v, ddof=1):.1f}$}}"
                    if means[name] == best and len(means) > 1:
                        text = f"\\best{{{text}}}"
                    elif means[name] == second:
                        text = f"\\second{{{text}}}"
                    rows[name].append(text)
    return rows


def block(state_type, title, tasks, groups=()):
    n_cols = 2 + len(REWARD_TYPES) * len(DATASETS) * len(METRICS)
    lines = [f"\\multicolumn{{{n_cols}}}{{c}}{{\\textbf{{{title}}}}} \\\\", r"\midrule"]
    rows_spec = [(str(t), [t]) for t in tasks] + list(groups)
    for i, (label, group) in enumerate(rows_spec):
        if i:
            lines.append(r"\midrule" if i == len(tasks) else f"\\cmidrule(lr){{1-{n_cols}}}")
        rows = cells(state_type, group)
        for a, (name, _) in enumerate(APPROACHES):
            first = f"\\multirow{{{len(APPROACHES)}}}{{*}}{{{label}}}" if a == 0 else ""
            lines.append(f"{first} & {name} & " + " & ".join(rows[name]) + r" \\")
    return lines


def image_tasks():
    return [t for t in range(1, 11) if any(
        value(root, "Image", rt, t, s, ds, METRICS[-1][0]) is not None
        for _, root in APPROACHES for rt, _ in REWARD_TYPES for ds, _ in DATASETS for s in SEEDS)]


def table():
    n_data = len(DATASETS) * len(METRICS)
    img = image_tasks()
    caption = (f"\\textbf{{DT vs.\\ DT+NRM vs.\\ DT+Oracle, all tasks ($K={K}$, epoch {EPOCH}, 500 test episodes, "
               f"{len(SEEDS)} seeds).}} Test episodes start from fixed positions on the 50 training layouts. "
               f"\\emph{{DT+NRM}}: the DT also receives the automaton state predicted by the NRM grounder, learned "
               f"from rewards only (same task, dataset and seed); \\emph{{DT+Oracle}}: the DT receives the "
               f"ground-truth automaton state (upper bound for DT+NRM). Metrics: mean undiscounted return "
               f"(\\emph{{Reward}}), cumulative discounted reward ($\\gamma=0.99$, \\emph{{CDR}}) and success rate "
               f"(\\emph{{SR}}, \\%). Mean $\\pm$ std over seeds {', '.join(map(str, SEEDS))}; the last rows of the "
               f"symbolic block average over tasks 1--6 and over tasks 7--10, with the std over seeds of the per-seed "
               f"average. "
               f"Best of the three methods in bold, second underlined. Image states: only tasks "
               f"{' and '.join(map(str, img))} with the distance reward on medium and medium-replay (--: not run).")
    header = [
        r"\begin{sidewaystable}",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{tab:overview_epoch{EPOCH}}}",
        r"\begin{adjustbox}{max width=\textwidth, max totalheight=0.62\paperwidth}",
        r"\begin{tabular}{ll " + " ".join("c" * n_data for _ in REWARD_TYPES) + "}",
        r"\toprule",
        "& & " + " & ".join(f"\\multicolumn{{{n_data}}}{{c}}{{\\textbf{{{name}}}}}" for _, name in REWARD_TYPES) + r" \\",
        "".join(f"\\cmidrule(lr){{{3 + i * n_data}-{2 + (i + 1) * n_data}}}" for i in range(len(REWARD_TYPES))),
        "& & " + " & ".join(f"\\multicolumn{{{len(METRICS)}}}{{c}}{{{name}}}" for _ in REWARD_TYPES for _, name in DATASETS) + r" \\",
        "".join(f"\\cmidrule(lr){{{3 + i * len(METRICS)}-{2 + (i + 1) * len(METRICS)}}}"
                for i in range(len(REWARD_TYPES) * len(DATASETS))),
        r"\textbf{Task} & \textbf{Method} & " + " & ".join(m for _ in range(len(REWARD_TYPES) * len(DATASETS)) for m, _ in METRICS) + r" \\",
        r"\midrule",
    ]
    lines = header + block("Symbolic", "Symbolic States", range(1, 11), GROUPS)
    if img:
        lines += [r"\midrule[\heavyrulewidth]"] + block("Image", "Image States", img)
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{sidewaystable}"]
    return "\n".join(lines)


if __name__ == "__main__":
    out = DT_DIR / "latex" / f"overview_epoch{EPOCH}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    preamble = ("% generato da DT/latex_dt_overview.py; servono \\usepackage{booktabs,multirow,adjustbox,rotating}\n"
                "\\providecommand{\\best}[1]{\\textbf{#1}}\n\\providecommand{\\second}[1]{\\underline{#1}}\n")
    out.write_text(preamble + table() + "\n")
    print(f"epoch {EPOCH}: tabella scritta in {out}")
