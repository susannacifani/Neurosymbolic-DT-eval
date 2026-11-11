"""Tabella LaTeX di confronto tra DT, DT+NRM e DT+Oracle (stato vero dell'automa).

Una sola tabella con un blocco per configurazione (Symbolic / Image x three_value / distance); in ogni
blocco, per ogni gruppo di task (1-6: solo visite; 7-10: anche vincoli di evitare) una riga per
approccio con la media sui task del gruppo. Colonne = per ogni dataset Reward, CDR e SR.
In ogni cella media +- deviazione standard tra i seed delle medie sui task del gruppo di ogni seed.
Con PER_TASK = True, invece, una tabella per configurazione con le righe di ogni singolo task
(1-10) e in fondo le due medie per gruppo (file comparison_epoch<E>_per_task.txt).
Le tabelle complete per task di ogni approccio si fanno con latex_dt_table.py.
La tabella viene scritta in un file .txt (in DT/latex/) da cui si incolla nel documento;
servono \\usepackage{booktabs,multirow,graphicx}.

Uso: imposta la CONFIGURAZIONE qui sotto e premi Run.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

# =====================================================================
# CONFIGURAZIONE: modifica qui cosa generare, poi premi Run
# =====================================================================
SEEDS = None             # es. [0, 1, 2]; None = i seed presenti per tutti e tre gli approcci
EPOCH = 200              # 200 = modelli finali; 50, 100 o 150 = modelli intermedi
PER_TASK = True          # True = una tabella per configurazione, con ogni task e in fondo le medie
OUTPUT = None            # None = DT/latex/comparison_epoch<E>.txt (o comparison_epoch<E>_per_task.txt)
# =====================================================================

K = 20                   # CONTEXT_LEN in decision_transformer.py
STATE_TYPES = [("Symbolic", "Symbolic States"), ("Image", "Image States")]
REWARD_TYPES = [("three_value", "Three-Value Reward"), ("distance", "Distance Reward")]
DATASETS = [("medium_expert", "medium-expert"), ("medium", "medium"), ("medium_replay", "medium-replay")]
TASK_GROUPS = [("Tasks 1--6", range(1, 7)), ("Tasks 7--10", range(7, 11))]
METRICS = [("Reward", lambda d: d["reward_puro"].mean()),
           ("CDR", lambda d: d["discounted_reward"].mean()),
           ("SR (\\%)", lambda d: 100.0 * d["success"].mean())]

DT_DIR = Path(__file__).resolve().parent
APPROACHES = [
    ("DT", DT_DIR),
    ("DT+NRM", DT_DIR / "automaton_state" / "grounder"),
    ("DT+Oracle", DT_DIR / "automaton_state" / "true"),
]
EVAL_DIR = "eval" if EPOCH == 200 else f"epoch{EPOCH:03d}/eval"


def find_seeds(root):
    return {int(p.name[4:]) for p in root.glob("*/*/task*/seed*") if p.name[4:].isdigit()}


SEEDS = SEEDS if SEEDS is not None else sorted(set.intersection(*(find_seeds(r) for _, r in APPROACHES)))


def value(root, state_type, reward_type, task, seed, dataset, metric):
    csv = root / state_type / reward_type / f"task{task}" / f"seed{seed}" / EVAL_DIR / f"eval_{dataset}.csv"
    return dict(METRICS)[metric](pd.read_csv(csv)) if csv.exists() else None


def fmt(values):
    if not values:
        return "--"
    if len(values) == 1:
        return f"${values[0]:.1f}$"
    return f"${np.mean(values):.1f}${{\\scriptsize$\\pm {np.std(values, ddof=1):.1f}$}}"


def test_episodes():
    """numero di episodi di test (dal primo eval_summary trovato)"""
    for summary in sorted(APPROACHES[0][1].glob("*/*/task*/seed*/eval/eval_summary_*.txt")):
        match = re.search(r"Episodes: (\d+)", summary.read_text())
        if match:
            return match[1]
    return "?"


def block(state_type, reward_type, title=None):
    """righe LaTeX di una configurazione e numero di medie incomplete (seed senza tutti i task)"""
    lines = [f"\\multicolumn{{11}}{{c}}{{\\textbf{{Configuration: {title}}}}} \\\\", r"\midrule"] if title else []
    incomplete = 0
    groups = TASK_GROUPS
    if PER_TASK:
        groups = [(f"Task {t}", [t]) for t in range(1, 11)] + [(f"Mean {g[6:]}", ts) for g, ts in TASK_GROUPS]
    for g, (group, tasks) in enumerate(groups):
        if g > 0:
            lines.append(r"\midrule" if PER_TASK and group.startswith("Mean") and not groups[g - 1][0].startswith("Mean")
                         else r"\cmidrule(lr){1-11}")
        for a, (name, root) in enumerate(APPROACHES):
            cells = []
            for ds, _ in DATASETS:
                for metric, _ in METRICS:
                    per_seed = []
                    for s in SEEDS:
                        values = [value(root, state_type, reward_type, t, s, ds, metric) for t in tasks]
                        if None not in values:
                            per_seed.append(np.mean(values))
                    incomplete += len(SEEDS) - len(per_seed)
                    cells.append(fmt(per_seed))
            first = f"\\multirow{{{len(APPROACHES)}}}{{*}}{{{group}}}" if a == 0 else ""
            lines.append(f"{first} & {name} & " + " & ".join(cells) + r" \\")
    return lines, incomplete


def table(configs):
    """configs: lista di (state_type, reward_type, titolo); una tabella con un blocco per configurazione"""
    seeds_text = ", ".join(map(str, SEEDS))
    n_seeds = f"{len(SEEDS)} seed{'s' if len(SEEDS) > 1 else ''}"
    seed_part = (f" Mean $\\pm$ std over {len(SEEDS)} seeds ({seeds_text})"
                 + (" (for the group means: of the per-seed average over the tasks of the group)." if PER_TASK
                    else " of the per-seed average over the tasks of the group.")
                 if len(SEEDS) > 1 else f" Seed {seeds_text}.")
    caption = (f"\\textbf{{DT vs.\\ DT+NRM vs.\\ DT+Oracle ($K={K}$, epoch {EPOCH}, {test_episodes()} test episodes, "
               f"{n_seeds}).}} Test episodes start from fixed positions on the 50 training layouts. "
               f"\\emph{{DT}}: Decision Transformer; \\emph{{DT+NRM}}: the DT also receives the automaton state "
               f"predicted by the NRM grounder, learned from rewards only (same task, dataset and seed); "
               f"\\emph{{DT+Oracle}}: the DT also receives the ground-truth automaton state, i.e.\\ the automaton "
               f"advances with the true symbols (a perfect grounder, not available in practice: an upper bound for "
               f"DT+NRM). All other settings are identical. Metrics: mean undiscounted return (\\emph{{Reward}}), "
               f"cumulative discounted reward ($\\gamma=0.99$, \\emph{{CDR}}), and success rate (\\emph{{SR}}, \\%), "
               f"{'for each task and ' if PER_TASK else ''}averaged over tasks 1--6 (visiting objects, possibly in order) "
               f"and over tasks 7--10 (also globally avoiding some objects, so an episode can end in a violation, "
               f"reward $-100$).{seed_part}")
    if PER_TASK:
        caption = f"\\textbf{{{configs[0][2]}.}} " + caption
    label = f"tab:comparison_epoch{EPOCH}" + (f"_{configs[0][0].lower()}_{configs[0][1]}" if PER_TASK else "")
    lines = [
        r"\begin{table}[p]",
        r"\centering",
        r"\small",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{ll ccc ccc ccc}",
        r"\toprule",
        "& & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{\\textbf{{{name}}}}}" for _, name in DATASETS) + r" \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}",
        r"\textbf{Tasks} & \textbf{Method} & " + " & ".join(m for _ in DATASETS for m, _ in METRICS) + r" \\",
        r"\midrule",
    ]
    incomplete = {}
    for c, (state_type, reward_type, title) in enumerate(configs):
        if c:
            lines.append(r"\midrule[\heavyrulewidth]")
        rows, incomplete[(state_type, reward_type)] = block(state_type, reward_type, None if PER_TASK else title)
        lines += rows
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lines), incomplete


if __name__ == "__main__":
    out = Path(OUTPUT) if OUTPUT else DT_DIR / "latex" / f"comparison_epoch{EPOCH}{'_per_task' if PER_TASK else ''}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    configs = [(st, rt, f"{st_title} --- {rt_title}") for st, st_title in STATE_TYPES for rt, rt_title in REWARD_TYPES]
    if PER_TASK:
        tables = [table([c]) for c in configs]
        tex = "\n\n".join(t for t, _ in tables)
        incomplete = {k: v for _, inc in tables for k, v in inc.items()}
    else:
        tex, incomplete = table(configs)
    header = (f"% DT vs DT+NRM vs DT+Oracle, K={K}, epoch {EPOCH}, seed {', '.join(map(str, SEEDS))} - generato da "
              f"DT/latex_dt_comparison.py\n% servono \\usepackage{{booktabs,multirow,graphicx}}\n")
    out.write_text(header + tex + "\n")
    for (state_type, reward_type), n in incomplete.items():
        if n:
            print(f"ATTENZIONE {state_type}/{reward_type}: {n} medie senza tutti i task (seed esclusi da quelle celle)")
    print(f"Confronto epoch {EPOCH}, seed {SEEDS}: tabella scritta in {out}")
