"""Tabella LaTeX di confronto tra DT, DT+NRM e DT+Oracle (stato vero dell'automa) a parita' di contesto K.

Una sola tabella con un blocco per configurazione (Symbolic / Image x three_value / distance); in ogni
blocco, per ogni gruppo di task (1-6: solo visite; 7-10: anche vincoli di evitare) una riga per
approccio con la media sui task del gruppo. Colonne = per ogni dataset Reward, CDR e SR.
In ogni cella media +- deviazione standard tra i seed delle medie sui task del gruppo di ogni seed.
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
SEEDS = None             # es. [0] oppure [0, 1, 2]; None = i seed presenti per tutti e tre gli approcci
K = 50                   # contesto (per K = 50 il DT standard e' in DT/, per gli altri K in context_ablation/)
OUTPUT = None            # None = DT/latex/comparison_K<k>.txt
# =====================================================================

K_FULL = 50              # contesto del DT standard (CONTEXT_LEN in decision_transformer.py)
STATE_TYPES = [("Symbolic", "Symbolic States"), ("Image", "Image States")]
REWARD_TYPES = [("three_value", "Three-Value Reward"), ("distance", "Distance Reward")]
DATASETS = [("medium_expert", "medium-expert"), ("medium", "medium"), ("medium_replay", "medium-replay")]
TASK_GROUPS = [("Tasks 1--6", range(1, 7)), ("Tasks 7--10", range(7, 11))]
METRICS = [("Reward", lambda d: d["reward_puro"].mean()),
           ("CDR", lambda d: d["discounted_reward"].mean()),
           ("SR (\\%)", lambda d: 100.0 * d["success"].mean())]

DT_DIR = Path(__file__).resolve().parent
APPROACHES = [
    ("DT", DT_DIR if K == K_FULL else DT_DIR / "context_ablation" / f"K{K}"),
    ("DT+NRM", DT_DIR / "automaton_state" / "grounder" / f"K{K}"),
    ("DT+Oracle", DT_DIR / "automaton_state" / "true" / f"K{K}"),
]


def find_seeds(root):
    return {int(p.name[4:]) for p in root.glob("*/*/task*/seed*") if p.name[4:].isdigit()}


SEEDS = SEEDS if SEEDS is not None else sorted(set.intersection(*(find_seeds(r) for _, r in APPROACHES)))


def value(root, state_type, reward_type, task, seed, dataset, metric):
    csv = root / state_type / reward_type / f"task{task}" / f"seed{seed}" / "eval" / f"eval_{dataset}.csv"
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


def block(state_type, reward_type, title):
    """righe LaTeX di una configurazione e numero di medie incomplete (seed senza tutti i task)"""
    lines = [f"\\multicolumn{{11}}{{c}}{{\\textbf{{Configuration: {title}}}}} \\\\", r"\midrule"]
    incomplete = 0
    for g, (group, tasks) in enumerate(TASK_GROUPS):
        if g > 0:
            lines.append(r"\cmidrule(lr){1-11}")
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


def table():
    seeds_text = ", ".join(map(str, SEEDS))
    n_seeds = f"{len(SEEDS)} seed{'s' if len(SEEDS) > 1 else ''}"
    seed_part = (f" Mean $\\pm$ std over {len(SEEDS)} seeds ({seeds_text}) of the per-seed average over the tasks "
                 f"of the group." if len(SEEDS) > 1 else f" Seed {seeds_text}.")
    caption = (f"\\textbf{{DT vs.\\ DT+NRM vs.\\ DT+Oracle ($K={K}$, {test_episodes()} test layouts, {n_seeds}).}} "
               f"\\emph{{DT}}: Decision Transformer; \\emph{{DT+NRM}}: the DT also receives the automaton state "
               f"predicted by the NRM grounder, learned from rewards only (same task, dataset and seed); "
               f"\\emph{{DT+Oracle}}: the DT also receives the ground-truth automaton state, i.e.\\ the automaton "
               f"advances with the true symbols (a perfect grounder, not available in practice: an upper bound for "
               f"DT+NRM). All other settings are identical. Metrics: mean undiscounted return (\\emph{{Reward}}), "
               f"cumulative discounted reward ($\\gamma=0.99$, \\emph{{CDR}}), and success rate (\\emph{{SR}}, \\%), "
               f"averaged over tasks 1--6 (visiting objects, possibly in order) and over tasks 7--10 (also "
               f"globally avoiding some objects, so an episode can end in a violation, reward $-100$).{seed_part}")
    lines = [
        r"\begin{table}[p]",
        r"\centering",
        r"\small",
        f"\\caption{{{caption}}}",
        f"\\label{{tab:comparison_k{K}}}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{ll ccc ccc ccc}",
        r"\toprule",
        "& & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{\\textbf{{{name}}}}}" for _, name in DATASETS) + r" \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}",
        r"\textbf{Tasks} & \textbf{Method} & " + " & ".join(m for _ in DATASETS for m, _ in METRICS) + r" \\",
        r"\midrule",
    ]
    incomplete = {}
    first = True
    for state_type, state_title in STATE_TYPES:
        for reward_type, reward_title in REWARD_TYPES:
            if not first:
                lines.append(r"\midrule[\heavyrulewidth]")
            first = False
            rows, incomplete[(state_type, reward_type)] = block(state_type, reward_type,
                                                                f"{state_title} --- {reward_title}")
            lines += rows
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lines), incomplete


if __name__ == "__main__":
    out = Path(OUTPUT) if OUTPUT else DT_DIR / "latex" / f"comparison_K{K}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    tex, incomplete = table()
    header = (f"% DT vs DT+NRM vs DT+Oracle, K={K}, seed {', '.join(map(str, SEEDS))} - generato da "
              f"DT/latex_dt_comparison.py\n% servono \\usepackage{{booktabs,multirow,graphicx}}\n")
    out.write_text(header + tex + "\n")
    for (state_type, reward_type), n in incomplete.items():
        if n:
            print(f"ATTENZIONE {state_type}/{reward_type}: {n} medie senza tutti i task (seed esclusi da quelle celle)")
    print(f"Confronto K={K}, seed {SEEDS}: tabella scritta in {out}")
