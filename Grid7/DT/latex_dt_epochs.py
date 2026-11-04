"""Tabella LaTeX della convergenza: best epoch di DT, DT+NRM e DT+Oracle (stato vero dell'automa) a parita'
di contesto K (gli stessi numeri di print_dt_epochs.py).

Una sola tabella con un blocco per configurazione (Symbolic / Image x three_value / distance) e una riga
per approccio; colonne = dataset e media sui tre dataset. In ogni cella media +- deviazione standard tra
i seed della media sui task di ogni seed.
La best epoch e' quella con la validation loss minima (riga "Best epoch" del log di training).
La tabella viene scritta in un file .txt (in DT/latex/) da cui si incolla nel documento;
servono \\usepackage{booktabs,graphicx}.

Uso: imposta la CONFIGURAZIONE qui sotto e premi Run.
"""
import re
from pathlib import Path

import numpy as np

# =====================================================================
# CONFIGURAZIONE: modifica qui cosa generare, poi premi Run
# =====================================================================
SEEDS = None             # es. [0] oppure [0, 1, 2]; None = i seed presenti per tutti e tre gli approcci
K = 50                   # contesto (per K = 50 il DT standard e' in DT/, per gli altri K in context_ablation/)
OUTPUT = None            # None = DT/latex/epochs_K<k>.txt
# =====================================================================

K_FULL = 50              # contesto del DT standard (CONTEXT_LEN in decision_transformer.py)
STATE_TYPES = [("Symbolic", "Symbolic States"), ("Image", "Image States")]
REWARD_TYPES = [("three_value", "Three-Value Reward"), ("distance", "Distance Reward")]
DATASETS = [("medium_expert", "medium-expert"), ("medium", "medium"), ("medium_replay", "medium-replay")]
TASKS = range(1, 11)

DT_DIR = Path(__file__).resolve().parent
APPROACHES = [
    ("DT", DT_DIR if K == K_FULL else DT_DIR / "context_ablation" / f"K{K}"),
    ("DT+NRM", DT_DIR / "automaton_state" / "grounder" / f"K{K}"),
    ("DT+Oracle", DT_DIR / "automaton_state" / "true" / f"K{K}"),
]
BEST_PATTERN = re.compile(r"Best epoch: (\d+)")
CONFIG_PATTERN = re.compile(r"max_epochs=(\d+), patience=(\d+)")


def find_seeds(root):
    return {int(p.name[4:]) for p in root.glob("*/*/task*/seed*") if p.name[4:].isdigit()}


SEEDS = SEEDS if SEEDS is not None else sorted(set.intersection(*(find_seeds(r) for _, r in APPROACHES)))


def log_path(root, state_type, reward_type, task, seed, dataset):
    return root / state_type / reward_type / f"task{task}" / f"seed{seed}" / "log" / f"train_log_{dataset}.txt"


def best_epoch(root, state_type, reward_type, task, seed, dataset):
    log = log_path(root, state_type, reward_type, task, seed, dataset)
    if not log.exists():
        return None
    match = BEST_PATTERN.search(log.read_text())
    return int(match[1]) if match else None


def fmt(values):
    if not values:
        return "--"
    if len(values) == 1:
        return f"${values[0]:.1f}$"
    return f"${np.mean(values):.1f}${{\\scriptsize$\\pm {np.std(values, ddof=1):.1f}$}}"


def early_stopping():
    """max_epochs e patience (dal primo log di training trovato)"""
    for log in sorted(APPROACHES[0][1].glob("*/*/task*/seed*/log/train_log_*.txt")):
        match = CONFIG_PATTERN.search(log.read_text())
        if match:
            return match[1], match[2]
    return "?", "?"


def cell(root, state_type, reward_type, datasets):
    """media +- std tra i seed della media (sui task e sui dataset dati) di ogni seed; seed incompleti esclusi"""
    per_seed = []
    for s in SEEDS:
        values = [best_epoch(root, state_type, reward_type, t, s, ds) for t in TASKS for ds in datasets]
        if None not in values:
            per_seed.append(np.mean(values))
    return fmt(per_seed), len(SEEDS) - len(per_seed)


def table():
    max_epochs, patience = early_stopping()
    seeds_text = ", ".join(map(str, SEEDS))
    n_seeds = f"{len(SEEDS)} seed{'s' if len(SEEDS) > 1 else ''}"
    seed_part = (f" Mean $\\pm$ std over {len(SEEDS)} seeds ({seeds_text}) of the per-seed average over the "
                 f"{len(TASKS)} tasks (\\emph{{All}}: over the {len(TASKS)} tasks and the {len(DATASETS)} datasets)."
                 if len(SEEDS) > 1 else f" Seed {seeds_text}.")
    caption = (f"\\textbf{{Convergence: Best Training Epoch ($K={K}$, {n_seeds}).}} Epoch with the minimum "
               f"validation loss (early stopping with patience {patience}, at most {max_epochs} epochs) for the DT, "
               f"the DT with the automaton state predicted by the NRM grounder (\\emph{{DT+NRM}}) and the DT with "
               f"the ground-truth automaton state (\\emph{{DT+Oracle}}); all other settings are identical. "
               f"Lower is faster.{seed_part}")
    lines = [
        r"\begin{table}[p]",
        r"\centering",
        r"\small",
        f"\\caption{{{caption}}}",
        f"\\label{{tab:epochs_k{K}}}",
        r"\begin{tabular}{l cccc}",
        r"\toprule",
        r"\textbf{Method} & " + " & ".join(f"\\textbf{{{name}}}" for _, name in DATASETS) + r" & \textbf{All} \\",
        r"\midrule",
    ]
    incomplete = 0
    first = True
    for state_type, state_title in STATE_TYPES:
        for reward_type, reward_title in REWARD_TYPES:
            if not first:
                lines.append(r"\midrule[\heavyrulewidth]")
            first = False
            lines += [f"\\multicolumn{{5}}{{c}}{{\\textbf{{Configuration: {state_title} --- {reward_title}}}}} \\\\",
                      r"\midrule"]
            for name, root in APPROACHES:
                cells = []
                for datasets in [[ds] for ds, _ in DATASETS] + [[ds for ds, _ in DATASETS]]:
                    text, n = cell(root, state_type, reward_type, datasets)
                    cells.append(text)
                    incomplete += n
                lines.append(f"{name} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines), incomplete


if __name__ == "__main__":
    out = Path(OUTPUT) if OUTPUT else DT_DIR / "latex" / f"epochs_K{K}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    tex, incomplete = table()
    header = (f"% Best epoch DT vs DT+NRM vs DT+Oracle, K={K}, seed {', '.join(map(str, SEEDS))} - generato da "
              f"DT/latex_dt_epochs.py\n% servono \\usepackage{{booktabs}}\n")
    out.write_text(header + tex + "\n")
    if incomplete:
        print(f"ATTENZIONE: {incomplete} celle hanno seed esclusi perche' mancano dei run")
    print(f"Best epoch K={K}, seed {SEEDS}: tabella scritta in {out}")
