"""Curve di apprendimento di DT, DT+NRM e DT+Oracle: metriche dei modelli alle epoch 50, 100, 150 e 200.

Figure: per ogni gruppo di task (1-6, 7-10) e per ogni metrica in PLOT_METRICS una figura con una riga per
reward e una colonna per dataset; in ogni pannello i tre metodi (media tra i seed delle medie sui task del gruppo,
banda = +- deviazione standard tra i seed). Con PER_TASK = True anche le stesse figure per ogni singolo task, in
DT/latex/convergence_per_task/ (per l'analisi).
Tabella LaTeX: success rate, stessi numeri delle figure, per i due gruppi. Per le figure dei gruppi anche il codice
LaTeX con la caption, in convergence_figures.txt.
Output in DT/latex/: convergence_<gruppo>_<metrica>.png, convergence_figures.txt (serve graphicx),
convergence_table.txt (servono booktabs, multirow).
Uso: python3 DT/plot_dt_convergence.py
"""
from functools import lru_cache
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =====================================================================
# CONFIGURAZIONE
# =====================================================================
SEEDS = [0, 1, 2]
STATE_TYPE = "Symbolic"
GROUPS = [("tasks1-6", "tasks 1--6", [1, 2, 3, 4, 5, 6]),
          ("tasks7-10", "tasks 7--10", [7, 8, 9, 10])]
PLOT_METRICS = ["sr"]          # tra "sr", "cdr", "reward"
PER_TASK = True
# =====================================================================

EPOCHS = [50, 100, 150, 200]
DT_DIR = Path(__file__).resolve().parent
APPROACHES = [("DT", DT_DIR, "#2a78d6", "o", "-"),
              ("DT+NRM", DT_DIR / "automaton_state" / "grounder", "#eb6834", "s", "--"),
              ("DT+Oracle", DT_DIR / "automaton_state" / "true", "#1baf7a", "^", ":")]
REWARD_TYPES = [("three_value", "Three-value reward"), ("distance", "Distance reward")]
DATASETS = [("medium_expert", "medium-expert"), ("medium", "medium"), ("medium_replay", "medium-replay")]
# (nome nei file, etichetta dell'asse y, colonna del csv, scala)
METRICS = [("sr", "success rate (%)", "success", 100.0),
           ("cdr", "CDR ($\\gamma=0.99$)", "discounted_reward", 1.0),
           ("reward", "reward", "reward_puro", 1.0)]


@lru_cache(maxsize=None)
def load(root, reward_type, task, seed, dataset, epoch):
    sub = "eval" if epoch == 200 else f"epoch{epoch:03d}/eval"
    return pd.read_csv(root / STATE_TYPE / reward_type / f"task{task}" / f"seed{seed}" / sub / f"eval_{dataset}.csv")


def curve(root, reward_type, tasks, dataset, metric):
    """(media, std) tra i seed della media sui task, per ogni epoch"""
    _, _, column, scale = metric
    per_seed = np.array([[np.mean([scale * load(root, reward_type, t, s, dataset, e)[column].mean() for t in tasks])
                          for e in EPOCHS] for s in SEEDS])
    return per_seed.mean(axis=0), per_seed.std(axis=0, ddof=1)


def figure(path, title, tasks, metric):
    key, label, *_ = metric
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(len(REWARD_TYPES), len(DATASETS), figsize=(9, 5.2), sharex=True)
    for i, (rt, rt_title) in enumerate(REWARD_TYPES):
        for j, (ds, ds_title) in enumerate(DATASETS):
            ax = axes[i, j]
            curves = {name: curve(root, rt, tasks, ds, metric) for name, root, *_ in APPROACHES}
            for name, _, color, marker, style in APPROACHES:
                mean, std = curves[name]
                ax.fill_between(EPOCHS, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)
                ax.plot(EPOCHS, mean, color=color, marker=marker, markersize=5, linestyle=style,
                        linewidth=2, label=name)
            if key == "sr":
                low = min(c[0].min() - c[1].max() for c in curves.values())
                ax.set_ylim(max(0, np.floor(low / 5) * 5 - 2), 100.5)
            ax.set_xticks(EPOCHS)
            ax.grid(axis="y", color="#dddddd", linewidth=0.6)
            ax.tick_params(colors="#555555")
            if i == 0:
                ax.set_title(ds_title, fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{rt_title}\n{label}")
            if i == len(REWARD_TYPES) - 1:
                ax.set_xlabel("training epoch")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(APPROACHES), frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(f"{path}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def latex_figure(key, title, metric):
    name = {"sr": "Success rate", "cdr": "Cumulative discounted reward ($\\gamma=0.99$)", "reward": "Reward"}[metric[0]]
    caption = (f"\\textbf{{Learning curves, {STATE_TYPE.lower()} states, {title}.}} {name} of DT, DT+NRM and "
               f"DT+Oracle for the models saved at epochs {', '.join(map(str, EPOCHS))} of training ($K=20$, 500 test "
               f"episodes). Rows: three-value and distance reward; columns: medium-expert, medium and medium-replay "
               f"datasets. Each line is the mean over seeds {', '.join(map(str, SEEDS))} of the average over "
               f"{title}; the shaded band is $\\pm$ one standard deviation over seeds.")
    return "\n".join([r"\begin{figure}[t]", r"\centering",
                      f"\\includegraphics[width=\\linewidth]{{convergence_{key}_{metric[0]}.png}}",
                      f"\\caption{{{caption}}}", f"\\label{{fig:convergence_{key}_{metric[0]}}}", r"\end{figure}"])


def fmt(mean, std):
    return f"${mean:.1f}${{\\scriptsize$\\pm {std:.1f}$}}"


def table():
    n = len(EPOCHS)
    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        f"\\caption{{\\textbf{{Convergence ({STATE_TYPE.lower()} states).}} Success rate (\\%) of the models saved at "
        f"epochs {', '.join(map(str, EPOCHS))}, averaged over the tasks of each group; mean $\\pm$ std over seeds "
        f"{', '.join(map(str, SEEDS))} of the per-seed average.}}",
        r"\label{tab:convergence}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lll " + " ".join("c" * n for _ in DATASETS) + "}",
        r"\toprule",
        "& & & " + " & ".join(f"\\multicolumn{{{n}}}{{c}}{{\\textbf{{{t}}}}}" for _, t in DATASETS) + r" \\",
        "".join(f"\\cmidrule(lr){{{4 + k * n}-{3 + (k + 1) * n}}}" for k in range(len(DATASETS))),
        r"\textbf{Tasks} & \textbf{Reward} & \textbf{Method} & " + " & ".join(str(e) for _ in DATASETS for e in EPOCHS) + r" \\",
        r"\midrule",
    ]
    for g, (_, title, tasks) in enumerate(GROUPS):
        if g:
            lines.append(r"\midrule")
        for r, (rt, rt_title) in enumerate(REWARD_TYPES):
            if r:
                lines.append(f"\\cmidrule(lr){{2-{3 + n * len(DATASETS)}}}")
            for a, (name, root, *_) in enumerate(APPROACHES):
                first = f"\\multirow{{{len(APPROACHES) * len(REWARD_TYPES)}}}{{*}}{{{title.capitalize()}}}" \
                    if r == 0 and a == 0 else ""
                second = f"\\multirow{{{len(APPROACHES)}}}{{*}}{{{rt_title.split()[0]}}}" if a == 0 else ""
                values = []
                for ds, _ in DATASETS:
                    mean, std = curve(root, rt, tasks, ds, METRICS[0])
                    values += [fmt(m, s) for m, s in zip(mean, std)]
                lines.append(f"{first} & {second} & {name} & " + " & ".join(values) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lines)


if __name__ == "__main__":
    out_dir = DT_DIR / "latex"
    out_dir.mkdir(exist_ok=True)
    figures = []
    for metric in [m for m in METRICS if m[0] in PLOT_METRICS]:
        for key, title, tasks in GROUPS:
            figure(out_dir / f"convergence_{key}_{metric[0]}", title, tasks, metric)
            figures.append(latex_figure(key, title, metric))
        if PER_TASK:
            (out_dir / "convergence_per_task").mkdir(exist_ok=True)
            for t in range(1, 11):
                figure(out_dir / "convergence_per_task" / f"task{t}_{metric[0]}", f"task {t}", [t], metric)
    (out_dir / "convergence_figures.txt").write_text(
        "% generato da DT/plot_dt_convergence.py; serve \\usepackage{graphicx}\n" + "\n\n".join(figures) + "\n")
    out = out_dir / "convergence_table.txt"
    out.write_text("% generato da DT/plot_dt_convergence.py; servono \\usepackage{booktabs,multirow,graphicx}\n"
                   + table() + "\n")
    print(f"figure e tabella in {out_dir}")
