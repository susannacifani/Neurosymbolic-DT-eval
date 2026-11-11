"""Tabelle LaTeX delle statistiche dei dataset 7x7 (Dataset/<reward>/task<N>/dataset_<dataset>.h5).

Due tabelle, ognuna con due blocchi di righe: task 1-6 (solo visite) e task 7-10 (visite + G !c,
gli unici con fallimento possibile):
- datasets: una riga per dataset. Length: lunghezza media;
  SR / Fail / Trunc.: esiti; R / CDR per three-value e distance. Media sui task del blocco.
- trajectory kinds: una riga per tipo di traiettoria (optimal / noisy / random), su tutte le
  traiettorie di quel tipo nei 3 dataset e nei task del blocco. Lunghezza ed esiti.
Esiti ricavati dal return three-value (100 / -100 / 0); le traiettorie sono le stesse per distance e
three-value e i reward sono coerenti con l'automa (entrambi verificati da check_datasets.py).

Uso: imposta la CONFIGURAZIONE e premi Run. Output in OUTPUT (e stampato a terminale).
"""
from pathlib import Path

import h5py
import numpy as np

DATASET_ROOT = Path(__file__).resolve().parent

# =====================================================================
# CONFIGURAZIONE
# =====================================================================
GROUPS = [("1--6", range(1, 7)), ("7--10", range(7, 11))]
DATASETS = ["medium_expert", "medium", "medium_replay"]
GAMMA = 0.99               # come nella valutazione del DT
OUTPUT = DATASET_ROOT / "latex" / "datasets.txt"
# =====================================================================

REWARD_TYPES = ["three_value", "distance"]
KINDS = ["optimal", "noisy", "random"]


def load(task, dataset):
    """una riga per traiettoria: tipo, lunghezza, esito, (R, CDR) per ogni reward"""
    rows = {}
    for rt in REWARD_TYPES:
        with h5py.File(DATASET_ROOT / rt / f"task{task}" / f"dataset_{dataset}.h5", "r") as f:
            for k in f.keys():
                g = f[k]
                r = g["rewards"][:]
                row = rows.setdefault(k, {"kind": str(g.attrs["kind"]), "length": int(g.attrs["length"])})
                row[rt] = (float(r.sum()), float(np.sum(r * GAMMA ** np.arange(len(r)))))
                if rt == "three_value":
                    # i reward sono differenze tra stati dell'automa: il return three-value e' il valore
                    # dello stato finale (100 vittoria, -100 fallimento, 0 nessuno dei due = troncata)
                    row["outcome"] = {100: "succ", -100: "fail", 0: "trunc"}[round(row[rt][0])]
    return list(rows.values())


def outcomes(rows):
    return [100.0 * np.mean([r["outcome"] == o for r in rows]) for o in ["succ", "fail", "trunc"]]


def dataset_row(per_task):
    """per_task: lista (un elemento per task) delle traiettorie di un dataset; media sui task"""
    vals = []
    for rows in per_task:
        v = [sum(r["length"] for r in rows), np.mean([r["length"] for r in rows]), *outcomes(rows)]
        for rt in REWARD_TYPES:
            v += [np.mean([r[rt][0] for r in rows]), np.mean([r[rt][1] for r in rows])]
        vals.append(v)
    return np.mean(vals, axis=0)


def main():
    data = {(t, ds): load(t, ds) for _, tasks in GROUPS for t in tasks for ds in DATASETS}

    # ---------------- tabella 1: dataset ----------------
    body = []
    for di, ds in enumerate(DATASETS):
        if di:
            body.append(r"\midrule")
        for gi, (gname, tasks) in enumerate(GROUPS):
            v = dataset_row([data[(t, ds)] for t in tasks])
            first = rf"\multirow{{{len(GROUPS)}}}{{*}}{{{ds.replace('_', '-')}}}" if gi == 0 else ""
            body.append(f"{first} & {gname} & {v[1]:.1f} & {v[2]:.1f} & {v[3]:.1f} & "
                        f"{v[4]:.1f} & {v[5]:.1f} & {v[6]:.1f} & {v[7]:.1f} & {v[8]:.1f} \\\\")
    t1 = r"""\begin{table*}[t]
\centering
\caption{Statistics of the datasets (each contains 5000 trajectories), averaged over tasks 1--6 (visit
only) and over tasks 7--10 (visit with global avoidance, the only tasks where a trajectory can fail).
Length: average trajectory length. SR, Fail, Trunc.: percentage
of trajectories ending with a success, a failure, or truncation after 100 steps. R: average return
(undiscounted sum of rewards); CDR: average cumulative discounted reward ($\gamma=0.99$).}
\label{tab:datasets}
\small
\resizebox{\textwidth}{!}{%
\begin{tabular}{llcccccccc}
\toprule
 & & & & & & \multicolumn{2}{c}{three-value} & \multicolumn{2}{c}{distance} \\
\cmidrule(lr){7-8}\cmidrule(lr){9-10}
Dataset & Tasks & Length & SR (\%) & Fail (\%) & Trunc. (\%) & R & CDR & R & CDR \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}%
}
\end{table*}"""

    # ---------------- tabella 2: tipi di traiettoria ----------------
    body = []
    for ki, kind in enumerate(KINDS):
        if ki:
            body.append(r"\midrule")
        for gi, (gname, tasks) in enumerate(GROUPS):
            rows = [r for t in tasks for ds in DATASETS for r in data[(t, ds)] if r["kind"] == kind]
            s, f, tr = outcomes(rows)
            first = rf"\multirow{{{len(GROUPS)}}}{{*}}{{{kind}}}" if gi == 0 else ""
            body.append(f"{first} & {gname} & {np.mean([r['length'] for r in rows]):.1f} & {s:.1f} & {f:.1f} & {tr:.1f} \\\\")
    t2 = r"""\begin{table}[t]
\centering
\caption{Statistics of each kind of trajectory, computed over all datasets, separately for tasks 1--6
and tasks 7--10. Length: average trajectory length. SR, Fail, Trunc.: percentage of trajectories
ending with a success, a failure, or truncation after 100 steps.}
\label{tab:trajectory-kinds}
\small
\begin{tabular}{llcccc}
\toprule
Trajectory & Tasks & Length & SR (\%) & Fail (\%) & Trunc. (\%) \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\end{table}"""

    text = "% richiede \\usepackage{booktabs, multirow, graphicx}\n\n" + t1 + "\n\n\n" + t2 + "\n"
    print(text)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(text)
    print(f"Salvato in {OUTPUT}")


if __name__ == "__main__":
    main()
