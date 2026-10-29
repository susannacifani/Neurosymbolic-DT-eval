"""Confronto a colpo d'occhio tra i diversi approcci del Decision Transformer.

Righe: gli approcci (DT con memoria completa, DT con contesto corto, DT con contesto corto +
stato dell'automa dal grounder (NRM), DT con contesto corto + stato vero dell'automa).
Colonne: Symbolic e Image, ognuno con three_value e distance; per ciascuno SR, CDR e passi.
Ogni numero e' la media su tutti i task e dataset scelti; con piu' seed, media +- deviazione
standard tra i seed (calcolata sulle medie di ogni seed).

Uso: imposta la CONFIGURAZIONE qui sotto e premi Run.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# =====================================================================
# CONFIGURAZIONE: modifica qui cosa stampare, poi premi Run
# =====================================================================
K_SHORT = 1              # contesto corto usato negli esperimenti (cartelle K<k>)
SEEDS = [0]              # es. [0] oppure [0, 1, 2]; None = tutti i seed trovati
SHOW_REPLAY = True       # tabella in piu' solo con medium_replay
SHOW_TASKS = True        # tabella in piu' con il success rate per task
# =====================================================================

DT_ROOT = Path(__file__).resolve().parent
STATE_TYPES = ["Symbolic", "Image"]
REWARD_TYPES = ["three_value", "distance"]
DATASETS = ["medium_expert", "medium", "medium_replay"]
NUM_TASKS = 10
COL_GAP = 2
GROUP_GAP = 4

# (etichetta, cartella dei risultati). Per il riferimento con memoria completa si usa DT/<State>;
# se non c'e' ancora, quello calcolato con la configurazione precedente (DT/old_config/<State>),
# segnato con *
APPROACHES = [
    #("DT, K=50 (memoria completa)", [DT_ROOT, DT_ROOT / "old_config"]),
    (f"DT, K={K_SHORT}", [DT_ROOT / "context_ablation" / f"K{K_SHORT}"]),
    (f"DT, K={K_SHORT} + NRM (grounder)", [DT_ROOT / "automaton_state" / "grounder" / f"K{K_SHORT}"]),
    (f"DT, K={K_SHORT} + automa vero", [DT_ROOT / "automaton_state" / "true" / f"K{K_SHORT}"]),
]
METRICS = [("SR", lambda d: 100.0 * d["success"].mean()),
           ("CDR", lambda d: d["discounted_reward"].mean()),
           ("Passi", lambda d: d["episode_length"].mean())]


def find_seeds():
    seeds = set()
    for _, roots in APPROACHES:
        for root in roots:
            seeds |= {int(p.name[4:]) for p in root.glob("*/*/task*/seed*") if p.name[4:].isdigit()}
    return sorted(seeds)


SEEDS = SEEDS if SEEDS is not None else find_seeds()


def is_old_config(root, state_type, reward_type):
    """True se i modelli in questa cartella sono stati allenati con la configurazione precedente
    (batch diverso da 64, oppure encoder lineare per Symbolic): lo si legge dal modello salvato"""
    model = next(root.glob(f"{state_type}/{reward_type}/task*/seed*/dt_model/*.pth"), None)
    if model is None:
        return False
    config = torch.load(model, map_location="cpu", weights_only=False)["config"]
    encoder_old = state_type == "Symbolic" and config.get("symbolic_encoder", "linear") != "embedding"
    return config.get("batch_size") != 64 or encoder_old


def load(roots, state_type, reward_type, datasets):
    """per ogni seed, la media della metrica su tutti i task e dataset; si usa la prima cartella in
    cui ci sono risultati; old = modelli allenati con la configurazione precedente"""
    for root in roots:
        per_seed = {}
        for seed in SEEDS:
            frames = [pd.read_csv(f) for task in range(1, NUM_TASKS + 1) for ds in datasets
                      if (f := root / state_type / reward_type / f"task{task}" / f"seed{seed}" / "eval" / f"eval_{ds}.csv").exists()]
            if frames:
                per_seed[seed] = {name: np.mean([fn(d) for d in frames]) for name, fn in METRICS} | {"n": len(frames)}
        if per_seed:
            return per_seed, is_old_config(root, state_type, reward_type)
    return {}, False


def fmt(values):
    if not values:
        return "N/A"
    if len(values) == 1:
        return f"{values[0]:.1f}"
    return f"{np.mean(values):.1f} ± {np.std(values, ddof=1):.1f}"


def table(datasets, title):
    columns = [(st, rt, m) for st in STATE_TYPES for rt in REWARD_TYPES for m, _ in METRICS]
    rows, notes = [], set()
    for label, roots in APPROACHES:
        row = {}
        for st in STATE_TYPES:
            for rt in REWARD_TYPES:
                per_seed, old = load(roots, st, rt, datasets)
                runs = sum(v["n"] for v in per_seed.values())
                expected = len(SEEDS) * NUM_TASKS * len(datasets)
                for m, _ in METRICS:
                    row[(st, rt, m)] = fmt([v[m] for v in per_seed.values()])
                mark = ("*" if old else "") + ("!" if per_seed and runs < expected else "")
                if mark:
                    row[(st, rt, METRICS[-1][0])] += mark
                    notes |= set(mark)
        rows.append((label, row))

    first_width = max(len("Approccio"), *(len(label) for label, _ in rows))
    widths = {c: max(len(c[2]), *(len(r[c]) for _, r in rows)) for c in columns}
    positions, pos = {}, first_width
    for i, c in enumerate(columns):
        pos += GROUP_GAP if i % len(METRICS) == 0 else COL_GAP
        positions[c] = pos
        pos += widths[c]
    total = pos

    def centered(line, text, start, end):
        center = (start + end) // 2
        return line.ljust(max(center - len(text) // 2, len(line) + 1 if line.strip() else 0)) + text

    state_line, reward_line = "", ""
    for st in STATE_TYPES:
        cols = [c for c in columns if c[0] == st]
        state_line = centered(state_line, st, positions[cols[0]], positions[cols[-1]] + widths[cols[-1]])
        for rt in REWARD_TYPES:
            cols_rt = [c for c in cols if c[1] == rt]
            reward_line = centered(reward_line, rt, positions[cols_rt[0]], positions[cols_rt[-1]] + widths[cols_rt[-1]])
    metric_line = "Approccio".ljust(first_width)
    for c in columns:
        metric_line = metric_line.ljust(positions[c]) + c[2].rjust(widths[c])

    print(f"\n=== {title} ===")
    print(state_line)
    print(reward_line)
    print(metric_line)
    print("-" * total)
    for label, row in rows:
        line = label.ljust(first_width)
        for c in columns:
            line = line.ljust(positions[c]) + row[c].rjust(widths[c])
        print(line)
    if "*" in notes:
        print("* = DT con memoria completa ancora con la configurazione precedente (encoder lineare per Symbolic, batch 32 per Image)")
    if "!" in notes:
        print("! = non tutti i run sono presenti per i seed scelti (la media usa quelli presenti)")


def task_table():
    """success rate per task: righe = task, colonne = approccio per stato (media su reward, dataset e seed)"""
    short = ["K=50", f"K={K_SHORT}", "+NRM", "+vero"]
    print("\n=== Success rate per task (media su reward, dataset e seed) ===")
    header = "Task    " + "    ".join("  ".join(f"{s:>6}" for s in short) for _ in STATE_TYPES)
    groups = "        " + "    ".join(st.center(len("  ".join(f"{s:>6}" for s in short))) for st in STATE_TYPES)
    print(groups)
    print(header)
    print("-" * len(header))
    for task in range(1, NUM_TASKS + 1):
        cells = []
        for st in STATE_TYPES:
            vals = []
            for _, roots in APPROACHES:
                sr = None
                for root in roots:
                    fs = [f for rt in REWARD_TYPES for ds in DATASETS for seed in SEEDS
                          if (f := root / st / rt / f"task{task}" / f"seed{seed}" / "eval" / f"eval_{ds}.csv").exists()]
                    if fs:
                        sr = np.mean([100.0 * pd.read_csv(f)["success"].mean() for f in fs])
                        break
                vals.append(f"{sr:6.1f}" if sr is not None else "   N/A")
            cells.append("  ".join(vals))
        print(f"task{task:<4}" + "    ".join(cells))


def main():
    print(f"DECISION TRANSFORMER - confronto tra approcci - seed {', '.join(map(str, SEEDS))}")
    print("SR = success rate (0-100) | CDR = cumulative discounted reward (gamma 0.99) | Passi = lunghezza media dell'episodio")
    print("NRM (grounder) = stato dell'automa calcolato con i simboli del grounder allenato solo dai reward")
    table(DATASETS, "Tutti i dataset (media su 10 task x 3 dataset)")
    if SHOW_REPLAY:
        table(["medium_replay"], "Solo medium_replay (media su 10 task)")
    if SHOW_TASKS:
        task_table()


if __name__ == "__main__":
    main()
