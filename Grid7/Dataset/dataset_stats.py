"""Statistiche dei dataset 7x7 (Dataset/<reward>/task<N>/dataset_<dataset>.h5).

Le traiettorie sono le stesse per distance e three_value (cambiano solo i reward), quindi tutto cio'
che non dipende dal reward si calcola una volta per task e dataset. Per ogni task e dataset:
- composizione: quante traiettorie optimal / noisy / random
- esiti per tipo: successo, fallimento (con l'oggetto che l'ha causato), limite di passi (100)
- lunghezze: media, mediana, minimo, massimo, 95 percentile; passi in piu' rispetto al percorso
  minimo (calcolato con il planner dalla stessa partenza); quante traiettorie sono esattamente minime
- progresso: quanta parte del compito e' stata completata da chi non ha vinto (sotto-obiettivi, come
  li misura il reward distance)
- oggetti: quante volte l'agente entra su un oggetto, e su quanti tipi diversi
- quanti passi (azioni) ricevono il reward di vittoria e quanti quello di fallimento
- per ogni tipo di reward: reward totale, CDR (reward scontato, gamma 0.99 come nella valutazione del
  DT), success rate, e quanti passi hanno reward positivo, negativo o nullo
- quante traiettorie hanno ogni lunghezza (istogramma)
In fondo una tabella riassuntiva per tutti i task.

Uso: imposta la CONFIGURAZIONE qui sotto e premi Run. Il report viene anche salvato in un file di
testo (SAVE_PATH).
"""
import contextlib
import io
import sys
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

DATASET_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(DATASET_ROOT.parent / "NeuralRewardMachines"))
sys.path.insert(0, str(DATASET_ROOT))

from LTL_tasks import formulas
from RL.Env.Environment import GridWorldEnv, EMPTY
from planner import shortest_path

# =====================================================================
# CONFIGURAZIONE: modifica qui cosa analizzare, poi premi Run
# =====================================================================
TASKS = range(1, 11)
DATASETS = ["medium_expert", "medium", "medium_replay"]
REWARD_TYPES = ["three_value", "distance"]
COMPUTE_OPTIMAL = False     # passi in piu' rispetto al minimo (planner, ~20 s per dataset)
LENGTH_HISTOGRAM = True    # quante traiettorie per ogni lunghezza
GAMMA = 0.99               # come nella valutazione del DT
SAVE_PATH = DATASET_ROOT / "dataset_stats.txt"   # None = non salvare
# =====================================================================

KINDS = ["optimal", "noisy", "random"]
OBJECT_NAMES = {0: "pickaxe (P)", 1: "lava (L)", 2: "door (D)", 3: "gem (G)"}


def make_env(task, reward_type):
    # la libreria dell'automa stampa le sue strutture interne: si nascondono
    with contextlib.redirect_stdout(io.StringIO()):
        return GridWorldEnv(formulas[task - 1], state_type="symbolic", reward_type=reward_type)


def analyze_trajectories(task, dataset):
    """quantita' che non dipendono dal tipo di reward (lette dal file three_value: le traiettorie sono le
    stesse per distance, verificato da check_datasets.py)"""
    env, env_dist = make_env(task, "three_value"), make_env(task, "distance")
    aut, aut_d = env.automaton, env_dist.automaton
    rows = []
    with h5py.File(DATASET_ROOT / "three_value" / f"task{task}" / f"dataset_{dataset}.h5", "r") as f:
        for k in sorted(f.keys(), key=int):
            g = f[k]
            grid, pos = g["grid"][:].astype(np.int64), g["positions"][:].astype(np.int64)
            T = int(g.attrs["length"])
            q, qd, best_progress, touches, types, fail_obj = 0, 0, 0.0, 0, set(), None
            for t in range(1, T + 1):
                sym = int(grid[pos[t][0], pos[t][1]])
                if sym != EMPTY:
                    touches += 1
                    types.add(sym)
                q, qd = aut.transitions[q][sym], aut_d.transitions[qd][sym]
                if aut_d.rewards[qd] != -100:
                    best_progress = max(best_progress, aut_d.rewards[qd] / 100.0)
                if aut.rewards[q] == -100:
                    fail_obj = sym
            outcome = "successo" if aut.acceptance[q] else "fallimento" if aut.rewards[q] == -100 else "limite di passi"
            opt = None
            if COMPUTE_OPTIMAL:
                env.reset(grid=grid, start_pos=pos[0])
                path = shortest_path(env, np.random.default_rng(0))
                opt = len(path) if path is not None else None
            rows.append({"key": k, "kind": str(g.attrs["kind"]), "length": T, "outcome": outcome, "fail_obj": fail_obj,
                         "progress": best_progress, "touches": touches, "types": len(types), "optimal": opt,
                         "start": tuple(pos[0]), "layout": grid.tobytes()})
    return rows


def reward_stats(task, dataset, reward_type):
    """per traiettoria: (reward totale, CDR); per il dataset: quanti passi hanno reward positivo,
    negativo o nullo"""
    per_key, step_signs = {}, Counter()
    with h5py.File(DATASET_ROOT / reward_type / f"task{task}" / f"dataset_{dataset}.h5", "r") as f:
        for k in f.keys():
            r = f[k]["rewards"][:]
            per_key[k] = (float(r.sum()), float(np.sum(r * GAMMA ** np.arange(len(r)))))
            step_signs["positivo"] += int((r > 0).sum())
            step_signs["negativo"] += int((r < 0).sum())
            step_signs["nullo"] += int((r == 0).sum())
    return per_key, step_signs


def pct(n, tot):
    return f"{100.0 * n / tot:5.1f}%" if tot else "    -"


def ms(values):
    return f"{np.mean(values):7.2f} ± {np.std(values):6.2f}" if len(values) else "      -"


def report_dataset(task, dataset, rows, rewards, out):
    groups = [(kind, [r for r in rows if r["kind"] == kind]) for kind in KINDS]
    groups = [(kind, rs) for kind, rs in groups if rs] + [("totale", rows)]
    out(f"\n{'=' * 100}\n task{task} / {dataset}   ({formulas[task - 1][2]})\n{'=' * 100}")
    out(f"Traiettorie: {len(rows)} | " + ", ".join(f"{kind}: {len(rs)}" for kind, rs in groups[:-1])
        + f" | disposizioni diverse: {len({r['layout'] for r in rows})} | caselle di partenza diverse: "
        f"{len({r['start'] for r in rows})}")

    out("\nESITI")
    out(f"  {'tipo':<8} {'n':>5}   {'successo':>15} {'fallimento':>15} {'limite passi':>15}")
    for kind, rs in groups:
        c = Counter(r["outcome"] for r in rs)
        out(f"  {kind:<8} {len(rs):>5}   " + " ".join(f"{c[o]:>6} ({pct(c[o], len(rs))})" for o in
                                                     ["successo", "fallimento", "limite di passi"]))
    # ogni traiettoria vincente (o fallita) riceve il reward di vittoria (o di fallimento) in un solo
    # passo, l'ultimo: percentuale sul totale dei passi (azioni) del dataset
    c, steps = Counter(r["outcome"] for r in rows), sum(r["length"] for r in rows)
    out(f"  passi (azioni) totali: {steps} | passi in cui arriva il reward di vittoria: {pct(c['successo'], steps).strip()}"
        f" | di fallimento: {pct(c['fallimento'], steps).strip()}")
    fails = [r for r in rows if r["outcome"] == "fallimento"]
    if fails:
        out("  fallimenti per oggetto toccato: " + ", ".join(
            f"{OBJECT_NAMES[o]}: {n} ({kind_split(fails, o)})" for o, n in Counter(r["fail_obj"] for r in fails).most_common()))

    out("\nLUNGHEZZE (passi)")
    out(f"  {'tipo':<8} {'media':>6} {'mediana':>8} {'min':>4} {'max':>4} {'95%':>5}" +
        ("   passi in piu' del minimo (solo successi)   esattamente minime" if COMPUTE_OPTIMAL else ""))
    for kind, rs in groups:
        L = np.array([r["length"] for r in rs])
        line = f"  {kind:<8} {L.mean():6.1f} {np.median(L):8.0f} {L.min():4d} {L.max():4d} {np.percentile(L, 95):5.0f}"
        if COMPUTE_OPTIMAL:
            succ = [r for r in rs if r["outcome"] == "successo" and r["optimal"] is not None]
            extra = [r["length"] - r["optimal"] for r in succ]
            exact = sum(e == 0 for e in extra)
            line += (f"   media {np.mean(extra):5.2f}, max {max(extra):3d}" if extra else "   -") + \
                    f"                     {exact:>5} ({pct(exact, len(succ))} dei successi)"
        out(line)
    if COMPUTE_OPTIMAL:
        opt = np.array([r["optimal"] for r in rows if r["optimal"] is not None])
        out(f"  percorso minimo dalla partenza: media {opt.mean():.1f}, mediana {np.median(opt):.0f}, max {opt.max()}")

    out("\nPROGRESSO E OGGETTI")
    for kind, rs in groups:
        not_won = [r["progress"] for r in rs if r["outcome"] != "successo"]
        out(f"  {kind:<8} entrate su oggetti per traiettoria: {np.mean([r['touches'] for r in rs]):5.1f} | tipi di oggetto "
            f"toccati: {np.mean([r['types'] for r in rs]):4.2f} | compito completato da chi non vince: "
            + (f"{100 * np.mean(not_won):5.1f}% in media ({len(not_won)} traiettorie)" if not_won else "-"))

    for reward_type, (per_key, step_signs) in rewards.items():
        out(f"\nREWARD {reward_type} (CDR con gamma {GAMMA})")
        out(f"  {'tipo':<8} {'SR':>7}   {'reward totale':>17}   {'CDR':>17}")
        for kind, rs in groups:
            R = [per_key[r["key"]][0] for r in rs]
            C = [per_key[r["key"]][1] for r in rs]
            sr = sum(r["outcome"] == "successo" for r in rs)
            out(f"  {kind:<8} {pct(sr, len(rs)):>7}   {ms(R)}   {ms(C)}")
        tot = sum(step_signs.values())
        out("  passi (azioni) con reward: " + ", ".join(
            f"{s} {pct(step_signs[s], tot).strip()}" for s in ["positivo", "negativo", "nullo"]))

    if LENGTH_HISTOGRAM:
        out("\nTRAIETTORIE PER LUNGHEZZA (lunghezza: quante)")
        for kind, rs in groups[:-1]:
            c = sorted(Counter(r["length"] for r in rs).items())
            out(f"  {kind}:")
            for i in range(0, len(c), 12):
                out("    " + "  ".join(f"{L:>3}:{n:<4}" for L, n in c[i:i + 12]))


def kind_split(fails, obj):
    c = Counter(r["kind"] for r in fails if r["fail_obj"] == obj)
    return ", ".join(f"{k} {n}" for k, n in c.items())


def main():
    lines = []

    def out(msg=""):
        print(msg, flush=True)
        lines.append(msg)

    summary = []
    for task in TASKS:
        for dataset in DATASETS:
            rows = analyze_trajectories(task, dataset)
            rewards = {rt: reward_stats(task, dataset, rt) for rt in REWARD_TYPES}
            report_dataset(task, dataset, rows, rewards, out)
            c = Counter(r["outcome"] for r in rows)
            n = len(rows)
            row = {"task": task, "dataset": dataset, "succ": 100 * c["successo"] / n, "fail": 100 * c["fallimento"] / n,
                   "timeout": 100 * c["limite di passi"] / n, "len": np.mean([r["length"] for r in rows])}
            for rt, (per_key, _) in rewards.items():
                row[rt] = (np.mean([v[0] for v in per_key.values()]), np.mean([v[1] for v in per_key.values()]))
            summary.append(row)

    out(f"\n{'=' * 100}\n RIEPILOGO (tutte le traiettorie di ogni dataset)\n{'=' * 100}")
    head = f"{'task':<7}{'dataset':<15}{'SR':>7}{'fall.':>7}{'limite':>8}{'lungh.':>8}"
    for rt in REWARD_TYPES:
        head += f"   {rt + ' reward / CDR':>26}"
    out(head)
    for dataset in DATASETS:
        for row in [r for r in summary if r["dataset"] == dataset]:
            line = (f"task{row['task']:<3}{dataset:<15}{row['succ']:6.1f}%{row['fail']:6.1f}%{row['timeout']:7.1f}%"
                    f"{row['len']:8.1f}")
            for rt in REWARD_TYPES:
                line += f"   {row[rt][0]:12.2f} / {row[rt][1]:9.2f}"
            out(line)
        out("")

    if SAVE_PATH:
        Path(SAVE_PATH).write_text("\n".join(lines) + "\n")
        print(f"Report salvato in {SAVE_PATH}")


if __name__ == "__main__":
    main()
