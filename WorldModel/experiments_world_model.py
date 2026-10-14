import gc
import traceback

import torch

from world_model import output_paths, run

# =====================================================================
# CONFIGURAZIONE: modifica qui cosa lanciare, poi premi Run
# (il world model non usa il reward: non c'e' distinzione distance / three_value)
# =====================================================================
STATE_TYPES = ["symbolic", "image"]
TASKS = range(1, 11)
DATASETS = ["medium_expert", "medium", "medium_replay"]
SEEDS = [0, 1, 2]

AE_EPOCHS = 20
DYN_EPOCHS = 50
# se False, i run gia' completati (log con i risultati sul test) vengono saltati
# (utile per riprendere una batteria interrotta); se True si rifa' tutto
OVERWRITE = False
# =====================================================================


def run_completed(paths):
    log = paths["log"]
    return (all(p.exists() for p in paths.values())
            and "Next-state position accuracy" in log.read_text(encoding="utf-8"))


summary = []
for state_type in STATE_TYPES:
    for task in TASKS:
        for dataset in DATASETS:
            for seed in SEEDS:
                name = f"{state_type} / task{task} / {dataset} / seed{seed}"
                print(f"\n{'=' * 60}\n {name}\n{'=' * 60}")
                if not OVERWRITE and run_completed(output_paths(state_type, task, dataset, seed)):
                    summary.append((name, "saltato", "(gia' presente)"))
                    continue
                try:
                    # run() fissa il seed all'inizio: ogni run e' riproducibile da solo
                    res = run(state_type=state_type, task=task, dataset=dataset,
                              ae_epochs=AE_EPOCHS, dyn_epochs=DYN_EPOCHS, seed=seed)
                    summary.append((name, f"{res['model_acc']:.2f}%", f"{res['ident_acc']:.2f}%"))
                except Exception:
                    # una run fallita (es. dataset mancante) non blocca le altre
                    traceback.print_exc()
                    summary.append((name, "FAILED", "-"))

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

print(f"\n{'=' * 60}\n SUMMARY - next-state position accuracy (model | identity baseline)\n{'=' * 60}")
for name, acc, base in summary:
    print(f"{name:<45} {acc:>8} | {base:>8}")
