import os
import sys
from pathlib import Path
os.environ["SDL_VIDEODRIVER"] = "dummy"

# cartella Grounding/ (dove sta questo script) e motore comune in NRM/NeuralRewardMachines
GROUNDING_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(GROUNDING_ROOT.parent / "NeuralRewardMachines"))

#da experiments.py
from LTL_tasks import formulas
from RL.NRM.utils import set_seed
from RL.Env.Environment import GridWorldEnv
from RL.A2C_nrm import recurrent_A2C
import gc
import re
import traceback
import torch





#da experiments.py viene chiamata A2C.py
#ho isolato solo il singolo pezzo che mi serve, per evitare di eseguire tutti i test completi




experiment = 0
method = 'nrm'





# TASK = 5
# task_id = TASK - 1
# formula = formulas[task_id]
# state_type = "image"
# use_dfa_state = False
# path = f"Results/Image/three_value/task{TASK}"

# env = GridWorldEnv(formula, "rgb_array", state_type=state_type, use_dfa_state=use_dfa_state, train=False)


# if not os.path.exists(path):
#     os.makedirs(path)
# set_seed(0)

# dataset_list = ["medium_expert", "medium", "medium_replay"]
# dataset_el = dataset_list[2]


# recurrent_A2C(task_id, dataset_el, env, path, experiment, method, feature_extraction)




# state_type = "image"
# state_type = "symbolic"
# reward_type = "three_value"
# reward_type = "distance"
use_dfa_state = False
dataset_list = ["medium_expert", "medium", "medium_replay"]

# for TASK in range(1, 11): 
#     print(f"\n{'='*50}")
#     print(f" AVVIO ESPERIMENTI PER IL TASK {TASK}")
#     print(f"{'='*50}")
    
#     task_id = TASK - 1
#     formula = formulas[task_id]
#     print(len(formulas))
#     print(formula)
    
#     # Crea il path specifico per il task
#     if state_type == "image":
#         feature_extraction = True
#         if reward_type == "three_value":
#             path = f"Results/Image/three_value/task{TASK}"
#         elif reward_type == "distance":
#             path = f"Results/Image/distance/task{TASK}"
#     elif state_type == "symbolic":
#         feature_extraction = False
#         if reward_type == "three_value":
#             path = f"Results/Symbolic/three_value/task{TASK}"
#         elif reward_type == "distance":
#             path = f"Results/Symbolic/distance/task{TASK}"
#     if not os.path.exists(path):
#         os.makedirs(path)
        
#     set_seed(0)

#     # 1. Inizializza l'ambiente UNA SOLA VOLTA per ogni task
#     env = GridWorldEnv(formula, "rgb_array", state_type=state_type, use_dfa_state=use_dfa_state, train=False)

#     # Ciclo interno: itera sui 3 dataset
#     for dataset_el in dataset_list:
#         print(f"\n---> Esecuzione Task {TASK} con dataset: {dataset_el}")
        
#         # 2. Avvia l'addestramento/valutazione
#         recurrent_A2C(task_id, dataset_el, state_type, reward_type, env, path, experiment, method, feature_extraction)

#     # 3. PULIZIA DELLA MEMORIA A FINE TASK
#     # Fondamentale per svuotare la VRAM della GPU ed evitare crash "Out Of Memory"
#     # man mano che carichi nuovi ambienti e modelli.
#     del env
#     gc.collect()
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
        
# print("\nTUTTI GLI ESPERIMENTI SONO STATI COMPLETATI CON SUCCESSO!")






# =====================================================================
# CONFIGURAZIONE: modifica qui cosa lanciare, poi premi Run
# =====================================================================
STATE_TYPES = ["symbolic", "image"]
REWARD_TYPES = ["three_value", "distance"]
TASKS = range(1, 11)
DATASETS = ["medium_expert", "medium", "medium_replay"]
SEEDS = [0, 1, 2]
# se False, i run gia' completati (log con il risultato finale sul test) vengono saltati
# (utile per riprendere una batteria interrotta); se True si rifa' tutto
OVERWRITE = False
# =====================================================================


def run_completed(path, dataset_el):
    log = os.path.join(path, "log", f"log_{dataset_el}.txt")
    model = os.path.join(path, "grounder", f"grounder_{dataset_el}.pth")
    return os.path.exists(model) and os.path.exists(log) and "RECALL PER CLASSE SUL TEST SET" in open(log).read()


summary = []
for state_type in STATE_TYPES:
    for reward_type in REWARD_TYPES:
        for TASK in TASKS:
            task_id = TASK - 1
            formula = formulas[task_id]
            feature_extraction = state_type == "image"

            env = GridWorldEnv(formula, "rgb_array", state_type=state_type, use_dfa_state=use_dfa_state,
                               train=False, reward_type=reward_type)

            for dataset_el, seed in [(d, s) for d in DATASETS for s in SEEDS]:
                # risultati in Grounding/<State>/<reward>/task<N>/seed<S>/{log,grounder}
                path = str(GROUNDING_ROOT / state_type.capitalize() / reward_type / f"task{TASK}" / f"seed{seed}")
                name = f"{state_type} / {reward_type} / task{TASK} / {dataset_el} / seed{seed}"
                print(f"\n{'=' * 70}\n {name}\n{'=' * 70}")
                if not OVERWRITE and run_completed(path, dataset_el):
                    summary.append((name, "saltato (gia' presente)"))
                    continue
                try:
                    # seed fissato prima di ogni run: ogni risultato e' riproducibile da solo,
                    # indipendentemente da quali altri run sono stati lanciati prima
                    set_seed(seed)
                    recurrent_A2C(task_id, dataset_el, state_type, reward_type, env, path, experiment, method, feature_extraction)
                    log = open(os.path.join(path, "log", f"log_{dataset_el}.txt")).read()
                    bal = re.search(r"BALANCED ACCURACY FINALE GLOBALE SUL TEST SET: ([\d.]+)", log).group(1)
                    summary.append((name, f"balanced {bal}%"))
                except Exception:
                    traceback.print_exc()
                    summary.append((name, "FAILED"))
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            del env
            gc.collect()

print(f"\n{'=' * 70}\n SUMMARY - balanced accuracy sul test\n{'=' * 70}")
for name, res in summary:
    print(f"{name:<63} {res}")
