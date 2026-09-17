import os
os.environ["SDL_VIDEODRIVER"] = "dummy"



#da experiments.py
from LTL_tasks import formulas
import absl.flags
import absl.app
#import os
from RL.NRM.utils import set_seed
from RL.Env.Environment import GridWorldEnv
from RL.A2C_nrm import recurrent_A2C
from plot import plot
import gc
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






state_type = "image"
reward_type = "three_value"

dataset_list = ["medium_expert", "medium", "medium_replay"]

for TASK in [7, 8, 9, 10]:
    print(f"\n{'='*50}")
    print(f" AVVIO ESPERIMENTI PER IL TASK {TASK}")
    print(f"{'='*50}")
    print(f"Stato: {state_type}, reward: {reward_type}")
    
    task_id = TASK - 1
    formula = formulas[task_id]
    print(len(formulas))
    print(formula)
    
    # Crea il path specifico per il task
    if state_type == "image":
        feature_extraction = True
        if reward_type == "three_value":
            path = f"Results/Image/three_value/task{TASK}"
        elif reward_type == "distance":
            path = f"Results/Image/distance/task{TASK}"
    elif state_type == "symbolic":
        feature_extraction = False
        if reward_type == "three_value":
            path = f"Results/Symbolic/three_value/task{TASK}"
        elif reward_type == "distance":
            path = f"Results/Symbolic/distance/task{TASK}"
    if not os.path.exists(path):
        os.makedirs(path)
        
    set_seed(0)

    # 1. Inizializza l'ambiente UNA SOLA VOLTA per ogni task
    env = GridWorldEnv(formula, "rgb_array", state_type=state_type, use_dfa_state=use_dfa_state, train=False)

    # Ciclo interno: itera sui 3 dataset
    for dataset_el in dataset_list:
        print(f"\n---> Esecuzione Task {TASK} con dataset: {dataset_el}")
        
        # 2. Avvia l'addestramento/valutazione
        recurrent_A2C(task_id, dataset_el, state_type, reward_type, env, path, experiment, method, feature_extraction)

    # 3. PULIZIA DELLA MEMORIA A FINE TASK
    # Fondamentale per svuotare la VRAM della GPU ed evitare crash "Out Of Memory"
    # man mano che carichi nuovi ambienti e modelli.
    del env
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
print("\nTUTTI GLI ESPERIMENTI SONO STATI COMPLETATI CON SUCCESSO!")

