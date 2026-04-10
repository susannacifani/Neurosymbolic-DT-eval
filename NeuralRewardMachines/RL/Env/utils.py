from pathlib import Path
import json
import numpy as np
from Environment import GridWorldEnv
from LTL_tasks import formulas

def setup_folders(state_type, reward_type, task_id):
    folder_state = state_type.capitalize() # "image" -> "Image"
    if "three" in reward_type:
        folder_reward = "three_value"
    else:
        folder_reward = "distance"
    folder_task = f"task{task_id + 1}" 
    base_path = Path("Results") / folder_state / folder_reward / folder_task
    paths = {
        "datasets": base_path / "datasets",
        "models": base_path / "dt_models",
        "eval": base_path / "eval"
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def save_in_folder(state_type, reward_type, task_id):
    folder_state = state_type.capitalize() # "image" -> "Image"
    
    if "three" in reward_type:
        folder_reward = "three_value"
    else:
        folder_reward = "distance"
        
    folder_task = f"task{task_id + 1}" 
    
    # Costruction: Results / Image / three_value / task1
    base_path = Path("Results") / folder_state / folder_reward / folder_task

    return base_path



# def save_dfa(task_id, state_type, reward_type):
#     formula = formulas[task_id]
#     env = GridWorldEnv(formula=formula, state_type=state_type)

#     master_dfa = {
#         "transitions": env.automaton.transitions,
#         "acceptance": [bool(a) for a in env.automaton.acceptance] # Convertiamo in bool per JSON
#     }

#     paths = save_in_folder(state_type, reward_type, task_id)
#     print(paths)

#     json_file = paths / f"master_dfa_task{task_id+1}.json"
#     with open(json_file, "w") as f:
#         json.dump(master_dfa, f)


