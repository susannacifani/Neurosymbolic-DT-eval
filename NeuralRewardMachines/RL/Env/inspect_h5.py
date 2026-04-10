# import h5py
# import numpy as np
# import os
# from utils import setup_folders

# NUM_TASK = 3


# task_id = NUM_TASK - 1
# #Image
# state_type = "image"
# #THREE
# reward_type = "three_value"
# file = 'dataset_medium_expert.h5'
# paths = setup_folders(state_type, reward_type, task_id)
# H5_PATH = paths["datasets"] / file



# def ispeziona_dataset_h5(file_path):
#     if not os.path.exists(file_path):
#         print("Il file non esiste!")
#         return

#     with h5py.File(file_path, 'r') as f:
#         num_traiettorie = len(f.keys())
#         print(f"--- Dataset HDF5 caricato: {num_traiettorie} traiettorie ---")

#         idx = '0' 
#         if idx not in f:
#             idx = list(f.keys())[0] 
            
#         grp = f[idx]
        
#         print(f"\nIspezione Gruppo n.{idx}:")
#         print(f" - Lunghezza (step): {grp.attrs['length']}")
#         print(f" - Start: {grp.attrs['start_pos']}")
        
#         #print(f" - Stati (Osservazioni): {grp['observations'][:].tolist()}")
#         print(f" - Azioni: {grp['actions'][:].tolist()}")
#         print(f" - Rewards: {grp['rewards'][:].tolist()}")
#         print(f" - Returns-to-Go: {grp['returns_to_go'][:].tolist()}")
#         print(f" - Reward Totale: {np.sum(grp['rewards'][:])}")
        
#         print(f" - DFA: {grp['dfa_states'][:].tolist()}")
#         print(f" - Image obs shape: {grp['observations'].shape}")
        
#         print(f" - Prima azione: {grp['actions'][0]}")

# ispeziona_dataset_h5(H5_PATH)



import h5py
import numpy as np
import os
from utils import setup_folders


NUM_TASK = 7

task_id = NUM_TASK - 1
state_type = "image"
reward_type = "three_value"  
file = 'dataset_medium_expert.h5'
paths = setup_folders(state_type, reward_type, task_id)
H5_PATH = paths["datasets"] / file
print(H5_PATH)

def cerca_fallimenti_h5(file_path):
    if not os.path.exists(file_path):
        print(f"Il file non esiste in: {file_path}")
        return

    with h5py.File(file_path, 'r') as f:
        num_traiettorie = len(f.keys())
        print(f"--- Dataset caricato: {num_traiettorie} traiettorie ---")
        
        traiettorie_con_fallimento = []

        for idx in f.keys():
            grp = f[idx]
            rewards = grp['rewards'][:]
            
            if any(r <= -50 for r in rewards):
                traiettorie_con_fallimento.append(idx)
        
        print(f"Trovate {len(traiettorie_con_fallimento)} traiettorie terminate in tragedia (-100)!")
        
        if traiettorie_con_fallimento:
            idx_esempio = traiettorie_con_fallimento[0]
            grp = f[idx_esempio]
            print(f"\n--- Esempio Tragedia (Gruppo n.{idx_esempio}) ---")
            print(f" - Start: {grp.attrs['start_pos']}")
            #print(f" - Stati (Osservazioni): {grp['observations'][:].tolist()}")
            print(f" - Azioni: {grp['actions'][:].tolist()}")
            print(f" - Rewards arrotondati: {[round(r, 1) for r in grp['rewards'][:].tolist()]}")
            print(f" - RTG arrotondati: {[round(r, 1) for r in grp['returns_to_go'][:].tolist()]}")
            print(f" - Finito al passo: {grp.attrs['length']}")

cerca_fallimenti_h5(H5_PATH)