import numpy as np
import h5py
import os
import random 
from Environment import GridWorldEnv
from LTL_tasks import formulas
from DijkstraTraj import task1, task2, task3, task4, task5, task6, task7, task8, task9, task10
from utils import setup_folders, save_in_folder

EPSILON = 0.2  # 20% probability of noise injection
MAX_STEPS = 50 
NUM_PER_POS = 84 

DIJKSTRA_FUNCS = {
                    0: task1, 1: task2, 2: task3, 3: task4, 4: task5,
                    5: task6, 6: task7, 7: task8, 8: task9, 9: task10
                 }




def medium_expert(task_id, file, state_type, reward_type):
    H5_PATH = file

    formula = formulas[task_id]


    env = GridWorldEnv(formula=formula, render_mode="rgb_array", state_type=state_type, train=False)

    with h5py.File(H5_PATH, 'a') as f:
        if len(f.keys()) > 0:
            start_idx = max([int(k) for k in f.keys()]) + 1
            print(f"Ripristinato dataset esistente. Ricomincio da {start_idx}.")
        else:
            start_idx = 0

    print(f"Inizio generazione...")
    diz, tot = DIJKSTRA_FUNCS[task_id]() 

    i = start_idx
    with h5py.File(H5_PATH, 'a') as f:
        # 50% top traj 
        print("50% top traj")
        for start_pos, trajectories in diz.items():
            pos_da_passare = list(start_pos)
            
            for path in trajectories:
                obs, reward, info = env.reset(start_pos=pos_da_passare) 
                
                buffer = { 'observations': [], 'dfa_states': [], 'actions': [], 'rewards': [], 'terminals': [] }
                
                for step_idx, act in enumerate(path):
                    dfa_state = obs[0]
                    current_state = obs[1]
                    
                    obs_nuova, reward, done, truncated, info = env.step(act)

                    if reward == 100.0:
                        done = True

                    # FIX WIN 
                    if reward_type == "three_value":
                        if step_idx == len(path) - 1:
                            reward = 100.0
                            done = True
                    
                    if state_type == "image":
                        buffer['observations'].append(current_state.detach().cpu().numpy().astype(np.float32))
                    elif state_type == "symbolic":
                        buffer['observations'].append(np.array(current_state, dtype=np.float32))
                    buffer['dfa_states'].append(np.array(dfa_state, dtype=np.float32))
                    buffer['actions'].append(act)
                    buffer['rewards'].append(reward)
                    buffer['terminals'].append(done)
                    obs = obs_nuova
                    
                    if done: break 

                # RTG
                raw_rewards = buffer['rewards']
                cumulativo = 0
                rtg_lista = []
                for r in reversed(raw_rewards):
                    cumulativo += r
                    rtg_lista.append(cumulativo)
                rtg_lista = list(reversed(rtg_lista))
                
                # H5 
                grp = f.create_group(str(i))
                grp.attrs['start_pos'] = np.array(start_pos)
                grp.attrs['length'] = len(buffer['actions'])
                grp.create_dataset('observations', data=np.stack(buffer['observations']))
                grp.create_dataset('dfa_states', data=np.stack(buffer['dfa_states']))
                grp.create_dataset('actions', data=np.array(buffer['actions']))
                grp.create_dataset('rewards', data=np.array(buffer['rewards']))
                grp.create_dataset('terminals', data=np.array(buffer['terminals']))
                grp.create_dataset('returns_to_go', data=np.array(rtg_lista))
                i += 1

        # 50% noise
        print("50% noise")
        for start_pos, trajectories in diz.items():
            pos_da_passare = list(start_pos)
            
            for path in trajectories:
                obs, reward, info = env.reset(start_pos=pos_da_passare) 
                
                buffer = { 'observations': [], 'dfa_states': [], 'actions': [], 'rewards': [], 'terminals': [] }
                
                for step_idx, act in enumerate(path):
                    if random.random() < EPSILON:
                        bad_act = random.choice([0, 1, 2, 3])
                        dfa_state = obs[0]
                        current_state = obs[1]
                        
                        obs_bad, bad_reward, bad_done, bad_trunc, info = env.step(bad_act)
                        

                        if state_type == "image":
                            buffer['observations'].append(current_state.detach().cpu().numpy().astype(np.float32))
                        elif state_type == "symbolic":
                            buffer['observations'].append(np.array(current_state, dtype=np.float32))
                        buffer['dfa_states'].append(np.array(dfa_state, dtype=np.float32))
                        buffer['actions'].append(bad_act)
                        buffer['rewards'].append(bad_reward)
                        buffer['terminals'].append(bad_done)
                        
                        if not bad_done and obs_bad[1].tolist() != current_state.tolist():
                            reverse_act = (bad_act + 2) % 4
                            dfa_state_rev = obs_bad[0]
                            current_state_rev = obs_bad[1]
                            
                            obs_rev, rev_reward, rev_done, rev_trunc, info = env.step(reverse_act)

                            if state_type == "image":
                                buffer['observations'].append(current_state_rev.detach().cpu().numpy().astype(np.float32))
                            elif state_type == "symbolic":
                                buffer['observations'].append(np.array(current_state_rev, dtype=np.float32))
                            buffer['dfa_states'].append(np.array(dfa_state_rev, dtype=np.float32))
                            buffer['actions'].append(reverse_act)
                            buffer['rewards'].append(rev_reward)
                            buffer['terminals'].append(rev_done)
                            obs = obs_rev 
                        else:
                            obs = obs_bad

                    dfa_state = obs[0]
                    current_state = obs[1]
                    
                    obs_nuova, reward, done, truncated, info = env.step(act)
                    
                    if reward == 100.0:
                        done = True

                    # FIX WIN 
                    if reward_type == "three_value":
                        if step_idx == len(path) - 1:
                            reward = 100.0
                            done = True

                    if state_type == "image":
                        buffer['observations'].append(current_state.detach().cpu().numpy().astype(np.float32))
                    elif state_type == "symbolic":
                        buffer['observations'].append(np.array(current_state, dtype=np.float32))
                    buffer['dfa_states'].append(np.array(dfa_state, dtype=np.float32))
                    buffer['actions'].append(act)
                    buffer['rewards'].append(reward)
                    buffer['terminals'].append(done)
                    obs = obs_nuova
                    
                    if done: break 

                # RTG
                raw_rewards = buffer['rewards']
                cumulativo = 0
                rtg_lista = []
                for r in reversed(raw_rewards):
                    cumulativo += r
                    rtg_lista.append(cumulativo)
                rtg_lista = list(reversed(rtg_lista))
                
                # H5
                grp = f.create_group(str(i))
                grp.attrs['start_pos'] = np.array(start_pos)
                grp.attrs['length'] = len(buffer['actions'])
                grp.create_dataset('observations', data=np.stack(buffer['observations']))
                grp.create_dataset('dfa_states', data=np.stack(buffer['dfa_states']))
                grp.create_dataset('actions', data=np.array(buffer['actions']))
                grp.create_dataset('rewards', data=np.array(buffer['rewards']))
                grp.create_dataset('terminals', data=np.array(buffer['terminals']))
                grp.create_dataset('returns_to_go', data=np.array(rtg_lista))
                i += 1
                
                if i % 100 == 0:
                    f.flush()
                    print(f"--> Checkpoint: {i} tot traj.")


def medium(task_id, file, state_type, reward_type):
    H5_PATH = file

    formula = formulas[task_id]

    env = GridWorldEnv(formula=formula, render_mode="rgb_array", state_type=state_type, train=False)

    with h5py.File(H5_PATH, 'a') as f:
        if len(f.keys()) > 0:
            start_idx = max([int(k) for k in f.keys()]) + 1
            print(f"Ripristinato dataset esistente. Ricomincio da {start_idx}.")
        else:
            start_idx = 0

    print(f"Inizio generazione...")
    diz, tot = DIJKSTRA_FUNCS[task_id]()

    i = start_idx
    with h5py.File(H5_PATH, 'a') as f:
        
        # 50% noise traj
        print("50% noise traj...")
        for start_pos, trajectories in diz.items():
            pos_da_passare = list(start_pos)
            
            for path in trajectories:
                obs, reward, info = env.reset(start_pos=pos_da_passare) 
                
                buffer = { 'observations': [], 'dfa_states': [], 'actions': [], 'rewards': [], 'terminals': [] }
                
                for step_idx, act in enumerate(path):
                    if random.random() < EPSILON:
                        bad_act = random.choice([0, 1, 2, 3])
                        dfa_state = obs[0]
                        current_state = obs[1]
                        
                        obs_bad, bad_reward, bad_done, bad_trunc, info = env.step(bad_act)

                        if state_type == "image":
                            buffer['observations'].append(current_state.detach().cpu().numpy().astype(np.float32))
                        elif state_type == "symbolic":
                            buffer['observations'].append(np.array(current_state, dtype=np.float32))
                        buffer['dfa_states'].append(np.array(dfa_state, dtype=np.float32))
                        buffer['actions'].append(bad_act)
                        buffer['rewards'].append(bad_reward)
                        buffer['terminals'].append(bad_done)
                        
                        if not bad_done and obs_bad[1].tolist() != current_state.tolist():
                            reverse_act = (bad_act + 2) % 4
                            dfa_state_rev = obs_bad[0]
                            current_state_rev = obs_bad[1]
                            
                            obs_rev, rev_reward, rev_done, rev_trunc, info = env.step(reverse_act)
                            
                            if state_type == "image":
                                buffer['observations'].append(current_state_rev.detach().cpu().numpy().astype(np.float32))
                            elif state_type == "symbolic":
                                buffer['observations'].append(np.array(current_state_rev, dtype=np.float32))
                            buffer['dfa_states'].append(np.array(dfa_state_rev, dtype=np.float32))
                            buffer['actions'].append(reverse_act)
                            buffer['rewards'].append(rev_reward)
                            buffer['terminals'].append(rev_done)
                            obs = obs_rev 
                        else:
                            obs = obs_bad

                    dfa_state = obs[0]
                    current_state = obs[1]
                    
                    obs_nuova, reward, done, truncated, info = env.step(act)

                    if reward == 100.0:
                        done = True
                    
                    # FIX WIN 
                    if reward_type == "three_value":
                        if step_idx == len(path) - 1:
                            reward = 100.0
                            done = True
                    
                    if state_type == "image":
                        buffer['observations'].append(current_state.detach().cpu().numpy().astype(np.float32))
                    elif state_type == "symbolic":
                        buffer['observations'].append(np.array(current_state, dtype=np.float32))
                    buffer['dfa_states'].append(np.array(dfa_state, dtype=np.float32))
                    buffer['actions'].append(act)
                    buffer['rewards'].append(reward)
                    buffer['terminals'].append(done)
                    obs = obs_nuova
                    
                    if done: break 

                # RTG
                raw_rewards = buffer['rewards']
                cumulativo = 0
                rtg_lista = []
                for r in reversed(raw_rewards):
                    cumulativo += r
                    rtg_lista.append(cumulativo)
                rtg_lista = list(reversed(rtg_lista))
                
                # H5 
                grp = f.create_group(str(i))
                grp.attrs['start_pos'] = np.array(start_pos)
                grp.attrs['length'] = len(buffer['actions'])
                grp.create_dataset('observations', data=np.stack(buffer['observations']))
                grp.create_dataset('dfa_states', data=np.stack(buffer['dfa_states']))
                grp.create_dataset('actions', data=np.array(buffer['actions']))
                grp.create_dataset('rewards', data=np.array(buffer['rewards']))
                grp.create_dataset('terminals', data=np.array(buffer['terminals']))
                grp.create_dataset('returns_to_go', data=np.array(rtg_lista))
                i += 1

        # 50% noise traj
        print("50% noise traj")
        for start_pos, trajectories in diz.items():
            pos_da_passare = list(start_pos)
            
            for path in trajectories:
                obs, reward, info = env.reset(start_pos=pos_da_passare) 
                
                buffer = { 'observations': [], 'dfa_states': [], 'actions': [], 'rewards': [], 'terminals': [] }
                
                for step_idx, act in enumerate(path):
                    if random.random() < EPSILON:
                        bad_act = random.choice([0, 1, 2, 3])
                        dfa_state = obs[0]
                        current_state = obs[1]
                        
                        obs_bad, bad_reward, bad_done, bad_trunc, info = env.step(bad_act)
                        

                        if state_type == "image":
                            buffer['observations'].append(current_state.detach().cpu().numpy().astype(np.float32))
                        elif state_type == "symbolic":
                            buffer['observations'].append(np.array(current_state, dtype=np.float32))
                        buffer['dfa_states'].append(np.array(dfa_state, dtype=np.float32))
                        buffer['actions'].append(bad_act)
                        buffer['rewards'].append(bad_reward)
                        buffer['terminals'].append(bad_done)
                        
                        if not bad_done and obs_bad[1].tolist() != current_state.tolist():
                            reverse_act = (bad_act + 2) % 4
                            dfa_state_rev = obs_bad[0]
                            current_state_rev = obs_bad[1]
                            
                            obs_rev, rev_reward, rev_done, rev_trunc, info = env.step(reverse_act)
                            

                            if state_type == "image":
                                buffer['observations'].append(current_state_rev.detach().cpu().numpy().astype(np.float32))
                            elif state_type == "symbolic":
                                buffer['observations'].append(np.array(current_state_rev, dtype=np.float32))
                            buffer['dfa_states'].append(np.array(dfa_state_rev, dtype=np.float32))
                            buffer['actions'].append(reverse_act)
                            buffer['rewards'].append(rev_reward)
                            buffer['terminals'].append(rev_done)
                            obs = obs_rev 
                        else:
                            obs = obs_bad

                    dfa_state = obs[0]
                    current_state = obs[1]
                    
                    obs_nuova, reward, done, truncated, info = env.step(act)

                    if reward == 100.0:
                        done = True

                    # FIX WIN 
                    if reward_type == "three_value":
                        if step_idx == len(path) - 1:
                            reward = 100.0
                            done = True
                    
                    if state_type == "image":
                        buffer['observations'].append(current_state.detach().cpu().numpy().astype(np.float32))
                    elif state_type == "symbolic":
                        buffer['observations'].append(np.array(current_state, dtype=np.float32))
                    buffer['dfa_states'].append(np.array(dfa_state, dtype=np.float32))
                    buffer['actions'].append(act)
                    buffer['rewards'].append(reward)
                    buffer['terminals'].append(done)
                    obs = obs_nuova
                    
                    if done: break 

                # RTG
                raw_rewards = buffer['rewards']
                cumulativo = 0
                rtg_lista = []
                for r in reversed(raw_rewards):
                    cumulativo += r
                    rtg_lista.append(cumulativo)
                rtg_lista = list(reversed(rtg_lista))
                
                # H5 
                grp = f.create_group(str(i))
                grp.attrs['start_pos'] = np.array(start_pos)
                grp.attrs['length'] = len(buffer['actions'])
                grp.create_dataset('observations', data=np.stack(buffer['observations']))
                grp.create_dataset('dfa_states', data=np.stack(buffer['dfa_states']))
                grp.create_dataset('actions', data=np.array(buffer['actions']))
                grp.create_dataset('rewards', data=np.array(buffer['rewards']))
                grp.create_dataset('terminals', data=np.array(buffer['terminals']))
                grp.create_dataset('returns_to_go', data=np.array(rtg_lista))
                i += 1
                
                if i % 100 == 0:
                    f.flush()
                    print(f"--> Checkpoint: {i} tot traj.")


def medium_replay(task_id, file, state_type, reward_type):
    H5_PATH = file

    formula = formulas[task_id]

    env = GridWorldEnv(formula=formula, render_mode="rgb_array", state_type=state_type, train=False)

    def save_to_h5(f, i, start_pos, buffer):
        raw_rewards = buffer['rewards']
        cumulativo = 0
        rtg_lista = []
        for r in reversed(raw_rewards):
            cumulativo += r
            rtg_lista.append(cumulativo)
        rtg_lista = list(reversed(rtg_lista))
        
        grp = f.create_group(str(i))
        grp.attrs['start_pos'] = np.array(start_pos)
        grp.attrs['length'] = len(buffer['actions'])
        grp.create_dataset('observations', data=np.stack(buffer['observations']))
        grp.create_dataset('dfa_states', data=np.stack(buffer['dfa_states']))
        grp.create_dataset('actions', data=np.array(buffer['actions']))
        grp.create_dataset('rewards', data=np.array(buffer['rewards']))
        grp.create_dataset('terminals', data=np.array(buffer['terminals']))
        grp.create_dataset('returns_to_go', data=np.array(rtg_lista))

    with h5py.File(H5_PATH, 'w') as f:
        i = 0
        diz, _ = DIJKSTRA_FUNCS[task_id]() 
        possible_starts = list(diz.keys())

        # 50% noise traj
        print(f"50% noise traj")
        for start_pos in possible_starts:
            trajectories = diz[start_pos][:NUM_PER_POS]
            
            for path in trajectories:
                obs, _, _ = env.reset(start_pos=list(start_pos))
                buffer = { 'observations': [], 'dfa_states': [], 'actions': [], 'rewards': [], 'terminals': [] }
                
                for step_idx, act in enumerate(path):
                    if random.random() < EPSILON:
                        bad_act = random.choice([0, 1, 2, 3])
                        obs_bad, r_bad, d_bad, _, _ = env.step(bad_act)
                        
                        if state_type == "image":
                            buffer['observations'].append(obs[1].detach().cpu().numpy().astype(np.float32))
                        elif state_type == "symbolic":
                            buffer['observations'].append(np.array(obs[1], dtype=np.float32))
                        buffer['dfa_states'].append(np.array(obs[0], dtype=np.float32))
                        buffer['actions'].append(bad_act)
                        buffer['rewards'].append(r_bad)
                        buffer['terminals'].append(d_bad)
                        
                        if not d_bad and obs_bad[1].tolist() != obs[1].tolist():
                            reverse_act = (bad_act + 2) % 4
                            obs, r_rev, d_rev, _, _ = env.step(reverse_act)
                            if state_type == "image":
                                buffer['observations'].append(obs_bad[1].detach().cpu().numpy().astype(np.float32))
                            elif state_type == "symbolic":
                                buffer['observations'].append(np.array(obs_bad[1], dtype=np.float32))
                            buffer['dfa_states'].append(np.array(obs_bad[0], dtype=np.float32))
                            buffer['actions'].append(reverse_act)
                            buffer['rewards'].append(r_rev)
                            buffer['terminals'].append(d_rev)
                        else: obs = obs_bad

                    obs_next, r, d, _, _ = env.step(act)

                    if r == 100.0:
                        d = True

                    # FIX WIN 
                    if reward_type == "three_value":
                        if step_idx == len(path) - 1:
                            r = 100.0
                            d = True
                

                    if state_type == "image":
                        buffer['observations'].append(obs[1].detach().cpu().numpy().astype(np.float32))
                    elif state_type == "symbolic":
                        buffer['observations'].append(np.array(obs[1], dtype=np.float32))
                    buffer['dfa_states'].append(np.array(obs[0], dtype=np.float32))
                    buffer['actions'].append(act)
                    buffer['rewards'].append(r)
                    buffer['terminals'].append(d)
                    obs = obs_next
                    if d: break
                
                save_to_h5(f, i, start_pos, buffer)
                i += 1

        # 50% random walk
        print(f"50% random walk")
        for start_pos in possible_starts:
            for _ in range(NUM_PER_POS):
                obs, _, _ = env.reset(start_pos=list(start_pos))
                buffer = { 'observations': [], 'dfa_states': [], 'actions': [], 'rewards': [], 'terminals': [] }
                
                for t in range(MAX_STEPS):
                    action = random.choice([0, 1, 2, 3])
                    obs_next, r, d, truncated, _ = env.step(action)
                    
                    if state_type == "image":
                        buffer['observations'].append(obs[1].detach().cpu().numpy().astype(np.float32))
                    elif state_type == "symbolic":
                        buffer['observations'].append(np.array(obs[1], dtype=np.float32))
                    buffer['dfa_states'].append(np.array(obs[0], dtype=np.float32))
                    buffer['actions'].append(action)
                    buffer['rewards'].append(r)
                    buffer['terminals'].append(d)
                    
                    obs = obs_next
                    if d or truncated: break
                
                save_to_h5(f, i, start_pos, buffer)
                i += 1
                if i % 100 == 0: print(f"Tot traj: {i}")


def generate_datasets(state_type, reward_type, task_id):
    paths = setup_folders(state_type, reward_type, task_id)

    h5_path = paths["datasets"] / 'dataset_medium_expert.h5'
    if not os.path.exists(h5_path): medium_expert(task_id, h5_path, state_type, reward_type)

    h5_path = paths["datasets"] / 'dataset_medium.h5'
    if not os.path.exists(h5_path): medium(task_id, h5_path, state_type, reward_type)

    h5_path = paths["datasets"] / 'dataset_medium_replay.h5'
    if not os.path.exists(h5_path): medium_replay(task_id, h5_path, state_type, reward_type)



