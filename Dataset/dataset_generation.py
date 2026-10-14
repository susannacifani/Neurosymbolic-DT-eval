import os
import random
import sys
from pathlib import Path

import h5py
import numpy as np

# cartella Dataset/ (dove sta questo script) e motore comune in NRM/NeuralRewardMachines
DATASET_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(DATASET_ROOT.parent / "NeuralRewardMachines"))
sys.path.insert(0, str(DATASET_ROOT))

from LTL_tasks import formulas
from RL.Env.Environment import GridWorldEnv
from DijkstraTraj import task1, task2, task3, task4, task5, task6, task7, task8, task9, task10

EPSILON = 0.2  # 20% probability of noise injection
MAX_STEPS = 50 
NUM_PER_POS = 84 

DIJKSTRA_FUNCS = {
                    0: task1, 1: task2, 2: task3, 3: task4, 4: task5,
                    5: task6, 6: task7, 7: task8, 8: task9, 9: task10
                 }




def final_observation(env, state_type):
    """Osservazione dopo l'ultima azione della traiettoria.
    Le righe del dataset sono (s_t, a_t, r_t) con s_t lo stato PRIMA dell'azione, quindi lo
    stato in cui l'episodio finisce (es. la casella della vittoria) non sarebbe salvato."""
    x, y = env._agent_location
    if state_type == "image":
        return env.image_locations[x, y].detach().cpu().numpy().astype(np.float32)
    return np.array(env._agent_location, dtype=np.float32)


def medium_expert(task_id, file, state_type, reward_type):
    H5_PATH = file

    formula = formulas[task_id]


    env = GridWorldEnv(formula=formula, render_mode="rgb_array", state_type=state_type, train=False, reward_type=reward_type)

    print(f"Inizio generazione...")
    diz, tot = DIJKSTRA_FUNCS[task_id]()

    i = 0
    with h5py.File(H5_PATH, 'w') as f:
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
                grp.create_dataset('final_observation', data=final_observation(env, state_type))
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
                        # la mossa di rumore puo' chiudere l'episodio (vittoria o fallimento)
                        if bad_done: break

                        if obs_bad[1].tolist() != current_state.tolist():
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
                            if rev_done: break
                        else:
                            obs = obs_bad

                    dfa_state = obs[0]
                    current_state = obs[1]

                    obs_nuova, reward, done, truncated, info = env.step(act)

                    if reward == 100.0:
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
                grp.create_dataset('final_observation', data=final_observation(env, state_type))
                i += 1
                
                if i % 100 == 0:
                    f.flush()
                    print(f"--> Checkpoint: {i} tot traj.")


def medium(task_id, file, state_type, reward_type):
    H5_PATH = file

    formula = formulas[task_id]

    env = GridWorldEnv(formula=formula, render_mode="rgb_array", state_type=state_type, train=False, reward_type=reward_type)

    print(f"Inizio generazione...")
    diz, tot = DIJKSTRA_FUNCS[task_id]()

    i = 0
    with h5py.File(H5_PATH, 'w') as f:
        
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
                        # la mossa di rumore puo' chiudere l'episodio (vittoria o fallimento)
                        if bad_done: break

                        if obs_bad[1].tolist() != current_state.tolist():
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
                            if rev_done: break
                        else:
                            obs = obs_bad

                    dfa_state = obs[0]
                    current_state = obs[1]
                    
                    obs_nuova, reward, done, truncated, info = env.step(act)

                    if reward == 100.0:
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
                grp.create_dataset('final_observation', data=final_observation(env, state_type))
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
                        # la mossa di rumore puo' chiudere l'episodio (vittoria o fallimento)
                        if bad_done: break

                        if obs_bad[1].tolist() != current_state.tolist():
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
                            if rev_done: break
                        else:
                            obs = obs_bad

                    dfa_state = obs[0]
                    current_state = obs[1]
                    
                    obs_nuova, reward, done, truncated, info = env.step(act)

                    if reward == 100.0:
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
                grp.create_dataset('final_observation', data=final_observation(env, state_type))
                i += 1
                
                if i % 100 == 0:
                    f.flush()
                    print(f"--> Checkpoint: {i} tot traj.")


def medium_replay(task_id, file, state_type, reward_type):
    H5_PATH = file

    formula = formulas[task_id]

    env = GridWorldEnv(formula=formula, render_mode="rgb_array", state_type=state_type, train=False, reward_type=reward_type)

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
        grp.create_dataset('final_observation', data=final_observation(env, state_type))

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
                        # la mossa di rumore puo' chiudere l'episodio (vittoria o fallimento)
                        if d_bad: break

                        if obs_bad[1].tolist() != obs[1].tolist():
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
                            if d_rev: break
                        else: obs = obs_bad

                    obs_next, r, d, _, _ = env.step(act)

                    if r == 100.0:
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


def dataset_folder(state_type, reward_type, task_id):
    """Dataset/<State>/<reward>/task<N>/ (contiene i 3 file .h5)"""
    folder = DATASET_ROOT / state_type.capitalize() / reward_type / f"task{task_id + 1}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def generate_datasets(state_type, reward_type, task_id):
    folder = dataset_folder(state_type, reward_type, task_id)

    generators = [("medium_expert", medium_expert), ("medium", medium), ("medium_replay", medium_replay)]
    for dataset_idx, (name, generator) in enumerate(generators):
        h5_path = folder / f'dataset_{name}.h5'
        if os.path.exists(h5_path):
            continue
        # stesso seed per symbolic e image (e per distance/three_value): a parita' di
        # task e dataset le traiettorie sono le stesse, cambia solo l'osservazione
        random.seed(1000 * task_id + dataset_idx)
        # si scrive su un file temporaneo e lo si rinomina solo a fine generazione,
        # cosi' un'interruzione non lascia un file parziale che verrebbe poi saltato
        tmp_path = h5_path.with_suffix('.h5.tmp')
        generator(task_id, tmp_path, state_type, reward_type)
        os.replace(tmp_path, h5_path)


if __name__ == "__main__":
    # =====================================================================
    # CONFIGURAZIONE: modifica qui cosa generare, poi premi Run
    # (i file gia' presenti vengono saltati)
    # =====================================================================
    STATE_TYPES = ["symbolic", "image"]
    REWARD_TYPES = ["distance", "three_value"]
    TASKS = range(1, 11)
    # =====================================================================
    for state_type in STATE_TYPES:
        for reward_type in REWARD_TYPES:
            for task in TASKS:
                print(f"\n{'=' * 60}\n {state_type} / {reward_type} / task{task}\n{'=' * 60}")
                generate_datasets(state_type, reward_type, task - 1)
