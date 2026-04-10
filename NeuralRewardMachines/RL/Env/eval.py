import torch
import numpy as np
import pandas as pd
from train_dt import DecisionTransformer 
from Environment import GridWorldEnv
from LTL_tasks import formulas
from utils import setup_folders
import os

possible_starts = [(0, 0), (1, 0), (2, 0), (0, 1), (2, 1), (3, 1),
                   (0, 2), (1, 2), (2, 2), (3, 2), (1, 3), (2, 3)]


def print_metrics(file):
    df = pd.read_csv(file)

    print(f"Reward: {df['reward_puro'].mean():.2f} +- {df['reward_puro'].std():.2f}")
    print(f"Cumulative Discounted Reward: {df['discounted_reward'].mean():.2f} +- {df['discounted_reward'].std():.2f}")
    print(f"Success Rate: {df['success'].mean() * 100:.1f}%")


def evaluate(task_id, dt_model, file):
    formula = formulas[task_id]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = DecisionTransformer(action_dim=4).to(device)
    model.load_state_dict(torch.load(dt_model)) 
    model.eval()

    env = GridWorldEnv(formula=formula, render_mode="rgb_array", state_type="image", train=False)
    
    gamma = 0.99 
    num_test = 10 
    target_return = 1.0 
    K = 10 
    

    all_runs_data = [] 

    for start_pos in possible_starts:
        print(f"\n--- Position: {start_pos} ---")
        
        for run in range(num_test):
            obs, reward, info = env.reset(start_pos=start_pos)
            current_img = obs[1] if isinstance(obs, list) else obs

            done = False
            truncated = False
            ep_reward = 0
            cumulative_discounted_reward = 0
            t_step = 0 
            
            states = [current_img]
            actions = []
            rtgs = [target_return]
            

            visited_items = set()
            if tuple(env._agent_location) == (1, 1): visited_items.add('pickaxe')
            if tuple(env._agent_location) == (3, 3): visited_items.add('lava')
            

            global_avoidance_tasks = [6, 7, 8, 9]

            while not done:
                states_window = states[-K:]
                while len(states_window) < K:
                    states_window.insert(0, np.zeros_like(states[0]))
                
                actions_window = actions[-K:]
                while len(actions_window) < K:
                    actions_window.insert(0, 0)
                
                rtgs_window = rtgs[-K:]
                while len(rtgs_window) < K:
                    rtgs_window.insert(0, target_return)

                s_tensor = torch.tensor(np.array(states_window), dtype=torch.float32).unsqueeze(0).to(device)
                a_tensor = torch.tensor(np.array(actions_window), dtype=torch.long).unsqueeze(0).to(device)
                r_tensor = torch.tensor(np.array(rtgs_window), dtype=torch.float32).unsqueeze(0).to(device)
                t_tensor = torch.tensor(np.arange(K), dtype=torch.long).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    s_input = s_tensor
                    action_preds = model(s_input, a_tensor, r_tensor, t_tensor)
                    action = torch.argmax(action_preds[0, -1]).item()
                
                step_result = env.step(action)
                next_obs, reward, done, truncated, info = step_result
                
                if isinstance(next_obs, list):
                    next_obs = next_obs[1]
                
                curr_pos = tuple(env._agent_location)
                if curr_pos == (1, 1): visited_items.add('pickaxe')
                if curr_pos == (3, 3): visited_items.add('lava')
                
                if task_id in global_avoidance_tasks:
                    if 'pickaxe' in visited_items and 'lava' in visited_items:
                        reward = 100.0
                        done = True

                cumulative_discounted_reward += (gamma ** t_step) * reward
                t_step += 1

                states.append(next_obs)
                actions.append(action)
                
                target_return -= (reward / 100.0)
                rtgs.append(target_return)
                ep_reward += reward
                
                if done or truncated:
                    break

            success_flag = 1 if ep_reward >= 100 else 0 

            all_runs_data.append({
                "start_pos": str(start_pos),
                "run_id": run + 1,
                "reward_puro": ep_reward,
                "discounted_reward": cumulative_discounted_reward,
                "success": success_flag,
                "episode_length": t_step,
                "action": actions
            })

            print(f"  Run {run+1} | Rewards: {ep_reward:.2f} | Cumulative Discounted: {cumulative_discounted_reward:.2f} | Length: {t_step}")

    df_risultati = pd.DataFrame(all_runs_data)
    df_risultati.to_csv(file, index=False)
    
    print("\n" + "="*50)
    print(f"Results saved in: {file}")
    print("="*50)

    print_metrics(file)



def eval_all(state_type, reward_type, task_id):
    paths = setup_folders(state_type, reward_type, task_id)

    dt_path = paths["models"] / 'dt_medium_expert_model.pth'
    csv_path = paths["eval"] / 'eval_dt_medium_expert.csv'
    if not os.path.exists(csv_path): evaluate(task_id, dt_path, csv_path)

    dt_path = paths["models"] / 'dt_medium_model.pth'
    csv_path = paths["eval"] / 'eval_dt_medium.csv'
    if not os.path.exists(csv_path): evaluate(task_id, dt_path, csv_path)

    dt_path = paths["models"] / 'dt_medium_replay_model.pth'
    csv_path = paths["eval"] / 'eval_dt_medium_replay.csv'
    if not os.path.exists(csv_path): evaluate(task_id, dt_path, csv_path)


