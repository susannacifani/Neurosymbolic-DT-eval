import torch
import numpy as np
import pandas as pd
from train_dt_symbolic import DecisionTransformer 
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



def evaluate(state_type, reward_type, task_id, dt_model_path, file):
    formula = formulas[task_id]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    env = GridWorldEnv(formula=formula, render_mode="rgb_array", state_type=state_type, train=False)

    state_dim = 2 

    model = DecisionTransformer(state_dim=state_dim, action_dim=4).to(device)
    model.load_state_dict(torch.load(dt_model_path, map_location=device)) 
    model.eval()

    target_return = 1.0 
    K = 10  
    gamma = 0.99
    
    all_runs_data = [] 

    for start_pos in possible_starts:
        print(f"\n--- Position: {start_pos} ---")
        for run in range(10): 
            obs, reward, info = env.reset(start_pos=list(start_pos))
            
            # obs[1]: coord (x, y)
            curr_pos_norm = np.array(obs[1], dtype=np.float32) / 3.0
            
            done, truncated = False, False
            ep_reward, t_step = 0, 0
            cumulative_discounted_reward = 0
            

            states = [curr_pos_norm]
            actions = [0]  
            rtgs = [target_return]
            
            history_actions = []

            # check task Avoidance (6, 7, 8, 9)
            visited_items = set()
            global_avoidance_tasks = [6, 7, 8, 9]

            while not done and not truncated:
                s_win = torch.tensor(np.array(states[-K:]), dtype=torch.float32).unsqueeze(0).to(device)
                a_win = torch.tensor(np.array(actions[-K:]), dtype=torch.long).unsqueeze(0).to(device)
                r_win = torch.tensor(np.array(rtgs[-K:]), dtype=torch.float32).unsqueeze(0).to(device)
                
                start_t = max(0, t_step - K + 1)
                t_win = torch.tensor(np.arange(start_t, start_t + s_win.shape[1]), dtype=torch.long).unsqueeze(0).to(device)

                with torch.no_grad():
                    action_preds = model(s_win, a_win, r_win, t_win)
                    action = torch.argmax(action_preds[0, -1]).item()
                
                next_obs, reward, done, truncated, info = env.step(action)

                # --- Check avoidance ---
                curr_pos = tuple(env._agent_location)
                if curr_pos == (1, 1): visited_items.add('pickaxe')
                if curr_pos == (3, 3): visited_items.add('lava')
                
                if task_id in global_avoidance_tasks:
                    if reward_type != "distance":
                        if 'pickaxe' in visited_items and 'lava' in visited_items:
                            reward = 100.0 
                            done = True

                next_pos_norm = np.array(next_obs[1], dtype=np.float32) / 3.0

                history_actions.append(int(action))

                ep_reward += reward
                cumulative_discounted_reward += (gamma ** t_step) * reward
                t_step += 1

                states.append(next_pos_norm)
                actions[-1] = action
                actions.append(0)
                
                # Update Returns-to-go (Reward Scaling / 100)
                rtgs.append(rtgs[-1] - (reward / 100.0))
                
                if t_step >= 50: break

            # Success if ep_reward >= 100 
            success_flag = 1 if ep_reward >= 100.0 else 0 
            
            all_runs_data.append({
                "start_pos": str(start_pos), 
                "run_id": run + 1,
                "reward_puro": ep_reward,
                "discounted_reward": cumulative_discounted_reward,
                "success": success_flag,
                "episode_length": t_step,
                "action": str(history_actions)
            })
            print(f"  Run {run+1} | Rewards: {ep_reward:.2f} | Cumulative Discounted: {cumulative_discounted_reward:.2f} | Length: {t_step}")


    df = pd.DataFrame(all_runs_data)
    df.to_csv(file, index=False)
    print(f"Results saved in: {file}")
    print_metrics(file)



def eval_all(state_type, reward_type, task_id):
    paths = setup_folders(state_type, reward_type, task_id)

    dt_path = paths["models"] / 'dt_medium_expert_model.pth'
    csv_path = paths["eval"] / 'eval_dt_medium_expert.csv'
    if not os.path.exists(csv_path): evaluate(state_type, reward_type, task_id, dt_path, csv_path)

    dt_path = paths["models"] / 'dt_medium_model.pth'
    csv_path = paths["eval"] / 'eval_dt_medium.csv'
    if not os.path.exists(csv_path): evaluate(state_type, reward_type, task_id, dt_path, csv_path)

    dt_path = paths["models"] / 'dt_medium_replay_model.pth'
    csv_path = paths["eval"] / 'eval_dt_medium_replay.csv'
    if not os.path.exists(csv_path): evaluate(state_type, reward_type, task_id, dt_path, csv_path)