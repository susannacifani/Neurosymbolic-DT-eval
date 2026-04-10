import pandas as pd
from pathlib import Path
import re 

def generate_master_table():
    base_path = Path("Results")
    all_results = []
    
    for csv_path in base_path.rglob("eval/eval_*.csv"):
        parts = csv_path.parts
        
        try:
            state_type = parts[-5]  
            reward_type = parts[-4] 
            task = parts[-3]        
            
            model = csv_path.stem.replace('eval_', '') 

            df = pd.read_csv(csv_path)
            
            rew_mean = df['reward_puro'].mean()
            rew_std = df['reward_puro'].std()
            
            disc_mean = df['discounted_reward'].mean()
            disc_std = df['discounted_reward'].std()
            
            success_rate = df['success'].mean() * 100
            
            match = re.search(r'\d+', task)
            task_num = int(match.group()) if match else 999 
            
            all_results.append({
                'State': state_type,
                'Task_Num': task_num,  
                'Task': task,
                'Model': model,
                'Reward Type': reward_type,
                'SR (%)': f"{success_rate:.1f}%",
                'Reward': f"{rew_mean:.2f} ± {rew_std:.2f}",
                'CDR': f"{disc_mean:.2f} ± {disc_std:.2f}"
            })
            
        except Exception as e:
            print(f"Error {csv_path}: {e}")

    if not all_results:
        print("No CSV founded!")
        return

    df_raw = pd.DataFrame(all_results)

    custom_model_order = ['medium_expert', 'medium', 'medium_replay'] 
    
    found_models = df_raw['Model'].unique().tolist()
    model_types_sorted = [m for m in custom_model_order if m in found_models] + \
                         [m for m in found_models if m not in custom_model_order]
    
    df_raw['Model'] = pd.Categorical(df_raw['Model'], categories=model_types_sorted, ordered=True)

    master_table = df_raw.pivot(
        index=['State', 'Task_Num', 'Task', 'Model'], 
        columns='Reward Type', 
        values=['Reward', 'CDR', 'SR (%)']
    )

    master_table = master_table.swaplevel(0, 1, axis=1)
    
    custom_reward_order = ['three_value', 'distance'] 
    
    found_rewards = df_raw['Reward Type'].unique().tolist()
    reward_types_sorted = sorted(found_rewards, key=lambda x: custom_reward_order.index(x) if x in custom_reward_order else 999)
    
    desired_metrics_order = ['Reward', 'CDR', 'SR (%)']
    
    new_columns_order = [(rt, metric) for rt in reward_types_sorted for metric in desired_metrics_order]
    
    master_table = master_table.reindex(columns=new_columns_order)
    
    master_table = master_table.sort_index(level=['State', 'Task_Num', 'Task', 'Model'])

    master_table = master_table.reset_index(level='Task_Num', drop=True)

    master_table.columns.names = [None, None]

    print("\n" + "="*80)
    print("TABLE")
    print("="*80)
    print(master_table.to_string(justify='center'))

    save_dir = base_path / "Summary"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    master_table.to_csv(save_dir / "master_table.csv")
    with open(save_dir / "master_table.md", "w") as f:
        f.write(master_table.to_markdown())

    print(f"\nSaved in {save_dir}")

if __name__ == "__main__":
    generate_master_table()