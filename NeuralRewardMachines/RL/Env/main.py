# SYMBOLIC
# import pandas as pd
# from utils import setup_folders
# from dataset_generation import generate_datasets
# from train_dt_symbolic import train_all
# from eval_symbolic import eval_all

# def print_metrics(state_type, reward_type, task_id):
#     paths = setup_folders(state_type, reward_type, task_id)
#     file = ['eval_dt_medium_expert.csv', 'eval_dt_medium.csv', 'eval_dt_medium_replay.csv']
#     for el in file: 
#         csv_path = paths["eval"] / el
#         df = pd.read_csv(csv_path)
#         print(f"Reward: {df['reward_puro'].mean():.2f} +- {df['reward_puro'].std():.2f}")
#         print(f"Cumulative Discounted Reward: {df['discounted_reward'].mean():.2f} +- {df['discounted_reward'].std():.2f}")
#         print(f"Success Rate: {df['success'].mean() * 100:.1f}%")

# def run_task(state_type, reward_type, task_id):
#     generate_datasets(state_type, reward_type, task_id)
#     train_all(state_type, reward_type, task_id)
#     eval_all(state_type, reward_type, task_id)

# if __name__ == "__main__":
#     state_type = "symbolic"
#     reward_type = "three_value"
#     # reward_type = "distance"

#     # single run
#     # TASK = 1
#     # task_id = TASK - 1
#     # run_task(state_type, reward_type, task_id)

#     # complete run
#     for NUM_TASK in range(1, 11):
#         task_id = NUM_TASK - 1

#         print("\n" + "="*60)
#         print(f"ELABORATION TASK {NUM_TASK} (ID: {task_id})")
#         print("="*60)

#         run_task(state_type, reward_type, task_id)
        
#         print(f"\nTASK {NUM_TASK} COMPLETed!")

#     print("\n10 TASK COMPLETED!")





# IMAGE -> three value
# from dataset_generation import generate_datasets
# from train_dt import train_all
# from eval import eval_all


# def run_task(state_type, reward_type, task_id):
#     generate_datasets(state_type, reward_type, task_id)
#     train_all(state_type, reward_type, task_id)
#     eval_all(state_type, reward_type, task_id)


# if __name__ == "__main__":
#     state_type = "image"
#     reward_type = "three_value"

#     # single run
#     # TASK = 1
#     # task_id = TASK - 1
#     # run_task(state_type, reward_type, task_id)

#     # complete run
#     for NUM_TASK in range(1, 11):
#         task_id = NUM_TASK - 1

#         print("\n" + "="*60)
#         print(f"ELABORATION TASK {NUM_TASK} (ID: {task_id})")
#         print("="*60)

#         run_task(state_type, reward_type, task_id)
        
#         print(f"\nTASK {NUM_TASK} COMPLETed!")

#     print("\n10 TASK COMPLETED!")





# IMAGE -> distance
from dataset_generation import generate_datasets
from train_dt import train_all
from eval_distance import eval_all_distance


def run_task(state_type, reward_type, task_id):
    generate_datasets(state_type, reward_type, task_id)
    train_all(state_type, reward_type, task_id)
    eval_all_distance(state_type, reward_type, task_id)


if __name__ == "__main__":
    state_type = "image"
    reward_type = "distance"

    # single run
    # TASK = 1
    # task_id = TASK - 1
    # run_task(state_type, reward_type, task_id)

    # complete run
    for NUM_TASK in range(1, 11):
        task_id = NUM_TASK - 1

        print("\n" + "="*60)
        print(f"ELABORATION TASK {NUM_TASK} (ID: {task_id})")
        print("="*60)

        run_task(state_type, reward_type, task_id)
        
        print(f"\nTASK {NUM_TASK} COMPLETed!")

    print("\n10 TASK COMPLETED!")