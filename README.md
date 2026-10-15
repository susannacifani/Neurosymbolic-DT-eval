# Neurosymbolic Offline RL on LTL Tasks

This repository studies offline reinforcement learning on non-Markovian tasks specified in **Linear Temporal Logic (LTL)**, in a 4x4 grid world with both **symbolic** (agent coordinates) and **image** (64x64 RGB) observations. It contains four components that share the same offline datasets:

1. **Dataset** – generation and validation of D4RL-style offline datasets (`medium_expert`, `medium`, `medium_replay`).
2. **Decision Transformer (DT)** – return-conditioned policy trained offline and evaluated in the environment.
3. **Grounding** – a Neural Reward Machine grounder that learns to map observations to the LTL symbols using **only the reward signal** (no symbol labels).
4. **World model** – an autoencoder plus a latent dynamics model that predicts the next state from the current state and the action.

The environment, the LTL-to-automaton machinery and the grounder build on [Neural Reward Machines](https://github.com/KRLGroup/NeuralRewardMachines) (Umili, Argenziano and Capobianco, ECAI 2024).

---

## Repository structure

```
.
├── Dataset/                  dataset generation and validation (+ the .h5 files, not on git)
│   ├── dataset_generation.py
│   ├── DijkstraTraj.py
│   └── check_datasets.py
├── DT/                       Decision Transformer (+ trained models and evaluations, not on git)
│   ├── decision_transformer.py
│   ├── experiments_dt.py
│   └── print_dt_table.py
├── Grounding/                symbol grounding (+ trained grounders and logs, not on git)
│   ├── experiments_grounding.py
│   └── print_grounding_table.py
├── WorldModel/               world model (+ trained models and logs, not on git)
│   ├── world_model.py
│   ├── experiments_world_model.py
│   └── print_world_model_table.py
├── NeuralRewardMachines/     shared code used by all components
│   ├── LTL_tasks.py              the 10 LTL tasks
│   └── RL/
│       ├── Env/                  grid world environment and Moore machine (automaton)
│       ├── NRM/                  probabilistic automaton, grounder networks, evaluation utils
│       └── A2C_nrm.py            offline training of the grounder
└── requirements.txt
```

Every script locates the repository on its own, so it can be launched from any folder (or with *Run* in an IDE). The commands below assume you are in the repository root.

## Setup

Tested on Ubuntu (WSL2) with Python 3.12, PyTorch 2.10 and an NVIDIA RTX 4060 GPU. Any Linux system should work; a GPU is optional but recommended for the image experiments. No display is needed: the scripts run pygame in headless mode (`SDL_VIDEODRIVER=dummy`).

```bash
git clone https://github.com/susannacifani/Neurosymbolic-DT-eval.git
cd Neurosymbolic-DT-eval
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Environment and tasks

The grid is 4x4. The agent starts in one of the 12 empty cells and moves with 4 actions (0: +y, 1: +x, 2: −y, 3: −x; moving into a wall leaves it in place). Four cells contain an object, each associated with an LTL symbol:

| Symbol | Object  | Cell (x, y) |
|--------|---------|-------------|
| `P`    | pickaxe | (1, 1)      |
| `L`    | lava    | (3, 3)      |
| `D`    | door    | (3, 0)      |
| `G`    | gem     | (0, 3)      |
| `E`    | empty   | all others  |

The 10 tasks (`NeuralRewardMachines/LTL_tasks.py`) combine visits, sequential visits and global avoidance:

| Task | Formula | Description |
|------|---------|-------------|
| 1  | `F P & F L` | visit pickaxe and lava (any order) |
| 2  | `F P & F L & F D` | visit pickaxe, lava and door |
| 3  | `F(P & F L)` | visit pickaxe, then lava |
| 4  | `F(P & F L) & F(D & F G)` | pickaxe then lava, and door then gem |
| 5  | `F(P & F L) & F D` | pickaxe then lava, and visit door |
| 6  | `F(P & F L) & F D & F G` | pickaxe then lava, and visit door and gem |
| 7  | `F P & F L & G !D` | visit pickaxe and lava, never touch the door |
| 8  | `F P & F L & G !D & G !G` | visit pickaxe and lava, never touch door or gem |
| 9  | `F(P & F L) & G !D` | pickaxe then lava, never touch the door |
| 10 | `F(P & F L) & G !D & G !G` | pickaxe then lava, never touch door or gem |

Each formula is compiled into a Moore machine. An episode ends when the automaton reaches an accepting state (**success**) or a state from which acceptance is impossible, i.e. a violated `G !x` (**failure**).

### Reward schemes

The reward is defined on the automaton state and given as the change in its value at each step:

- **`three_value`** – sparse: +100 when the task is completed, −100 on failure, 0 otherwise.
- **`distance`** – dense: the automaton state value grows with progress towards acceptance (e.g. 0 → 50 → 100 for task 1, steps of 25 for tasks with four sub-goals), and −100 on failure. The per-step reward is the increase in progress, so the episode return is still 100 on success.

---

## 1. Dataset

The ready-made datasets can be downloaded instead of generated: see [Large files](#large-files).

### Generation

```bash
python3 Dataset/dataset_generation.py
```

Edit the configuration block at the bottom of the script to choose state types, reward schemes and tasks. Files that already exist are skipped. Each file is written to a temporary file first and renamed at the end, so an interrupted run never leaves a partial file. Output:

```
Dataset/<Symbolic|Image>/<distance|three_value>/task<N>/dataset_<medium_expert|medium|medium_replay>.h5
```

Each file contains **2016 trajectories** (168 per start cell × 12 start cells). They are built from the k = 84 shortest paths (`DijkstraTraj.py`) that solve the task from each start cell:

| Dataset | Content |
|---------|---------|
| `medium_expert` | 50% optimal shortest paths + 50% noisy shortest paths |
| `medium` | 100% noisy shortest paths |
| `medium_replay` | 50% noisy shortest paths + 50% uniformly random walks (max 50 steps) |

*Noisy* means that at each step, with probability ε = 0.2, the agent takes a random action and then the reverse action to come back, before continuing along the path. A noisy or random move can end the episode (success or failure). The trajectory is then cut there, so no step is ever recorded after the end of an episode.

Generation is seeded per task and dataset, so **symbolic and image files contain the same trajectories** (only the observation differs), and so do `distance` and `three_value` (only the reward differs).

### File format

Each trajectory is a group `"<i>"` with:

| Field | Shape | Meaning |
|-------|-------|---------|
| `observations` | (T, 2) or (T, 3, 64, 64) | state **before** each action |
| `actions` | (T,) | action taken |
| `rewards` | (T,) | reward obtained **by** that action |
| `terminals` | (T,) | whether that action ended the episode |
| `returns_to_go` | (T,) | sum of the rewards from step t to the end |
| `final_observation` | (2,) or (3, 64, 64) | state **after** the last action |
| `dfa_states` | (T, n_states) | one-hot automaton state before each action |
| attrs `start_pos`, `length` | | start cell and number of steps T |

> ⚠️ `observations[t]` is the state *before* action t, while `rewards[t]` is the reward *after* it. The cell where the episode ends (where the success or failure happens) is only in `final_observation`.
>
> The numbering of the automaton states in `dfa_states` can differ between runs of the generator (the automaton library numbers states non-deterministically). Only the automaton *behaviour* is fixed.

### Validation

```bash
python3 Dataset/check_datasets.py                                   # everything
python3 Dataset/check_datasets.py symbolic distance,three_value 1,2,3
```

For every trajectory it replays the automaton and checks the transitions, the per-step rewards, `terminals`, `returns_to_go`, the success label and `final_observation`, and that no steps are recorded after the end. It also checks that symbolic and image files contain identical trajectories. It ends with `RISULTATO: TUTTO OK` when everything matches.

---

## 2. Decision Transformer

```bash
python3 DT/experiments_dt.py            # train + evaluate (configuration at the top of the file)
python3 DT/print_dt_table.py            # results table
```

In `experiments_dt.py`, set `STATE_TYPES`, `REWARD_TYPES`, `TASKS`, `DATASETS` and `SEEDS`. With `OVERWRITE = False`, runs whose model or evaluation already exist are skipped, so an interrupted run can simply be relaunched.

**Model.** GPT-style causal transformer over (return-to-go, state, action) tokens. It has 3 layers, 4 heads, hidden size 128 and dropout 0.1. The state encoder is a CNN for images and a linear layer for symbolic states. The context length is K = 50, which covers the whole episode.

**Training.** Cross-entropy on the actions, with AdamW (lr 1e-4, weight decay 1e-4) and batch size 64 (symbolic) or 32 (image). The trajectories are split 80/20 into train and validation. Training uses early stopping on the validation loss (patience 10, at most 500 epochs), and the best model is kept.

**Evaluation.** One deterministic episode from each of the 12 start cells. The target return is the maximum (100) and episodes are capped at 100 steps. Each episode ends in success, failure or timeout, read from the automaton state.

**Output**, in `DT/<State>/<reward>/task<N>/seed<S>/`:

```
dt_model/dt_<dataset>.pth          model + the configuration it was trained with
log/train_log_<dataset>.txt        hyperparameters, train/val loss and accuracy per epoch
eval/eval_<dataset>.csv            one row per episode (outcome, reward, discounted reward, length, actions)
eval/eval_summary_<dataset>.txt    summary of the evaluation
```

**Table.** `print_dt_table.py [metric] [seeds]` reports mean ± standard deviation across seeds:
- default `all`: Reward, CDR (cumulative discounted reward, γ = 0.99) and SR (success rate, 0–100) together;
- or a single metric: `reward`, `discounted`, `success` or `length`.

---

## 3. Grounding

```bash
python3 Grounding/experiments_grounding.py      # configuration at the top of the file
python3 Grounding/print_grounding_table.py [seeds]
```

**Method.** A grounder network maps each observation to a probability distribution over the 5 symbols (`P`, `L`, `D`, `G`, `E`):
- an MLP (2 → 128 → 128 → 5, with BatchNorm) for symbolic states;
- a CNN for images.

The predicted symbols are fed to a **fixed, differentiable probabilistic automaton** built from the task formula, which outputs the probability of each reward class at every step. The grounder is trained only to reproduce the observed reward sequence. It never sees the true symbols.

**Training details** (`NeuralRewardMachines/RL/A2C_nrm.py`):
- **Split.** 80/10/10 into train/validation/test trajectories. For `distance` the split is balanced between low-return and high-return trajectories.
- **Labels.** The reward class of every observation, including `final_observation` (which carries the success or failure class).
- **Optimisation.** The automaton is normalised (row-stochastic transitions) and the loss is the negative log-likelihood. Class weights are 1/frequency to compensate for rare classes such as success. Adam (lr 1e-3) with ReduceLROnPlateau, for 100 epochs.
- **Model selection.** The model with the best balanced accuracy on validation is kept.
- **Data loading.** There are only 16 distinct observations (one per cell), so each file is read once and batches are built from that table. The inputs are identical to reading the file every epoch.

**Metrics** (on the test set, written at the end of each log):
- **Balanced accuracy**: mean recall over the reward classes. It shows whether grounder + automaton predict the reward correctly.
- **Symbol accuracy**: whether the grounder predicts the true symbol of each cell, in two versions:
  - all symbols;
  - *formula symbols only, up to the symmetries of the formula*. Objects the formula does not use count as empty, and symbol permutations that leave the formula unchanged (e.g. `P ↔ L` in task 1) are not counted as errors. The symmetries are computed automatically from the automaton. This is the version reported in the table, because it measures only what can be identified from the rewards.
- The log also contains the per-class recall and the full map *true symbol → predicted symbol* for every cell. The map makes reasoning shortcuts visible: high reward accuracy with wrong symbols.

**Output**, in `Grounding/<State>/<reward>/task<N>/seed<S>/`:

```
grounder/grounder_<dataset>.pth
log/log_<dataset>.txt
```

---

## 4. World model

```bash
python3 WorldModel/experiments_world_model.py                     # configuration at the top of the file
python3 WorldModel/print_world_model_table.py [metric] [seeds]    # metric: acc (default), baseline, mse, recon
```

**Method.** It has two parts, trained one after the other:
- **Autoencoder.** It compresses each observation into a latent vector z. It is convolutional for images (z of size 16) and an MLP for symbolic states (z of size 4). It is trained for 20 epochs on reconstruction.
- **Dynamics model.** An MLP that predicts the next latent state z' from (z, action). It is trained for 50 epochs on the latent MSE.

Every transition is used, including the last one of each trajectory (towards `final_observation`). The split is 80/10/10 by trajectory.

The world model does not use the reward, and `distance` and `three_value` datasets contain the same trajectories. So there is **one world model per task and dataset**, trained on the `distance` files.

**Metrics** (test set):
- **acc**: next-state position accuracy. The predicted z' is decoded and matched to the closest of the 16 cells. The metric is the percentage of transitions where that cell is the true next cell.
- **baseline**: the same accuracy for the identity model z' = z ("the agent does not move"). It is correct only when the agent bumps into a wall.
- **mse**: latent MSE. It depends on the scale of the latent space, so do not compare symbolic and image.
- **recon**: best validation reconstruction MSE of the autoencoder.

**Output**, in `WorldModel/<State>/task<N>/seed<S>/`:

```
model/autoencoder_<dataset>.pth, model/dynamics_<dataset>.pth
latent/latent_transitions_<dataset>.npz     z, a, z' for train / val / test
log/log_<dataset>.txt
```

---

## Seeds and reproducibility

All launchers set the seed before every single run, so each run is reproducible on its own. The seed controls the network initialisation and the train/validation/test split. Results are reported as mean ± standard deviation over seeds 0, 1 and 2.

To add a seed, set `SEEDS = [<seed>]` in the launcher: completed runs are skipped. The table scripts use all seeds they find, or the ones given on the command line (e.g. `0,1`).

## Large files

Datasets (`*.h5`, about 0.5 GB symbolic and 93 GB image), trained models and logs are not stored on git (see `.gitignore`).

**Download the datasets** from Google Drive: <https://drive.google.com/drive/folders/1vBQvduY6Vnc7rrQoShhDYi_72Nc3YBfD?usp=sharing>

| Archive | Content | Size |
|---------|---------|------|
| `Dataset_Symbolic.zip` | all symbolic datasets (`distance` and `three_value`) | ~22 MB |
| `Dataset_Image_distance.zip` | image datasets, `distance` reward | ~6 GB |
| `Dataset_Image_three_value.zip` | image datasets, `three_value` reward | ~6 GB |

Extract the archives in the **repository root**. They already contain the `Dataset/...` folder structure, so the files end up where the scripts expect them:

```bash
unzip Dataset_Symbolic.zip                # -> Dataset/Symbolic/<reward>/task<N>/dataset_<dataset>.h5
unzip Dataset_Image_distance.zip          # -> Dataset/Image/distance/task<N>/...
unzip Dataset_Image_three_value.zip       # -> Dataset/Image/three_value/task<N>/...
python3 Dataset/check_datasets.py         # optional: verify the files (ends with "TUTTO OK")
```

You only need the archives for the state type you want to use. Alternatively, the datasets can be regenerated with `Dataset/dataset_generation.py`, which reproduces exactly the same trajectories.

## Acknowledgements

The grid world environment, the LTL-to-automaton machinery and the grounder build on the official repository of *Neural Reward Machines* (Elena Umili, Francesco Argenziano and Roberto Capobianco, ECAI 2024): [KRLGroup/NeuralRewardMachines](https://github.com/KRLGroup/NeuralRewardMachines).
