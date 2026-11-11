"""Dataset offline per la griglia 7x7 con NUM_LAYOUTS disposizioni fisse degli oggetti.

Le disposizioni sono le stesse in training e in test; cambia solo la partenza (casuale tra le caselle
vuote). Sono le prime NUM_LAYOUTS risolvibili a partire dal seed LAYOUT_SEED_BASE; lo stesso seed da'
la stessa disposizione in ogni task.

Tipi di dataset:
  medium_expert: 50% percorsi ottimi + 50% percorsi ottimi con rumore
  medium:        100% percorsi ottimi con rumore
  medium_replay: 50% percorsi ottimi con rumore + 50% passeggiate casuali
Percorso ottimo: planner.shortest_path, uno a caso tra i percorsi minimi. Rumore: a ogni passo, con
probabilita' EPSILON, una mossa casuale, dopo la quale si ripianifica dalla situazione raggiunta.

Training: Dataset/<reward>/task<N>/dataset_<dataset>.h5, NUM_TRAJECTORIES traiettorie; la traiettoria i
usa la disposizione i % NUM_LAYOUTS. Un gruppo per traiettoria con
  attrs:  seed (della disposizione), layout (indice), length (T), kind (optimal / noisy / random)
  grid       (7, 7)   disposizione, indicizzata [x, y] (0 P, 1 L, 2 D, 3 G, 4 vuota)
  positions  (T+1, 2) posizione dell'agente prima di ogni azione + posizione finale
  actions, rewards, terminals, returns_to_go  (T,)
Le osservazioni (mappa o immagine) si ricostruiscono da grid e positions: lo stesso file serve per
symbolic e image.
Test: Dataset/<reward>/task<N>/test_<dataset>.npz, NUM_TEST episodi (layout, grid, start, seen): l'episodio
j usa la disposizione j % NUM_LAYOUTS; gli episodi sono gli stessi per i 3 dataset, seen indica se la
coppia (disposizione, partenza) compare nel training di quel dataset.
I file gia' presenti vengono saltati. Verifica: Dataset/check_datasets.py.
Uso (dalla cartella NRM): python3 Grid7/Dataset/dataset_generation.py
"""
import os
import sys
from pathlib import Path

import h5py
import numpy as np

DATASET_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(DATASET_ROOT.parent / "NeuralRewardMachines"))
sys.path.insert(0, str(DATASET_ROOT))

from LTL_tasks import formulas
from RL.Env.Environment import EMPTY, GridWorldEnv
from planner import shortest_path

# =====================================================================
# CONFIGURAZIONE
# =====================================================================
REWARD_TYPES = ["three_value", "distance"]
TASKS = range(1, 11)
DATASET_NAMES = ["medium_expert", "medium", "medium_replay"]
# =====================================================================

EPSILON = 0.75
NUM_TRAJECTORIES = 5000
NUM_LAYOUTS = 50
NUM_TEST = 500
LAYOUT_SEED_BASE = 3_000_000


def dataset_folder(reward_type, task_id):
    folder = DATASET_ROOT / reward_type / f"task{task_id + 1}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def run_episode(env, kind, rng):
    """Una traiettoria di tipo kind dall'ambiente gia' resettato; None se non c'e' soluzione."""
    traj = {"positions": [env._agent_location.copy()], "actions": [], "rewards": [], "terminals": []}

    def do(action):
        _, reward, done, truncated, _ = env.step(action)
        traj["positions"].append(env._agent_location.copy())
        traj["actions"].append(int(action))
        traj["rewards"].append(float(reward))
        traj["terminals"].append(bool(done))
        return done or truncated

    if kind == "random":
        while not do(rng.integers(4)):
            pass
        return traj

    path = shortest_path(env, rng)
    if path is None:
        return None
    if kind == "optimal":
        for action in path:
            if do(action):
                return traj
        return traj

    # se non c'e' piu' un percorso senza fallimento, path e' vuoto e si continua con mosse casuali
    while True:
        if rng.random() < EPSILON or not path:
            if do(int(rng.integers(4))):
                return traj
            path = shortest_path(env, rng) or []
        elif do(path.pop(0)):
            return traj


def kinds(dataset):
    half = NUM_TRAJECTORIES // 2
    if dataset == "medium_expert":
        return ["optimal"] * half + ["noisy"] * (NUM_TRAJECTORIES - half)
    if dataset == "medium":
        return ["noisy"] * NUM_TRAJECTORIES
    return ["noisy"] * half + ["random"] * (NUM_TRAJECTORIES - half)


def traj_rng(task_id, dataset, i):
    if dataset == "medium":
        return np.random.default_rng([LAYOUT_SEED_BASE, task_id, i])
    return np.random.default_rng([LAYOUT_SEED_BASE, task_id, DATASET_NAMES.index(dataset), i])


def layout_pool(env, n_layouts=NUM_LAYOUTS):
    """(seed, grid) delle prime n_layouts disposizioni risolvibili da LAYOUT_SEED_BASE"""
    pool, seed = [], LAYOUT_SEED_BASE
    while len(pool) < n_layouts:
        env.reset(seed=seed)
        if shortest_path(env, np.random.default_rng(0)) is not None:
            pool.append((seed, env.grid.copy()))
        seed += 1
    return pool


def random_start(grid, rng):
    empty = np.argwhere(grid == EMPTY)
    return empty[rng.integers(len(empty))]


def generate_training(env, pool, task_id, dataset, path):
    tmp = path.with_suffix(".h5.tmp")
    with h5py.File(tmp, "w") as f:
        for i, kind in enumerate(kinds(dataset)):
            lay = i % NUM_LAYOUTS
            seed, grid = pool[lay]
            rng = traj_rng(task_id, dataset, i)
            while True:
                env.reset(grid=grid, start_pos=random_start(grid, rng))
                traj = run_episode(env, kind, rng)
                if traj is not None:
                    break
            # a 64 bit: con distance i reward valgono es. 100/3
            rewards = np.array(traj["rewards"], dtype=np.float64)
            g = f.create_group(str(i))
            g.attrs["seed"] = seed
            g.attrs["layout"] = lay
            g.attrs["length"] = len(traj["actions"])
            g.attrs["kind"] = kind
            g.create_dataset("grid", data=env.grid.astype(np.int8))
            g.create_dataset("positions", data=np.array(traj["positions"], dtype=np.int8))
            g.create_dataset("actions", data=np.array(traj["actions"], dtype=np.int8))
            g.create_dataset("rewards", data=rewards)
            g.create_dataset("terminals", data=np.array(traj["terminals"], dtype=bool))
            g.create_dataset("returns_to_go", data=np.cumsum(rewards[::-1])[::-1])
    os.replace(tmp, path)


def generate_test(env, pool, task_id, train_path, test_path):
    with h5py.File(train_path, "r") as f:
        starts = {(int(f[k].attrs["layout"]), tuple(int(v) for v in f[k]["positions"][0])) for k in f}
    rng = np.random.default_rng([LAYOUT_SEED_BASE, task_id, 10 ** 6])
    rows = []
    for j in range(NUM_TEST):
        lay = j % NUM_LAYOUTS
        grid = pool[lay][1]
        while True:
            start = random_start(grid, rng)
            env.reset(grid=grid, start_pos=start)
            if shortest_path(env, np.random.default_rng(0)) is not None:
                break
        rows.append((lay, grid, start, (lay, tuple(int(v) for v in start)) in starts))
    tmp = test_path.with_suffix(".tmp.npz")
    np.savez(tmp, layout=np.array([r[0] for r in rows]), grid=np.stack([r[1] for r in rows]),
             start=np.stack([r[2] for r in rows]), seen=np.array([r[3] for r in rows]))
    os.replace(tmp, test_path)


def generate(reward_type, task_id, dataset):
    folder = dataset_folder(reward_type, task_id)
    path, test_path = folder / f"dataset_{dataset}.h5", folder / f"test_{dataset}.npz"
    env = GridWorldEnv(formulas[task_id], state_type="symbolic", reward_type=reward_type)
    pool = layout_pool(env)
    if not path.exists():
        generate_training(env, pool, task_id, dataset, path)
        print(f"  salvato {path}", flush=True)
    if not test_path.exists():
        generate_test(env, pool, task_id, path, test_path)
        print(f"  salvato {test_path}", flush=True)


if __name__ == "__main__":
    for reward_type in REWARD_TYPES:
        for task in TASKS:
            print(f"\n{'=' * 60}\n {reward_type} / task{task}\n{'=' * 60}")
            for dataset in DATASET_NAMES:
                generate(reward_type, task - 1, dataset)
