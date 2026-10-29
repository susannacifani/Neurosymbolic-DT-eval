"""Dataset offline per la griglia 7x7 con oggetti in posizioni casuali.

Ogni traiettoria ha il suo seed, che decide la disposizione degli oggetti e la partenza
(casuale tra le caselle vuote): le disposizioni sono tutte diverse e non si possono enumerare,
quindi se ne campiona un numero fissato (NUM_TRAJECTORIES).

Tipi di dataset (come nella 4x4):
  medium_expert: 50% percorsi ottimi + 50% percorsi ottimi con rumore
  medium:        100% percorsi ottimi con rumore
  medium_replay: 50% percorsi ottimi con rumore + 50% passeggiate casuali
Percorso ottimo: planner.shortest_path (BFS su casella x stato dell'automa); tra i percorsi minimi
ne sceglie uno a caso. Rumore (policy ottima epsilon-greedy): a ogni passo, con probabilita' EPSILON,
una mossa casuale; dopo ogni mossa casuale si ripianifica un percorso minimo dalla situazione in cui
ci si trova (casella e stato dell'automa). EPSILON = 0.75 mette la policy con rumore a meta' tra
quella casuale e quella ottima (circa il 50% del CDR normalizzato), cosi' medium sta in mezzo tra
medium_expert e medium_replay. Ogni traiettoria si ferma appena l'episodio finisce (vittoria,
fallimento o limite di passi).
Le disposizioni senza soluzione (raro: gli oggetti da evitare bloccano ogni strada) si scartano.

Cartelle: Dataset/<reward>/task<N>/dataset_<dataset>.h5. Non c'e' distinzione symbolic / image:
il file non contiene osservazioni, quindi lo stesso file serve per entrambi.
Formato: un gruppo per traiettoria con
  attrs:  seed, length (T), kind (optimal / noisy / random)
  grid       (7, 7)   disposizione: simbolo di ogni casella (0 P, 1 L, 2 D, 3 G, 4 vuota)
  positions  (T+1, 2) posizione dell'agente prima di ogni azione + posizione finale
  actions, rewards, terminals, returns_to_go  (T,); reward e return-to-go a 64 bit, cosi' restano
  esatti (con distance valgono es. 100/3) anche nelle somme
Non si salvano le osservazioni: mappa (symbolic) o immagine (image) si ricostruiscono da
grid e positions, quindi lo stesso file serve per entrambi.
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
from RL.Env.Environment import GridWorldEnv
from planner import shortest_path

EPSILON = 0.75              # probabilita' di una mossa casuale a ogni passo (traiettorie con rumore)
NUM_TRAJECTORIES = 5000     # traiettorie (= disposizioni) per dataset
DATASET_NAMES = ["medium_expert", "medium", "medium_replay"]


def base_seed(task_id, dataset_idx):
    """seed della prima traiettoria: blocchi separati per task e dataset. La valutazione usera'
    seed >= 1_000_000, quindi disposizioni mai viste in training."""
    return 100_000 * task_id + 10_000 * dataset_idx


def run_episode(env, kind, rng):
    """Una traiettoria di tipo kind dall'ambiente gia' resettato. Restituisce un dict con le liste
    (positions ha un elemento in piu': la posizione finale); None se non c'e' soluzione."""
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

    # con rumore: mossa casuale con probabilita' EPSILON, poi si ripianifica da dove si e' arrivati
    # (se non c'e' piu' un percorso senza fallimento, path vuoto: si continua con mosse casuali)
    while True:
        if rng.random() < EPSILON or not path:
            if do(int(rng.integers(4))):
                return traj
            path = shortest_path(env, rng) or []
        elif do(path.pop(0)):
            return traj


def generate_dataset(reward_type, task_id, dataset_idx, path):
    name = DATASET_NAMES[dataset_idx]
    env = GridWorldEnv(formulas[task_id], state_type="symbolic", reward_type=reward_type)
    if name == "medium_expert":
        kinds = ["optimal"] * (NUM_TRAJECTORIES // 2) + ["noisy"] * (NUM_TRAJECTORIES - NUM_TRAJECTORIES // 2)
    elif name == "medium":
        kinds = ["noisy"] * NUM_TRAJECTORIES
    else:
        kinds = ["noisy"] * (NUM_TRAJECTORIES // 2) + ["random"] * (NUM_TRAJECTORIES - NUM_TRAJECTORIES // 2)

    seed, skipped = base_seed(task_id, dataset_idx), 0
    with h5py.File(path, "w") as f:
        for i, kind in enumerate(kinds):
            while True:
                env.reset(seed=seed)
                # stream casuale separato per percorso e rumore, sempre riproducibile dal seed
                traj = run_episode(env, kind, np.random.default_rng([seed, 1]))
                seed += 1
                if traj is not None:
                    break
                skipped += 1   # disposizione senza soluzione: si passa al seed successivo
            # reward a 64 bit come nella 4x4: con distance valgono es. 100/3, che a 32 bit si arrotonda
            rewards = np.array(traj["rewards"], dtype=np.float64)
            g = f.create_group(str(i))
            g.attrs["seed"] = seed - 1
            g.attrs["length"] = len(traj["actions"])
            g.attrs["kind"] = kind
            g.create_dataset("grid", data=env.grid.astype(np.int8))
            g.create_dataset("positions", data=np.array(traj["positions"], dtype=np.int8))
            g.create_dataset("actions", data=np.array(traj["actions"], dtype=np.int8))
            g.create_dataset("rewards", data=rewards)
            g.create_dataset("terminals", data=np.array(traj["terminals"], dtype=bool))
            g.create_dataset("returns_to_go", data=np.cumsum(rewards[::-1])[::-1])
            if (i + 1) % 1000 == 0:
                print(f"  {name}: {i + 1}/{len(kinds)} traiettorie", flush=True)
    return skipped


def dataset_folder(reward_type, task_id):
    """Dataset/<reward>/task<N>/ (contiene i 3 file .h5, validi sia per symbolic sia per image)"""
    folder = DATASET_ROOT / reward_type / f"task{task_id + 1}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def generate_datasets(reward_type, task_id):
    for dataset_idx, name in enumerate(DATASET_NAMES):
        h5_path = dataset_folder(reward_type, task_id) / f"dataset_{name}.h5"
        if not h5_path.exists():
            # si scrive su un file temporaneo e lo si rinomina solo a fine generazione,
            # cosi' un'interruzione non lascia un file parziale che verrebbe poi saltato
            tmp_path = h5_path.with_suffix(".h5.tmp")
            skipped = generate_dataset(reward_type, task_id, dataset_idx, tmp_path)
            os.replace(tmp_path, h5_path)
            print(f"  {name}: salvato {h5_path} (disposizioni senza soluzione scartate: {skipped})", flush=True)


if __name__ == "__main__":
    # =====================================================================
    # CONFIGURAZIONE: modifica qui cosa generare, poi premi Run
    # (i file gia' presenti vengono saltati)
    # =====================================================================
    REWARD_TYPES = ["distance", "three_value"]
    TASKS = range(1, 11)
    # =====================================================================
    for reward_type in REWARD_TYPES:
        for task in TASKS:
            print(f"\n{'=' * 60}\n {reward_type} / task{task}\n{'=' * 60}")
            generate_datasets(reward_type, task - 1)
