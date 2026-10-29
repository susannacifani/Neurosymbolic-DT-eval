"""Verifica completa dei dataset .h5 generati da dataset_generation.py.

Per ogni file e ogni traiettoria controlla:
- numero di traiettorie (2016) e coerenza delle lunghezze tra i campi
- che ogni osservazione corrisponda a una casella della griglia (per image: l'immagine esatta
  prodotta dall'ambiente)
- transizioni: posizione(t) + azione(t) -> posizione(t+1)
- reward di ogni singolo passo, ricalcolato simulando l'automa della formula
- flag 'terminals' e fine dell'episodio: nessun passo dopo una vittoria o un fallimento
- returns_to_go = somma dei reward futuri
- etichetta di vittoria coerente con lo stato finale dell'automa
- 'final_observation' presente e uguale alla casella in cui porta l'ultima azione
- dfa_states coerente con l'automa a meno della numerazione degli stati
  (la libreria che costruisce l'automa numera gli stati in modo diverso a ogni esecuzione)
Inoltre verifica che symbolic e image abbiano le stesse identiche traiettorie.

Uso (dalla cartella NRM):
    python3 Dataset/check_datasets.py [state_types] [reward_types] [tasks]
    es. python3 Dataset/check_datasets.py symbolic,image distance,three_value 1,2,3
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
# motore comune (ambiente, automa) in NRM/NeuralRewardMachines
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "NeuralRewardMachines"))

import h5py
import numpy as np
from LTL_tasks import formulas
from RL.Env.Environment import GridWorldEnv

# i dataset stanno accanto a questo script (Dataset/<State>/<reward>/task<N>/)
ROOT = Path(os.environ.get("DATASET_ROOT", Path(__file__).resolve().parent))
STATES = sys.argv[1].split(",") if len(sys.argv) > 1 else ["symbolic", "image"]
REWARDS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["distance", "three_value"]
TASKS = [int(t) for t in sys.argv[3].split(",")] if len(sys.argv) > 3 else list(range(1, 11))
DATASETS = ["medium_expert", "medium", "medium_replay"]
EXPECTED_N = 2016

MOVES = {0: (0, 1), 1: (1, 0), 2: (0, -1), 3: (-1, 0)}
OBJECTS = {(3, 0): 2, (1, 1): 0, (0, 3): 3, (3, 3): 1}   # simbolo dell'oggetto in ogni casella (come _current_symbol)
EMPTY = 4

ERRORS = ["traj_count", "length", "unknown_obs", "transition", "reward_step", "terminal_flag",
          "steps_after_end", "rtg", "dfa_mapping", "win_label", "final_obs_missing", "final_obs_wrong"]


def check_file(path, env, pos_of):
    aut = env.automaton
    err = dict.fromkeys(ERRORS, 0)
    dfa_map = {}
    trajs, wins = [], 0
    with h5py.File(path, "r") as f:
        keys = sorted(f.keys(), key=int)
        if len(keys) != EXPECTED_N:
            err["traj_count"] = len(keys)
        for k in keys:
            g = f[k]
            obs, act, rew = g["observations"][:], g["actions"][:], g["rewards"][:]
            term, rtg = g["terminals"][:], g["returns_to_go"][:]
            dfa = g["dfa_states"][:].argmax(axis=1)
            L = len(act)
            if not (g.attrs["length"] == L == len(obs) == len(rew) == len(term) == len(rtg) == len(dfa)):
                err["length"] += 1
            pos = [pos_of(o) for o in obs]
            if any(p is None for p in pos):
                err["unknown_obs"] += 1
                continue
            q, ended, nxt = 0, False, pos[0]
            for i in range(L):
                if ended:
                    err["steps_after_end"] += 1
                    break
                if dfa_map.setdefault(int(dfa[i]), q) != q:
                    err["dfa_mapping"] += 1
                nxt = tuple(int(v) for v in np.clip(np.array(pos[i]) + np.array(MOVES[int(act[i])]), 0, 3))
                if i + 1 < L and nxt != pos[i + 1]:
                    err["transition"] += 1
                q_new = aut.transitions[q][OBJECTS.get(nxt, EMPTY)]
                expected = 0 if q_new == q else aut.rewards[q_new] - aut.rewards[q]
                if abs(float(rew[i]) - expected) > 1e-6:
                    err["reward_step"] += 1
                q = q_new
                ended = aut.rewards[q] in (100, -100)
                if bool(term[i]) != ended:
                    err["terminal_flag"] += 1
            if "final_observation" not in g:
                err["final_obs_missing"] += 1
            elif pos_of(g["final_observation"][:]) != nxt:
                err["final_obs_wrong"] += 1
            if not np.allclose(rtg, np.cumsum(rew[::-1])[::-1]):
                err["rtg"] += 1
            won = bool(aut.acceptance[q])
            wins += won
            if env.reward_type == "three_value":
                labeled = bool((rew == 100).any())
            else:
                labeled = abs(float(rew.sum()) - 100) < 1e-6
            if won != labeled:
                err["win_label"] += 1
            trajs.append((tuple(pos), tuple(int(a) for a in act)))
    return err, wins, trajs


def main():
    all_ok = True
    traj_by_key = {}
    for state in STATES:
        for rt in REWARDS:
            for t in TASKS:
                env = GridWorldEnv(formulas[t - 1], "rgb_array", state_type=state, train=False, reward_type=rt)
                if state == "symbolic":
                    pos_of = lambda o: tuple(int(v) for v in o)
                else:
                    canon = {np.asarray(img, dtype=np.float32).tobytes(): (x, y)
                             for (x, y), img in env.image_locations.items()}
                    pos_of = lambda o, c=canon: c.get(np.asarray(o, dtype=np.float32).tobytes())
                for ds in DATASETS:
                    path = ROOT / state.capitalize() / rt / f"task{t}" / f"dataset_{ds}.h5"
                    if not path.exists():
                        print(f"MANCA {path}")
                        all_ok = False
                        continue
                    err, wins, trajs = check_file(path, env, pos_of)
                    traj_by_key[(state, rt, t, ds)] = trajs
                    bad = {k: v for k, v in err.items() if v}
                    all_ok &= not bad
                    print(f"{state:<8} {rt:<11} task{t:<2} {ds:<13} vinte={wins:4d}/{len(trajs)}  "
                          f"{'OK' if not bad else f'ERRORI {bad}'}", flush=True)
    for (state, rt, t, ds), trajs in traj_by_key.items():
        if state == "image" and ("symbolic", rt, t, ds) in traj_by_key:
            same = trajs == traj_by_key[("symbolic", rt, t, ds)]
            all_ok &= same
            print(f"symbolic vs image {rt} task{t} {ds}: traiettorie {'IDENTICHE' if same else 'DIVERSE'}")
    print("\nRISULTATO:", "TUTTO OK" if all_ok else "CI SONO ERRORI")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
