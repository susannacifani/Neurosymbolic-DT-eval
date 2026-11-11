"""Verifica dei dataset 7x7 generati da dataset_generation.py.

Per ogni file di training e ogni traiettoria:
- numero di traiettorie e coerenza delle lunghezze tra i campi
- disposizione valida e partenza su una casella vuota
- transizioni, reward ricalcolati con l'automa, terminals, returns_to_go
- fine solo per vittoria, fallimento o limite di passi, nessun passo dopo la fine
- percorsi "optimal": lunghezza minima e vittoria
- osservazioni ricostruite da grid e positions uguali a quelle dell'ambiente (su un campione)
Disposizioni: training e test usano tutte e sole le NUM_LAYOUTS disposizioni del pool, ognuna lo stesso
numero di volte; partenze di test risolvibili; episodi di test uguali per i 3 dataset.
Inoltre: stesse traiettorie per distance e three_value.

Uso (dalla cartella NRM): python3 Grid7/Dataset/check_datasets.py [reward_types] [tasks]
"""
import sys
from collections import Counter, deque
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "NeuralRewardMachines"))
sys.path.insert(0, str(ROOT))

from LTL_tasks import formulas
from RL.Env.Environment import GridWorldEnv, symbolic_observation, OBJECT_TYPES, EMPTY
from dataset_generation import NUM_LAYOUTS, NUM_TRAJECTORIES, layout_pool
from planner import shortest_path

REWARDS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["distance", "three_value"]
TASKS = [int(t) for t in sys.argv[2].split(",")] if len(sys.argv) > 2 else list(range(1, 11))
DATASETS = ["medium_expert", "medium", "medium_replay"]
OBS_SAMPLE = 200
ERRORS = ["traj_count", "dtype", "length", "layout", "start", "transition", "reward_step", "terminal_flag",
          "steps_after_end", "bad_ending", "rtg", "not_optimal", "symbolic_obs", "image_obs"]


def min_length(env):
    """distanza minima dalla vittoria (BFS indipendente dal planner)"""
    aut = env.automaton
    start = (tuple(int(v) for v in env._agent_location), env.curr_automaton_state)
    seen, frontier = {start}, deque([(start, 0)])
    while frontier:
        (pos, q), d = frontier.popleft()
        for a in range(4):
            nxt = tuple(int(v) for v in np.clip(np.array(pos) + env._action_to_direction[a], 0, env.size - 1))
            q2 = aut.transitions[q][int(env.grid[nxt])]
            if aut.rewards[q2] == -100 or (nxt, q2) in seen:
                continue
            if aut.acceptance[q2]:
                return d + 1
            seen.add((nxt, q2))
            frontier.append(((nxt, q2), d + 1))
    return None


def check_file(path, env, env_img):
    aut = env.automaton
    err = dict.fromkeys(ERRORS, 0)
    stats = {"kinds": Counter(), "outcomes": Counter(), "lengths": [], "traj": []}
    with h5py.File(path, "r") as f:
        keys = sorted(f.keys(), key=int)
        if len(keys) != NUM_TRAJECTORIES:
            err["traj_count"] = len(keys)
        for n, k in enumerate(keys):
            g = f[k]
            grid, pos = g["grid"][:].astype(np.int64), g["positions"][:].astype(np.int64)
            act, rew, term, rtg = g["actions"][:], g["rewards"][:], g["terminals"][:], g["returns_to_go"][:]
            if rew.dtype != np.float64 or rtg.dtype != np.float64:
                err["dtype"] += 1
            kind, T = g.attrs["kind"], len(act)
            stats["kinds"][kind] += 1
            stats["lengths"].append(T)
            stats["traj"].append((grid.tobytes(), pos.tobytes(), act.tobytes()))
            if not (g.attrs["length"] == T == len(rew) == len(term) == len(rtg) == len(pos) - 1):
                err["length"] += 1
                continue
            if not (all((grid == o).sum() == env.copies for o in OBJECT_TYPES) and (grid != EMPTY).sum() == env.copies * 4):
                err["layout"] += 1
            if grid[tuple(pos[0])] != EMPTY:
                err["start"] += 1
            q, ended = 0, False
            for t in range(T):
                if ended:
                    err["steps_after_end"] += 1
                    break
                nxt = np.clip(pos[t] + env._action_to_direction[int(act[t])], 0, env.size - 1)
                if not np.array_equal(nxt, pos[t + 1]):
                    err["transition"] += 1
                q_new = aut.transitions[q][int(grid[tuple(pos[t + 1])])]
                expected = 0 if q_new == q else aut.rewards[q_new] - aut.rewards[q]
                if abs(float(rew[t]) - expected) > 1e-6:
                    err["reward_step"] += 1
                q = q_new
                done = aut.rewards[q] in (100, -100)
                if bool(term[t]) != done:
                    err["terminal_flag"] += 1
                ended = done or (t + 1) >= env.max_num_steps
            outcome = "vittoria" if aut.acceptance[q] else "fallimento" if aut.rewards[q] == -100 else "limite di passi"
            stats["outcomes"][(kind, outcome)] += 1
            if not ended or (kind == "optimal" and outcome == "limite di passi"):
                err["bad_ending"] += 1
            if not np.allclose(rtg, np.cumsum(rew[::-1])[::-1]):
                err["rtg"] += 1
            if kind == "optimal":
                env.reset(grid=grid, start_pos=pos[0])
                if outcome != "vittoria" or min_length(env) != T:
                    err["not_optimal"] += 1
            if n < OBS_SAMPLE:
                o_sym, _, _ = env.reset(grid=grid, start_pos=pos[0])
                o_img, _, _ = env_img.reset(grid=grid, start_pos=pos[0])
                for t in range(T + 1):
                    if not np.array_equal(o_sym, symbolic_observation(grid, pos[t])):
                        err["symbolic_obs"] += 1
                        break
                    if not np.array_equal(o_img, env_img.renderer.render(grid, pos[t])):
                        err["image_obs"] += 1
                        break
                    if t < T:
                        o_sym = env.step(int(act[t]))[0]
                        o_img = env_img.step(int(act[t]))[0]
    return err, stats


def check_layouts(path, test_path, env, pool):
    """controlli sulle disposizioni del training e del test rispetto al pool"""
    grids = {g.tobytes(): i for i, (_, g) in enumerate(pool)}
    with h5py.File(path, "r") as f:
        keys = sorted(f.keys(), key=int)
        lay = [grids.get(f[k]["grid"][:].astype(np.int64).tobytes()) for k in keys]
        attr = [int(f[k].attrs["layout"]) for k in keys]
    test = np.load(test_path)
    solvable = True
    for g, s in zip(test["grid"], test["start"]):
        env.reset(grid=g, start_pos=s)
        solvable &= shortest_path(env, np.random.default_rng(0)) is not None
    counts = np.bincount([l for l in lay if l is not None], minlength=NUM_LAYOUTS)
    return {"training: disposizioni del pool": None not in lay and lay == attr,
            f"training: ogni disposizione {NUM_TRAJECTORIES // NUM_LAYOUTS} volte":
                bool((counts == NUM_TRAJECTORIES // NUM_LAYOUTS).all()),
            "test: disposizioni del pool": all(grids.get(g.astype(np.int64).tobytes()) == l
                                               for g, l in zip(test["grid"], test["layout"])),
            "test: partenze risolvibili": bool(solvable)}


def main():
    all_ok = True
    by_key = {}
    for rt in REWARDS:
        for t in TASKS:
            env = GridWorldEnv(formulas[t - 1], state_type="symbolic", reward_type=rt)
            env_img = GridWorldEnv(formulas[t - 1], state_type="image", reward_type=rt)
            pool = layout_pool(env)
            tests = []
            for ds in DATASETS:
                path = ROOT / rt / f"task{t}" / f"dataset_{ds}.h5"
                test_path = ROOT / rt / f"task{t}" / f"test_{ds}.npz"
                if not path.exists() or not test_path.exists():
                    print(f"MANCA {path if not path.exists() else test_path}")
                    all_ok = False
                    continue
                err, st = check_file(path, env, env_img)
                layouts = check_layouts(path, test_path, env, pool)
                by_key[(rt, t, ds)] = st["traj"]
                test = np.load(test_path)
                tests.append((test["layout"], test["grid"], test["start"]))
                bad = {k: v for k, v in err.items() if v}
                bad_layouts = [k for k, v in layouts.items() if not v]
                all_ok &= not bad and not bad_layouts
                outcomes = ", ".join(f"{kind} {o}: {c}" for (kind, o), c in sorted(st["outcomes"].items()))
                print(f"{rt:<11} task{t:<2} {ds:<13} {'OK' if not bad else f'ERRORI {bad}'}"
                      f"{'' if not bad_layouts else f' | DISPOSIZIONI: {bad_layouts}'}\n"
                      f"    tipi: {dict(st['kinds'])} | lunghezza media {np.mean(st['lengths']):.1f} (max {max(st['lengths'])})\n"
                      f"    esiti: {outcomes}\n"
                      f"    test: {len(test['seen'])} episodi, partenza vista in training: {100 * test['seen'].mean():.1f}%",
                      flush=True)
            same = len(tests) == len(DATASETS) and all(
                all(np.array_equal(a, b) for a, b in zip(tests[0], x)) for x in tests[1:])
            all_ok &= same
            print(f"{rt:<11} task{t:<2} episodi di test uguali per i {len(DATASETS)} dataset: {same}")
    for (rt, t, ds), trajs in by_key.items():
        if rt == "three_value" and ("distance", t, ds) in by_key:
            same = trajs == by_key[("distance", t, ds)]
            all_ok &= same
            print(f"distance vs three_value task{t} {ds}: traiettorie {'IDENTICHE' if same else 'DIVERSE'}")
    print("\nRISULTATO:", "TUTTO OK" if all_ok else "CI SONO ERRORI")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
