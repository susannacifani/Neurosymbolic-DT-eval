"""Verifiche del Decision Transformer e del collegamento con automa e grounder (sui modelli gia' allenati,
non modifica nulla). Se una verifica fallisce lo script si ferma con un errore.

1. valutazione = training: gli input costruiti passo per passo come in evaluate() (storia fino al
   passo t, segnaposto al posto dell'azione corrente, embedding delle osservazioni riusati) danno gli
   stessi logit del calcolo con la finestra intera e il padding (traiettorie che stanno nel contesto)
2. nessuna fuga dal futuro in training: senza dropout, cambiare tutte le azioni non cambia la
   previsione del primo passo, e training ed eval coincidono
3. return-to-go: quello aggiornato in valutazione (togliendo i reward) = quello dei file
4. tracker dell'automa con i simboli veri = stato dell'automa dell'ambiente, a ogni passo (10 task)
5. grounder: per ogni oggetto, il simbolo che gli assegna (informativo: le scorciatoie si vedono qui
   e nella symbol accuracy di Grounding/print_grounding_table.py)
Uso (dalla cartella NRM): python3 Grid7/DT/check_dt.py
"""
import collections
import random

import h5py
import numpy as np
import torch

import decision_transformer as dt
from grounder import grounder_path, load_grounder, predict_symbols, true_symbols

TOL = 1e-5
CASES = [("Symbolic", None), ("automaton_state/true/Symbolic", "true"),
         ("automaton_state/grounder/Symbolic", "grounder")]


def val_keys(h5, seed=0):
    with h5py.File(h5) as f:
        keys = sorted(f.keys(), key=int)
    random.Random(seed).shuffle(keys)
    return keys[int(0.8 * len(keys)):]


def load(path):
    ck = torch.load(path, map_location=dt.device)
    m = dt.DecisionTransformer("symbolic", ck["config"].get("num_automaton_states", 0)).to(dt.device)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m, ck["config"]


def check_eval_inputs(folder, automaton, task=7, reward="three_value", dataset="medium", n=40):
    path = dt.ROOT / "DT" / folder / reward / f"task{task}" / "seed0" / "dt_model" / f"dt_{dataset}.pth"
    if not path.exists():
        print(f"   (manca {path}, saltato)")
        return
    m, cfg = load(path)
    K = cfg["context_len"]
    grounder = grounder_path("symbolic", reward, task, dataset, 0) if automaton == "grounder" else None
    tracker = dt.make_tracker(automaton, "symbolic", reward, task, grounder)
    h5 = dt.dataset_path(reward, task, dataset)
    with h5py.File(h5) as f:
        keys = [k for k in val_keys(h5) if len(f[k]["actions"]) <= K][:n]
    ds = dt.DTDataset(h5, keys, K, tracker=tracker)
    worst = rtg_err = 0.0
    with h5py.File(h5) as f, torch.no_grad():
        for i, k in enumerate(keys):
            item = ds[i]
            T = int(item["mask"].sum())
            b = {kk: v[None].to(dt.device) for kk, v in item.items()}
            full = m(b["states"], b["actions"], b["rtgs"], b["timesteps"], b.get("automaton"), b["mask"])[0, :T]
            rew = f[k]["rewards"][:]
            rtg = [float(f[k]["returns_to_go"][0]) / dt.RTG_SCALE]
            for t in range(1, T):
                rtg.append(rtg[-1] - rew[t - 1] / dt.RTG_SCALE)
            rtg_err = max(rtg_err, np.abs(np.array(rtg) - f[k]["returns_to_go"][:T] / dt.RTG_SCALE).max())
            codes = torch.from_numpy(ds.cache[k]["codes"][:T].astype(np.int64)).to(dt.device)
            embs = [m.embed_states(codes[t:t + 1][:, None])[:, 0] for t in range(T)]
            for t in range(T):
                acts = torch.tensor([list(f[k]["actions"][:t]) + [0]], device=dt.device)
                r = torch.tensor([rtg[:t + 1]], dtype=torch.float32, device=dt.device)
                ts = torch.arange(0, t + 1, device=dt.device)[None]
                aut = None
                if tracker:
                    aut = torch.nn.functional.one_hot(torch.from_numpy(ds.cache[k]["automaton"][:t + 1]).to(dt.device),
                                                      tracker.num_states).float()[None]
                lg = m(None, acts, r, ts, aut, state_emb=torch.stack(embs[:t + 1], 1))[0, -1]
                worst = max(worst, (lg - full[t]).abs().max().item())
    assert worst < TOL, f"{folder}: logit di valutazione diversi dal training ({worst})"
    assert rtg_err < 1e-9, f"{folder}: return-to-go diverso dai file ({rtg_err})"
    print(f"OK 1,3 {folder}: logit valutazione = training (diff {worst:.1e}), return-to-go = file")


def check_no_leak(task=7, reward="three_value", dataset="medium"):
    m, cfg = load(dt.ROOT / "DT" / "Symbolic" / reward / f"task{task}" / "seed0" / "dt_model" / f"dt_{dataset}.pth")
    h5 = dt.dataset_path(reward, task, dataset)
    keys = val_keys(h5)[:64]
    ds = dt.DTDataset(h5, keys, cfg["context_len"])
    b = {k: torch.stack([ds[i][k] for i in range(len(keys))]).to(dt.device) for k in ds[0]}
    m.train()
    for mod in m.modules():
        if isinstance(mod, torch.nn.Dropout):
            mod.p = 0.0
        if isinstance(mod, torch.nn.MultiheadAttention):
            mod.dropout = 0.0
    with torch.no_grad():
        base = m(b["states"], b["actions"], b["rtgs"], b["timesteps"], None, b["mask"])
        alt = m(b["states"], (b["actions"] + 1) % dt.NUM_ACTIONS, b["rtgs"], b["timesteps"], None, b["mask"])
        m.eval()
        ev = m(b["states"], b["actions"], b["rtgs"], b["timesteps"], None, b["mask"])
    leak, mode = (base[:, 0] - alt[:, 0]).abs().max().item(), (base - ev)[b["mask"]].abs().max().item()
    assert leak < TOL and mode < TOL, f"fuga dal futuro o training diverso da eval: {leak}, {mode}"
    print(f"OK 2 nessuna fuga dal futuro (diff {leak:.1e}), training senza dropout = eval (diff {mode:.1e})")


def check_tracker(episodes=300):
    for task in range(1, 11):
        env = dt.GridWorldEnv(dt.formulas[task - 1], state_type="symbolic", reward_type="three_value")
        tr = dt.make_tracker("true", "symbolic", "three_value", task)
        rng = np.random.default_rng(task)
        bad = 0
        for _ in range(episodes):
            env.reset(seed=int(rng.integers(1e9)))
            q = tr.step(0, true_symbols(dt.cell_codes(env.grid, env._agent_location[None]))[0])
            bad += int(q != env.curr_automaton_state)
            for _ in range(100):
                _, _, done, trunc, _ = env.step(int(rng.integers(dt.NUM_ACTIONS)))
                q = tr.step(q, true_symbols(dt.cell_codes(env.grid, env._agent_location[None]))[0])
                bad += int(q != env.curr_automaton_state)
                if done or trunc:
                    break
        assert bad == 0, f"task{task}: il tracker non segue l'automa dell'ambiente ({bad} passi)"
    print("OK 4 tracker dell'automa = automa dell'ambiente (10 task)")


def show_grounders(seed=0, dataset="medium"):
    names = ["P", "L", "D", "G", "vuota"]
    for rt in ["three_value", "distance"]:
        for task in range(1, 11):
            g = load_grounder(grounder_path("symbolic", rt, task, dataset, seed), "symbolic")
            h5 = dt.dataset_path(rt, task, dataset)
            with h5py.File(h5) as f:
                codes = np.concatenate([dt.cell_codes(f[k]["grid"][:], f[k]["positions"][:].astype(np.int64))
                                        for k in val_keys(h5, seed)[500:]])
            st, sg = true_symbols(codes), predict_symbols(g, codes)
            conf = collections.Counter(zip(st.tolist(), sg.tolist()))
            reads = ", ".join(f"{names[s]}->{names[max(range(5), key=lambda p: conf[(s, p)])]}" for s in range(5))
            print(f"   5 grounder {rt:<11} task{task:<2} seed{seed} {dataset}: {reads}")


def main():
    for folder, automaton in CASES:
        check_eval_inputs(folder, automaton)
    check_no_leak()
    check_tracker()
    show_grounders()
    print("TUTTE LE VERIFICHE PASSATE (la 5 e' informativa)")


if __name__ == "__main__":
    main()
