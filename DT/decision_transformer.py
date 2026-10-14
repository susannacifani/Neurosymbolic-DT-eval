"""Decision Transformer: training e valutazione per tutte le configurazioni
(symbolic / image x distance / three_value).

Dataset: Dataset/<State>/<reward_type>/task<N>/dataset_<dataset>.h5
Output:  DT/<State>/<reward_type>/task<N>/seed<S>/
         dt_model/dt_<dataset>.pth            modello + configurazione con cui e' stato allenato
         log/train_log_<dataset>.txt          iperparametri, loss/accuracy per epoca, tempi
         eval/eval_<dataset>.csv              un episodio per riga
         eval/eval_summary_<dataset>.txt      riepilogo della valutazione
"""
import os
import sys
import random
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

# cartella NRM (Dataset/ e DT/) e motore comune in NRM/NeuralRewardMachines
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "NeuralRewardMachines"))

from LTL_tasks import formulas
from RL.Env.Environment import GridWorldEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_ACTIONS = 4
MAX_EP_LEN = 100       # gli episodi durano al massimo 50 passi
RTG_SCALE = 100.0
POSSIBLE_STARTS = [(0, 0), (1, 0), (2, 0), (0, 1), (2, 1), (3, 1),
                   (0, 2), (1, 2), (2, 2), (3, 2), (1, 3), (2, 3)]

# iperparametri
CONTEXT_LEN = 50       # passi di storia visti dal modello: l'intero episodio (max 50 passi)
FIXED_VAL = True       # validation su blocchi fissi (ogni passo una volta) invece che una finestra casuale per traiettoria
WEIGHT_DECAY = 1e-4
BATCH_SIZE = {"image": 32, "symbolic": 64}
LR = 1e-4
HIDDEN_SIZE = 128
N_LAYERS = 3
N_HEADS = 4
DROPOUT = 0.1
MAX_EPOCHS = 500
PATIENCE = 10


def dataset_path(state_type, reward_type, task, dataset):
    return ROOT / "Dataset" / state_type.capitalize() / reward_type / f"task{task}" / f"dataset_{dataset}.h5"


def run_dir(state_type, reward_type, task, seed):
    return ROOT / "DT" / state_type.capitalize() / reward_type / f"task{task}" / f"seed{seed}"


def output_paths(state_type, reward_type, task, dataset, seed, out_dir=None):
    """Percorsi dei file di una run; le cartelle vengono create solo quando servono.
    out_dir sostituisce DT/<State>/<reward>/task<N>/seed<S> (utile per i test)."""
    base = Path(out_dir) if out_dir else run_dir(state_type, reward_type, task, seed)
    return {"model": base / "dt_model" / f"dt_{dataset}.pth",
            "log": base / "log" / f"train_log_{dataset}.txt",
            "csv": base / "eval" / f"eval_{dataset}.csv",
            "summary": base / "eval" / f"eval_summary_{dataset}.txt"}


def preprocess_state(obs, state_type):
    # image: gia' normalizzata dall'ambiente; symbolic: coordinate (x, y) in [0, 3] -> [0, 1]
    if state_type == "image":
        return np.asarray(obs, dtype=np.float32)
    return np.asarray(obs, dtype=np.float32) / 3.0


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class DTDataset(Dataset):
    """Finestre di `context_len` passi consecutivi.
    all_windows=False: una finestra casuale per traiettoria (training).
    all_windows=True: ogni traiettoria divisa in blocchi consecutivi non sovrapposti, sempre
    gli stessi; ogni passo viene valutato esattamente una volta (validation)."""

    def __init__(self, h5_path, keys, state_type, context_len, all_windows=False):
        self.h5_path = h5_path
        self.state_type = state_type
        self.K = context_len
        self._file = None
        with h5py.File(h5_path, "r") as f:
            lengths = {k: len(f[k]["actions"]) for k in keys}
            if state_type == "symbolic":
                # i dati symbolic sono piccoli: si caricano tutti in memoria
                self.cache = {k: {n: f[k][n][:] for n in ("observations", "actions", "returns_to_go")}
                              for k in keys}
        if all_windows:
            self.index = [(k, s) for k in keys for s in range(0, lengths[k], self.K)]
        else:
            self.index = [(k, None) for k in keys]

    def __len__(self):
        return len(self.index)

    def _traj(self, key):
        if self.state_type == "symbolic":
            return self.cache[key]
        # le immagini si leggono dal file solo quando servono (RAM limitata)
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")
        return self._file[key]

    def __getitem__(self, idx):
        key, start = self.index[idx]
        g = self._traj(key)
        traj_len = len(g["actions"])
        if start is None:
            start = np.random.randint(0, traj_len - self.K + 1) if traj_len > self.K else 0
        end = min(start + self.K, traj_len)

        states = preprocess_state(g["observations"][start:end], self.state_type)
        actions = np.asarray(g["actions"][start:end], dtype=np.int64)
        rtgs = np.asarray(g["returns_to_go"][start:end], dtype=np.float32) / RTG_SCALE
        timesteps = np.arange(start, end)
        n = end - start

        # padding a destra: con la maschera causale i passi reali non vedono mai il
        # padding (che sta dopo), quindi equivale a una finestra senza padding, come in
        # valutazione; il padding e' escluso dalla loss tramite la maschera
        pad = self.K - n
        if pad > 0:
            states = np.concatenate([states, np.zeros((pad, *states.shape[1:]), dtype=np.float32)])
            actions = np.pad(actions, (0, pad))
            rtgs = np.pad(rtgs, (0, pad))
            timesteps = np.pad(timesteps, (0, pad))
        mask = np.concatenate([np.ones(n, dtype=bool), np.zeros(pad, dtype=bool)])

        return {
            "states": torch.from_numpy(states),
            "actions": torch.from_numpy(actions),
            "rtgs": torch.from_numpy(rtgs),
            "timesteps": torch.from_numpy(timesteps).long(),
            "mask": torch.from_numpy(mask),
        }


# ---------------------------------------------------------------------------
# Modello
# ---------------------------------------------------------------------------
class DecisionTransformer(nn.Module):
    def __init__(self, state_type):
        super().__init__()
        h = HIDDEN_SIZE
        self.hidden_size = h
        if state_type == "image":
            self.embed_state = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=4, stride=2), nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2), nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=2), nn.ReLU(),
                nn.Flatten(), nn.Linear(3136, h),
            )
        else:
            self.embed_state = nn.Sequential(nn.Linear(2, h), nn.Tanh())
        self.embed_rtg = nn.Linear(1, h)
        self.embed_action = nn.Embedding(NUM_ACTIONS, h)
        self.embed_timestep = nn.Embedding(MAX_EP_LEN, h)

        layer = nn.TransformerEncoderLayer(d_model=h, nhead=N_HEADS, dim_feedforward=4 * h,
                                           dropout=DROPOUT, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=N_LAYERS)
        self.predict_action = nn.Linear(h, NUM_ACTIONS)

    def forward(self, states, actions, rtgs, timesteps):
        B, T = states.shape[:2]
        state_emb = self.embed_state(states.reshape(B * T, *states.shape[2:])).view(B, T, self.hidden_size)
        time_emb = self.embed_timestep(timesteps)
        state_emb = state_emb + time_emb
        action_emb = self.embed_action(actions) + time_emb
        rtg_emb = self.embed_rtg(rtgs.unsqueeze(-1)) + time_emb

        # ogni passo t diventa (R_t, s_t, a_t); l'azione a_t si predice dal token s_t,
        # che con la maschera causale vede R_t, s_t e tutto il passato, ma non a_t
        stacked = torch.stack((rtg_emb, state_emb, action_emb), dim=2).view(B, 3 * T, self.hidden_size)
        causal = torch.triu(torch.ones(3 * T, 3 * T, device=states.device), diagonal=1).bool()
        out = self.transformer(stacked, mask=causal)
        return self.predict_action(out[:, 1::3])


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def _run_epoch(model, loader, optimizer=None):
    """Loss media e accuracy delle azioni, pesate sui passi reali (padding escluso)."""
    loss_sum, correct, count = 0.0, 0, 0
    for batch in loader:
        b = {k: v.to(device) for k, v in batch.items()}
        logits = model(b["states"], b["actions"], b["rtgs"], b["timesteps"])[b["mask"]]
        target = b["actions"][b["mask"]]
        loss = nn.functional.cross_entropy(logits, target)
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_sum += loss.item() * len(target)
        correct += (logits.argmax(-1) == target).sum().item()
        count += len(target)
    return loss_sum / count, 100.0 * correct / count


def train(state_type, reward_type, task, dataset, seed=0, context_len=None, fixed_val=None,
          weight_decay=None, max_epochs=MAX_EPOCHS, out_dir=None):
    context_len = CONTEXT_LEN if context_len is None else context_len
    fixed_val = FIXED_VAL if fixed_val is None else fixed_val
    weight_decay = WEIGHT_DECAY if weight_decay is None else weight_decay
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    paths = output_paths(state_type, reward_type, task, dataset, seed, out_dir)
    for key in ("model", "log"):
        paths[key].parent.mkdir(parents=True, exist_ok=True)
    h5_path = dataset_path(state_type, reward_type, task, dataset)

    with h5py.File(h5_path, "r") as f:
        keys = sorted(f.keys(), key=int)
    random.Random(seed).shuffle(keys)
    split = int(0.8 * len(keys))
    bs = BATCH_SIZE[state_type]
    train_set = DTDataset(h5_path, keys[:split], state_type, context_len)
    val_set = DTDataset(h5_path, keys[split:], state_type, context_len, all_windows=fixed_val)
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=bs, shuffle=False)

    model = DecisionTransformer(state_type).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=weight_decay)
    config = {"state_type": state_type, "reward_type": reward_type, "task": task, "dataset": dataset,
              "seed": seed, "context_len": context_len, "fixed_val": fixed_val, "weight_decay": weight_decay,
              "batch_size": bs, "lr": LR, "hidden_size": HIDDEN_SIZE, "n_layers": N_LAYERS,
              "n_heads": N_HEADS, "dropout": DROPOUT, "max_epochs": max_epochs, "patience": PATIENCE}

    t_start = time.time()
    with open(paths["log"], "w") as log:
        def write(msg):
            print(msg)
            log.write(msg + "\n")
            log.flush()

        write(f"Dataset: {h5_path}")
        write("Config: " + ", ".join(f"{k}={v}" for k, v in config.items()))
        write(f"Device: {device} | parameters: {sum(p.numel() for p in model.parameters()):,}")
        write(f"Trajectories train: {split} | val: {len(keys) - split} | val windows: {len(val_set)}")
        best_val, best_acc, best_epoch, best_state, counter = float("inf"), 0.0, 0, None, 0
        for epoch in range(1, max_epochs + 1):
            t0 = time.time()
            model.train()
            train_loss, train_acc = _run_epoch(model, train_loader, optimizer)
            model.eval()
            with torch.no_grad():
                val_loss, val_acc = _run_epoch(model, val_loader)
            write(f"Epoch {epoch:03d} | Train loss: {train_loss:.4f} acc: {train_acc:5.1f}% | "
                  f"Val loss: {val_loss:.4f} acc: {val_acc:5.1f}% | {time.time() - t0:.1f}s")
            if val_loss < best_val:
                best_val, best_acc, best_epoch, counter = val_loss, val_acc, epoch, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                counter += 1
                if counter >= PATIENCE:
                    write(f"Early stopping at epoch {epoch}")
                    break
        write(f"Best epoch: {best_epoch} | val loss: {best_val:.4f} | val acc: {best_acc:.1f}%")
        write(f"Training time: {(time.time() - t_start) / 60:.1f} min")

    # il modello si scrive una sola volta, a training finito: se il training si
    # interrompe, il .pth non esiste e il launcher non lo scambia per completo
    model_path = paths["model"]
    tmp_path = model_path.with_suffix(".pth.tmp")
    torch.save({"state_dict": best_state, "config": config,
                "best_epoch": best_epoch, "best_val_loss": best_val, "best_val_acc": best_acc}, tmp_path)
    os.replace(tmp_path, model_path)
    return model_path


# ---------------------------------------------------------------------------
# Valutazione
# ---------------------------------------------------------------------------
def evaluate(state_type, reward_type, task, dataset, seed=0, num_runs=1, gamma=0.99,
             target_return=1.0, out_dir=None):
    # ambiente e policy sono deterministici: piu' run dalla stessa partenza danno lo
    # stesso episodio, quindi di default si fa 1 run per ciascuna delle 12 partenze
    paths = output_paths(state_type, reward_type, task, dataset, seed, out_dir)
    paths["csv"].parent.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(paths["model"], map_location=device)
    K = ckpt["config"]["context_len"]
    model = DecisionTransformer(state_type).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    env = GridWorldEnv(formulas[task - 1], "rgb_array", state_type=state_type, use_dfa_state=False,
                       train=False, reward_type=reward_type)
    rows = []
    t_start = time.time()
    for start_pos in POSSIBLE_STARTS:
        for run in range(num_runs):
            obs, _, _ = env.reset(start_pos=list(start_pos))
            # storia allineata come nel training: lo slot t contiene (R_t, s_t, a_t);
            # l'azione dello slot corrente e' un segnaposto (0), invisibile al token s_t
            # per via della maschera causale, e viene sostituita dall'azione scelta
            states = [preprocess_state(obs, state_type)]
            actions = [0]
            rtgs = [target_return]
            timesteps = [0]
            ep_reward, disc_reward, t = 0.0, 0.0, 0
            chosen = []

            while True:
                s = torch.from_numpy(np.stack(states[-K:]))[None].to(device)
                a = torch.tensor(actions[-K:], dtype=torch.long)[None].to(device)
                r = torch.tensor(rtgs[-K:], dtype=torch.float32)[None].to(device)
                ts = torch.tensor(timesteps[-K:], dtype=torch.long)[None].to(device)
                with torch.no_grad():
                    action = int(torch.argmax(model(s, a, r, ts)[0, -1]).item())

                obs, reward, done, truncated, _ = env.step(action)
                actions[-1] = action
                chosen.append(action)
                ep_reward += reward
                disc_reward += (gamma ** t) * reward
                t += 1
                if done or truncated:
                    break
                states.append(preprocess_state(obs, state_type))
                actions.append(0)
                rtgs.append(rtgs[-1] - reward / RTG_SCALE)
                timesteps.append(t)

            # esito: vittoria = automa in stato accettante; fallimento = automa in uno
            # stato di fallimento (reward -100, es. lava/porta nei task "evita"); altrimenti
            # l'episodio e' finito per timeout (50 passi)
            q = env.curr_automaton_state
            if env.automaton.acceptance[q]:
                outcome = "success"
            elif env.automaton.rewards[q] == -100:
                outcome = "failure"
            else:
                outcome = "timeout"
            rows.append({
                "start_pos": str(start_pos),
                "run_id": run + 1,
                "outcome": outcome,
                "success": int(outcome == "success"),
                "reward_puro": ep_reward,
                "discounted_reward": disc_reward,
                "episode_length": t,
                "action": str(chosen),
            })

    df = pd.DataFrame(rows)
    csv_path = paths["csv"]
    tmp_path = csv_path.with_suffix(".csv.tmp")
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, csv_path)

    res = {
        "success_rate": 100.0 * (df["outcome"] == "success").mean(),
        "failure_rate": 100.0 * (df["outcome"] == "failure").mean(),
        "timeout_rate": 100.0 * (df["outcome"] == "timeout").mean(),
        "reward_mean": df["reward_puro"].mean(), "reward_std": df["reward_puro"].std(),
        "disc_reward_mean": df["discounted_reward"].mean(), "disc_reward_std": df["discounted_reward"].std(),
        "length_mean": df["episode_length"].mean(),
        "episodes": len(df), "csv": csv_path,
    }
    summary = [
        f"Model: {paths['model']}",
        "Config: " + ", ".join(f"{k}={v}" for k, v in ckpt["config"].items()),
        f"Episodes: {res['episodes']} ({len(POSSIBLE_STARTS)} starts x {num_runs} runs) | "
        f"target_return={target_return} | gamma={gamma}",
        f"Success rate: {res['success_rate']:.1f}% | failure: {res['failure_rate']:.1f}% | timeout: {res['timeout_rate']:.1f}%",
        f"Reward: {res['reward_mean']:.2f} +- {res['reward_std']:.2f}",
        f"Discounted reward: {res['disc_reward_mean']:.2f} +- {res['disc_reward_std']:.2f}",
        f"Episode length: {res['length_mean']:.2f}",
        f"Evaluation time: {(time.time() - t_start) / 60:.1f} min",
    ]
    paths["summary"].write_text("\n".join(summary) + "\n")
    return res
