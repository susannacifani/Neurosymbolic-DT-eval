import argparse
import copy
import random
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn

# cartella NRM (Dataset/ e WorldModel/)
ROOT = Path(__file__).resolve().parent.parent
# il world model non usa il reward e i dataset distance e three_value contengono le stesse
# identiche traiettorie (cambia solo il reward), quindi i dati si leggono sempre da distance
DATA_REWARD_TYPE = "distance"
NUM_ACTIONS = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Dati: frame + coppie (t, t+1) dentro la stessa traiettoria
# ---------------------------------------------------------------------------
def load_data(h5_path, seed):
    # Nel GridWorld gli stati distinti sono pochissimi (una immagine per cella), quindi
    # invece di tenere in RAM decine di migliaia di copie identiche si salva una sola
    # copia per stato distinto ("canonical") e, per ogni frame, l'indice dello stato.
    canonical_list, state_of_bytes = [], {}
    frame_label, actions, pair_idx, traj_of_pair = [], [], [], []
    offset = 0
    with h5py.File(h5_path, "r") as f:
        keys = list(f.keys())
        for t, k in enumerate(keys):
            # in coda l'osservazione finale (dopo l'ultima azione): cosi' anche l'ultima
            # transizione, quella che porta sull'oggetto finale, entra nei dati
            obs = np.concatenate([f[k]["observations"][:], f[k]["final_observation"][:][None]])
            act = f[k]["actions"][:]
            assert len(act) == len(obs) - 1
            for frame in obs:
                b = frame.tobytes()
                if b not in state_of_bytes:
                    state_of_bytes[b] = len(canonical_list)
                    canonical_list.append(frame.copy())
                frame_label.append(state_of_bytes[b])
            for i in range(len(act)):
                pair_idx.append((offset + i, offset + i + 1))
                actions.append(int(act[i]))
                traj_of_pair.append(t)
            offset += len(obs)

    canonical = torch.from_numpy(np.stack(canonical_list)).float()
    frame_label = torch.tensor(frame_label, dtype=torch.long)
    pair_idx = torch.tensor(pair_idx, dtype=torch.long)
    actions = torch.tensor(actions, dtype=torch.long)
    traj_of_pair = np.array(traj_of_pair)

    # split per traiettoria, cosi' nessuna traiettoria di test e' vista in training
    traj_ids = list(range(len(keys)))
    random.Random(seed).shuffle(traj_ids)
    n = len(traj_ids)
    split = {
        "train": set(traj_ids[: int(0.8 * n)]),
        "val": set(traj_ids[int(0.8 * n): int(0.9 * n)]),
        "test": set(traj_ids[int(0.9 * n):]),
    }
    pairs = {}
    for name, ids in split.items():
        mask = torch.from_numpy(np.isin(traj_of_pair, list(ids)))
        pairs[name] = (pair_idx[mask], actions[mask])

    return canonical, frame_label, pairs


# ---------------------------------------------------------------------------
# Modelli
# ---------------------------------------------------------------------------
class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, latent_dim),
        )
        self.decoder_fc = nn.Linear(latent_dim, 128 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(self.decoder_fc(z).view(-1, 128, 8, 8))

    def forward(self, x):
        return self.decode(self.encode(x))


class MLPAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, input_dim),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        return self.decode(self.encode(x))


class DynamicsModel(nn.Module):
    def __init__(self, latent_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + NUM_ACTIONS, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, z, a):
        a_onehot = nn.functional.one_hot(a, NUM_ACTIONS).float()
        return self.net(torch.cat([z, a_onehot], dim=-1))


# ---------------------------------------------------------------------------
# Fase 1: autoencoder
# ---------------------------------------------------------------------------
def frames_of_split(pairs, split):
    return torch.unique(pairs[split][0].flatten())


def eval_recon(ae, canonical, frame_label, idx, batch_size):
    ae.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(idx), batch_size):
            x = canonical[frame_label[idx[i:i + batch_size]]].to(device)
            total += nn.functional.mse_loss(ae(x), x, reduction="sum").item()
            n += x.numel()
    return total / n


def train_autoencoder(ae, canonical, frame_label, pairs, epochs, batch_size, lr, log):
    opt = torch.optim.Adam(ae.parameters(), lr=lr)
    train_idx = frames_of_split(pairs, "train")
    val_idx = frames_of_split(pairs, "val")
    best_val, best_state = float("inf"), None

    for epoch in range(1, epochs + 1):
        ae.train()
        perm = train_idx[torch.randperm(len(train_idx))]
        losses = []
        for i in range(0, len(perm), batch_size):
            x = canonical[frame_label[perm[i:i + batch_size]]].to(device)
            if len(x) < 2:
                continue
            loss = nn.functional.mse_loss(ae(x), x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        val_loss = eval_recon(ae, canonical, frame_label, val_idx, batch_size)
        if val_loss < best_val:
            best_val, best_state = val_loss, copy.deepcopy(ae.state_dict())
        log(f"[AE] Epoch {epoch} | Train reconstruction MSE: {np.mean(losses):.6f} | Val reconstruction MSE: {val_loss:.6f}")

    ae.load_state_dict(best_state)
    log(f"[AE] Best val reconstruction MSE: {best_val:.6f}")
    return ae


# ---------------------------------------------------------------------------
# Fase 2: modello di dinamica nel latente
# ---------------------------------------------------------------------------
def encode_all(ae, canonical, frame_label):
    # si codificano solo gli stati distinti, poi si espande con le etichette dei frame
    ae.eval()
    with torch.no_grad():
        z_canonical = ae.encode(canonical.to(device)).cpu()
    return z_canonical[frame_label]


def eval_latent_mse(model, Z, pairs_split, batch_size):
    idx, act = pairs_split
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(idx), batch_size):
            z = Z[idx[i:i + batch_size, 0]].to(device)
            z_next = Z[idx[i:i + batch_size, 1]].to(device)
            a = act[i:i + batch_size].to(device)
            total += nn.functional.mse_loss(model(z, a), z_next, reduction="sum").item()
            n += z_next.numel()
    return total / n


def train_dynamics(model, Z, pairs, epochs, batch_size, lr, log):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    idx, act = pairs["train"]
    best_val, best_state = float("inf"), None

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(idx))
        losses = []
        for i in range(0, len(perm), batch_size):
            b = perm[i:i + batch_size]
            z = Z[idx[b, 0]].to(device)
            z_next = Z[idx[b, 1]].to(device)
            a = act[b].to(device)
            loss = nn.functional.mse_loss(model(z, a), z_next)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        val_loss = eval_latent_mse(model, Z, pairs["val"], batch_size)
        if val_loss < best_val:
            best_val, best_state = val_loss, copy.deepcopy(model.state_dict())
        log(f"[DYN] Epoch {epoch} | Train latent MSE: {np.mean(losses):.6f} | Val latent MSE: {val_loss:.6f}")

    model.load_state_dict(best_state)
    log(f"[DYN] Best val latent MSE: {best_val:.6f}")
    return model


# ---------------------------------------------------------------------------
# Test finale
# ---------------------------------------------------------------------------
def nearest_state(decoded, canonical):
    # a quale stato "vero" (cella della griglia) somiglia di piu' lo stato decodificato
    d = torch.cdist(decoded.flatten(1), canonical.flatten(1))
    return d.argmin(dim=1)


def evaluate_test(ae, model, Z, canonical, state_label, pairs, batch_size):
    idx, act = pairs["test"]
    ae.eval()
    model.eval()
    canonical = canonical.to(device)
    sums = {"model_mse": 0.0, "ident_mse": 0.0, "model_acc": 0, "ident_acc": 0}
    n_elems, n = 0, 0
    with torch.no_grad():
        for i in range(0, len(idx), batch_size):
            z = Z[idx[i:i + batch_size, 0]].to(device)
            z_next = Z[idx[i:i + batch_size, 1]].to(device)
            a = act[i:i + batch_size].to(device)
            true_label = state_label[idx[i:i + batch_size, 1]].to(device)

            z_pred = model(z, a)
            sums["model_mse"] += nn.functional.mse_loss(z_pred, z_next, reduction="sum").item()
            sums["ident_mse"] += nn.functional.mse_loss(z, z_next, reduction="sum").item()
            sums["model_acc"] += (nearest_state(ae.decode(z_pred), canonical) == true_label).sum().item()
            sums["ident_acc"] += (nearest_state(ae.decode(z), canonical) == true_label).sum().item()
            n_elems += z_next.numel()
            n += len(z)

    return {
        "model_mse": sums["model_mse"] / n_elems,
        "ident_mse": sums["ident_mse"] / n_elems,
        "model_acc": 100.0 * sums["model_acc"] / n,
        "ident_acc": 100.0 * sums["ident_acc"] / n,
    }


def output_paths(state_type, task, dataset, seed):
    """WorldModel/<State>/task<N>/seed<S>/{model,latent,log}/..._<dataset>"""
    run_dir = ROOT / "WorldModel" / state_type.capitalize() / f"task{task}" / f"seed{seed}"
    return {
        "autoencoder": run_dir / "model" / f"autoencoder_{dataset}.pth",
        "dynamics": run_dir / "model" / f"dynamics_{dataset}.pth",
        "latent": run_dir / "latent" / f"latent_transitions_{dataset}.npz",
        "log": run_dir / "log" / f"log_{dataset}.txt",
    }


def run(state_type="image", task=1, dataset="medium", latent_dim=None,
        ae_epochs=20, dyn_epochs=50, batch_size=256, lr=1e-3, seed=0):
    set_seed(seed)
    latent_dim = latent_dim or (16 if state_type == "image" else 4)
    folder_state = state_type.capitalize()

    h5_path = ROOT / "Dataset" / folder_state / DATA_REWARD_TYPE / f"task{task}" / f"dataset_{dataset}.h5"
    paths = output_paths(state_type, task, dataset, seed)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    log_path = paths["log"]
    open(log_path, "w").close()

    def log(msg):
        print(msg)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    log(f"Dataset: {h5_path} | seed: {seed}")
    canonical, frame_label, pairs = load_data(h5_path, seed)
    log(f"Frames: {len(frame_label)} | distinct states: {len(canonical)} | latent_dim: {latent_dim}")
    for s in ("train", "val", "test"):
        log(f"Transitions {s}: {len(pairs[s][0])}")

    if state_type == "image":
        ae = ConvAutoencoder(latent_dim).to(device)
    else:
        ae = MLPAutoencoder(canonical.shape[1], latent_dim).to(device)
    ae = train_autoencoder(ae, canonical, frame_label, pairs, ae_epochs, batch_size, lr, log)
    torch.save(ae.state_dict(), paths["autoencoder"])

    Z = encode_all(ae, canonical, frame_label)
    latent_dataset = {}
    for s in ("train", "val", "test"):
        idx, act = pairs[s]
        latent_dataset[f"{s}_z"] = Z[idx[:, 0]].numpy()
        latent_dataset[f"{s}_a"] = act.numpy()
        latent_dataset[f"{s}_z_next"] = Z[idx[:, 1]].numpy()
    np.savez(paths["latent"], **latent_dataset)
    dyn = DynamicsModel(latent_dim).to(device)
    dyn = train_dynamics(dyn, Z, pairs, dyn_epochs, batch_size, lr, log)
    torch.save(dyn.state_dict(), paths["dynamics"])

    res = evaluate_test(ae, dyn, Z, canonical, frame_label, pairs, batch_size)
    log("\nTEST RESULTS")
    log(f"Latent MSE - model: {res['model_mse']:.6f} | identity baseline (z'=z): {res['ident_mse']:.6f}")
    log(f"Next-state position accuracy - model: {res['model_acc']:.2f}% | identity baseline (z'=z): {res['ident_acc']:.2f}%")
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state_type", choices=["image", "symbolic"], default="image")
    parser.add_argument("--task", type=int, default=1)
    parser.add_argument("--dataset", choices=["medium_expert", "medium", "medium_replay"], default="medium")
    parser.add_argument("--latent_dim", type=int, default=None)
    parser.add_argument("--ae_epochs", type=int, default=20)
    parser.add_argument("--dyn_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
