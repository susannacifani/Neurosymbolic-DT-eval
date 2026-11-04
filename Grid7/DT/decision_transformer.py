"""Decision Transformer per la griglia 7x7 con oggetti in posizioni casuali.

Dataset: Dataset/<reward_type>/task<N>/dataset_<dataset>.h5 (stesso file per symbolic e image: contiene
disposizione e posizioni, le osservazioni si ricostruiscono qui).
Osservazione vista dal modello:
  symbolic: mappa (5, 7, 7) con un canale per tipo di oggetto e uno per l'agente -> piccola CNN
  image:    immagine (3, 112, 112), identica a quella dell'ambiente, composta sulla GPU con i
            tasselli dell'ImageRenderer -> CNN
Per entrambe il dataset fornisce i "codici" delle caselle (contenuto + 5 se c'e' l'agente), da cui
la mappa o l'immagine si ottengono in modo esatto.
Valutazione (solo a training finito): le prime NUM_EVAL_EPISODES disposizioni risolvibili a partire dal
seed EVAL_SEED_BASE, mai viste in training e uguali per tutti i modelli dello stesso task. Le
disposizioni senza soluzione si escludono, come nei dataset.
Opzione automaton (di default disattivata): il DT riceve a ogni passo anche lo stato dell'automa della
formula (quali sotto-obiettivi sono gia' stati raggiunti), come vettore one-hot.
  automaton="true":     stato vero, calcolato con i simboli veri (come un grounder perfetto)
  automaton="grounder": l'automa avanza con i simboli predetti dal grounder allenato solo dai reward
                        (Grounding/<State>/<reward>/task<N>/seed<S>/grounder/grounder_<dataset>.pth,
                        stesso dataset e stesso seed del DT); se il grounder sbaglia, lo stato e' sbagliato

Output:  DT/<State>/<reward_type>/task<N>/seed<S>/
         dt_model/dt_<dataset>.pth            modello + configurazione con cui e' stato allenato
         log/train_log_<dataset>.txt          iperparametri, loss/accuracy per epoca, tempi
         eval/eval_<dataset>.csv              un episodio per riga
         eval/eval_summary_<dataset>.txt      riepilogo della valutazione
"""
import json
import os
import sys
import random
import time
from pathlib import Path

# necessario per gli algoritmi deterministici di cuBLAS (va impostato prima di usare la GPU)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
# memoria della GPU in segmenti espandibili: senza, la memoria riservata si frammenta (fino a 11.5 GB
# riservati per 4.7 GB usati) e quella in eccesso finisce nella RAM, con epoch fino a 10 volte piu' lente.
# Cambia solo come si alloca la memoria, non i calcoli: stessi pesi, bit per bit
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# cartella Grid7 (Dataset/ e DT/) e motore comune in Grid7/NeuralRewardMachines
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "NeuralRewardMachines"))
sys.path.insert(0, str(ROOT / "Dataset"))
sys.path.insert(0, str(ROOT / "Grounding"))

from LTL_tasks import formulas
from RL.Env.Environment import GridWorldEnv, ImageRenderer, OBJECT_TYPES
from planner import shortest_path
from grounder import grounder_path, load_grounder, predict_symbols, true_symbols

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# modalita' deterministica: sulla GPU alcune operazioni (es. i gradienti delle convoluzioni) usano di
# default algoritmi che sommano in un ordine variabile, per cui lo stesso training con lo stesso seed
# da' modelli diversi (con image fino a ~5 punti di success rate). Con queste impostazioni la GPU
# usa solo algoritmi riproducibili: stesso seed, stesso modello.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

NUM_ACTIONS = 4
GRID_SIZE = 7
NUM_CONTENTS = ImageRenderer.NUM_CONTENTS   # P, L, D, G, vuota; codice = contenuto + 5 * agente
MAX_EP_LEN = 128       # dimensione della tabella dei timestep (gli episodi durano al massimo 100 passi)
RTG_SCALE = 100.0
NUM_EVAL_EPISODES = 500
EVAL_CHUNK = 256             # episodi elaborati insieme in valutazione (limita la memoria della GPU)
EVAL_SEED_BASE = 1_000_000   # i dataset usano seed < 1_000_000: disposizioni di valutazione mai viste

# iperparametri
CONTEXT_LEN = 50       # copre l'intero episodio per le traiettorie ottime; con rumore (epsilon 0.75) il 12% delle
                       # traiettorie di medium supera i 50 passi (34% in medium_replay, contando le random)
FIXED_VAL = True       # validation su blocchi fissi (ogni passo una volta) invece che una finestra casuale per traiettoria
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 64        # uguale per symbolic e image
LR = 1e-4
HIDDEN_SIZE = 128
N_LAYERS = 3
N_HEADS = 4
DROPOUT = 0.1
MAX_EPOCHS = 500
PATIENCE = 10


def dataset_path(reward_type, task, dataset):
    return ROOT / "Dataset" / reward_type / f"task{task}" / f"dataset_{dataset}.h5"


# cartella dei risultati; la variabile d'ambiente DT_RESULTS_ROOT permette di spostarla (per i test)
RESULTS_ROOT = Path(os.environ.get("DT_RESULTS_ROOT", ROOT / "DT"))


def run_dir(state_type, reward_type, task, seed):
    return RESULTS_ROOT / state_type.capitalize() / reward_type / f"task{task}" / f"seed{seed}"


def output_paths(state_type, reward_type, task, dataset, seed, out_dir=None):
    """Percorsi dei file di una run; le cartelle vengono create solo quando servono.
    out_dir sostituisce DT/<State>/<reward>/task<N>/seed<S> (utile per i test e le ablation)."""
    base = Path(out_dir) if out_dir else run_dir(state_type, reward_type, task, seed)
    return {"model": base / "dt_model" / f"dt_{dataset}.pth",
            "log": base / "log" / f"train_log_{dataset}.txt",
            "csv": base / "eval" / f"eval_{dataset}.csv",
            "summary": base / "eval" / f"eval_summary_{dataset}.txt"}


def cell_codes(grid, positions):
    """codici delle caselle per ogni posizione dell'agente: (T, 7, 7), indicizzati [x, y] come grid"""
    codes = np.repeat(np.asarray(grid, dtype=np.int64)[None], len(positions), axis=0)
    t = np.arange(len(positions))
    codes[t, positions[:, 0], positions[:, 1]] += NUM_CONTENTS
    return codes


# ---------------------------------------------------------------------------
# Stato dell'automa
# ---------------------------------------------------------------------------
class AutomatonTracker:
    """Stato dell'automa della formula con una numerazione canonica.

    La libreria che costruisce l'automa numera gli stati in modo diverso a ogni esecuzione, quindi
    qui gli stati vengono rinumerati con una visita in ampiezza dallo stato iniziale, seguendo i
    simboli in ordine fisso: la stessa formula da' sempre la stessa numerazione, in qualunque processo.

    L'automa avanza con il simbolo di ogni osservazione, compresa la prima (come nel training del
    grounder): simbolo vero (grounder=None) o predetto dal grounder. Con i simboli veri la prima
    osservazione non cambia nulla (la partenza e' sempre una casella vuota, che nello stato iniziale
    lascia l'automa fermo, per tutti i task)."""

    def __init__(self, env, grounder=None):
        self.env = env
        self.grounder = grounder
        aut = env.automaton
        order, seen, i = [0], {0}, 0
        while i < len(order):
            q = order[i]
            i += 1
            for sym in range(len(env.dictionary_symbols)):
                nxt = aut.transitions[q][sym]
                if nxt not in seen:
                    seen.add(nxt)
                    order.append(nxt)
        self.canonical = {q: c for c, q in enumerate(order)}
        self.num_states = len(order)

    def symbols(self, codes):
        """simbolo di ogni osservazione (codici (N, 7, 7)): vero o predetto dal grounder"""
        if self.grounder is None:
            return true_symbols(codes)
        return predict_symbols(self.grounder, codes)

    def step(self, q, symbol):
        """stato dell'automa (numerazione della libreria) dopo aver letto symbol"""
        return self.env.automaton.transitions[q][int(symbol)]

    def states_from_symbols(self, symbols):
        """stato canonico dell'automa a ogni osservazione, dopo averne letto il simbolo"""
        q, states = 0, []
        for s in symbols:
            q = self.step(q, s)
            states.append(self.canonical[q])
        return np.array(states, dtype=np.int64)


def make_tracker(automaton, state_type, reward_type, task, grounder=None):
    """automaton: None, "true" o "grounder" (in quel caso grounder = percorso del .pth)"""
    if automaton is None:
        return None
    if automaton not in ("true", "grounder"):
        raise ValueError(f"automaton deve essere None, 'true' o 'grounder', non {automaton!r}")
    env = GridWorldEnv(formulas[task - 1], state_type="symbolic", reward_type=reward_type)
    if automaton == "true":
        return AutomatonTracker(env)
    if not Path(grounder).exists():
        raise FileNotFoundError(f"grounder non trovato: {grounder} (va allenato prima, Grounding/experiments_grounding.py)")
    return AutomatonTracker(env, load_grounder(grounder, state_type))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class DTDataset(Dataset):
    """Finestre di `context_len` passi consecutivi; lo stato di ogni passo sono i codici delle 49
    caselle (da cui il modello ricostruisce mappa o immagine).
    all_windows=False: una finestra casuale per traiettoria (training).
    all_windows=True: ogni traiettoria divisa in blocchi consecutivi non sovrapposti, sempre gli
    stessi; ogni passo viene valutato esattamente una volta (validation)."""

    def __init__(self, h5_path, keys, context_len, all_windows=False, tracker=None):
        self.K = context_len
        self.tracker = tracker
        self.cache = {}
        with h5py.File(h5_path, "r") as f:
            for k in keys:
                g = f[k]
                grid, positions = g["grid"][:].astype(np.int64), g["positions"][:].astype(np.int64)
                self.cache[k] = {"codes": cell_codes(grid, positions[:-1]).astype(np.int8),   # osservazione prima di ogni azione
                                 "actions": g["actions"][:].astype(np.int64),
                                 "rtgs": (g["returns_to_go"][:] / RTG_SCALE).astype(np.float32)}
        if tracker is not None:
            # simboli di tutte le osservazioni in una volta sola (con il grounder: una passata sulla GPU)
            symbols = tracker.symbols(np.concatenate([self.cache[k]["codes"] for k in keys]))
            ends = np.cumsum([len(self.cache[k]["codes"]) for k in keys])
            for k, sym in zip(keys, np.split(symbols, ends[:-1])):
                self.cache[k]["automaton"] = tracker.states_from_symbols(sym)
        if all_windows:
            self.index = [(k, s) for k in keys for s in range(0, len(self.cache[k]["actions"]), self.K)]
        else:
            self.index = [(k, None) for k in keys]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        key, start = self.index[idx]
        g = self.cache[key]
        traj_len = len(g["actions"])
        if start is None:
            start = np.random.randint(0, traj_len - self.K + 1) if traj_len > self.K else 0
        end = min(start + self.K, traj_len)
        n = end - start

        # padding a destra: con la maschera causale i passi reali non vedono mai il padding (che
        # sta dopo), quindi equivale a una finestra senza padding, come in valutazione; il padding
        # e' escluso dalla loss tramite la maschera
        pad = self.K - n
        codes = np.zeros((self.K, GRID_SIZE, GRID_SIZE), dtype=np.int64)
        codes[:n] = g["codes"][start:end]
        item = {
            "states": torch.from_numpy(codes),
            "actions": torch.from_numpy(np.pad(g["actions"][start:end], (0, pad))),
            "rtgs": torch.from_numpy(np.pad(g["rtgs"][start:end], (0, pad))),
            "timesteps": torch.from_numpy(np.pad(np.arange(start, end), (0, pad))).long(),
            "mask": torch.from_numpy(np.concatenate([np.ones(n, dtype=bool), np.zeros(pad, dtype=bool)])),
        }
        if self.tracker is not None:
            one_hot = np.zeros((self.K, self.tracker.num_states), dtype=np.float32)
            one_hot[np.arange(n), g["automaton"][start:end]] = 1.0
            item["automaton"] = torch.from_numpy(one_hot)
        return item


# ---------------------------------------------------------------------------
# Modello
# ---------------------------------------------------------------------------
class ObservationEncoder(nn.Module):
    """Dai codici delle caselle (B, 7, 7) all'osservazione e poi al vettore di dimensione h.
    image:    immagine composta con i tasselli dell'ambiente (identica a env.render) -> CNN
    symbolic: mappa one-hot (5, 7, 7): 4 canali per tipo di oggetto + 1 per l'agente -> CNN"""

    def __init__(self, state_type, h):
        super().__init__()
        self.state_type = state_type
        if state_type == "image":
            renderer = ImageRenderer(GRID_SIZE)
            self.register_buffer("tiles", torch.from_numpy(renderer.tiles))   # (10, 3, c, c)
            self.cell = renderer.cell
            self.cnn = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=4, stride=2), nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2), nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=2), nn.ReLU(),
                nn.Flatten())
            side = renderer.px
        else:
            self.cnn = nn.Sequential(
                nn.Conv2d(len(OBJECT_TYPES) + 1, 32, kernel_size=3, padding=1), nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
                nn.Flatten())
            side = GRID_SIZE
        with torch.no_grad():
            flat = self.cnn(self.observation(torch.zeros(1, GRID_SIZE, GRID_SIZE, dtype=torch.long))).shape[1]
        self.out = nn.Linear(flat, h)

    def observation(self, codes):
        """codici (B, 7, 7) indicizzati [x, y] -> osservazione identica a quella dell'ambiente:
        image (B, 3, 112, 112) con riga = y e colonna = x; symbolic (B, 5, 7, 7) indicizzata [x, y]"""
        if self.state_type == "image":
            B, n, c = codes.shape[0], GRID_SIZE, self.cell
            t = self.tiles[codes.transpose(1, 2)]                          # (B, y, x, 3, c, c)
            return t.permute(0, 3, 1, 4, 2, 5).reshape(B, 3, n * c, n * c)
        content, agent = codes % NUM_CONTENTS, codes >= NUM_CONTENTS
        maps = [(content == obj) for obj in OBJECT_TYPES] + [agent]
        return torch.stack(maps, dim=1).float()

    def forward(self, codes):
        return self.out(self.cnn(self.observation(codes)))


class DecisionTransformer(nn.Module):
    def __init__(self, state_type, num_automaton_states=0):
        super().__init__()
        h = HIDDEN_SIZE
        self.hidden_size = h
        self.embed_state = ObservationEncoder(state_type, h)
        self.embed_rtg = nn.Linear(1, h)
        self.embed_action = nn.Embedding(NUM_ACTIONS, h)
        self.embed_timestep = nn.Embedding(MAX_EP_LEN, h)
        # stato dell'automa (one-hot), sommato all'embedding dell'osservazione; solo se richiesto
        self.embed_automaton = nn.Linear(num_automaton_states, h) if num_automaton_states > 0 else None

        layer = nn.TransformerEncoderLayer(d_model=h, nhead=N_HEADS, dim_feedforward=4 * h,
                                           dropout=DROPOUT, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=N_LAYERS)
        self.predict_action = nn.Linear(h, NUM_ACTIONS)

    def embed_states(self, states, mask=None):
        """embedding delle osservazioni (B, T, h) dai codici (B, T, 7, 7).
        mask (B, T): passi reali (False = padding). Se c'e', l'osservazione si calcola solo per i
        passi reali e il padding ha embedding nullo: con la maschera causale i passi reali non vedono
        mai il padding (che sta dopo), quindi le previsioni sui passi reali non cambiano, ma la CNN
        non lavora sui passi finti (con episodi di ~8 passi e K = 50 sarebbero la maggior parte)"""
        B, T = states.shape[:2]
        flat = states.reshape(B * T, GRID_SIZE, GRID_SIZE)
        if mask is None:
            return self.embed_state(flat).view(B, T, self.hidden_size)
        real = mask.reshape(-1)
        state_emb = torch.zeros(B * T, self.hidden_size, device=states.device)
        state_emb[real] = self.embed_state(flat[real])
        return state_emb.view(B, T, self.hidden_size)

    def forward(self, states, actions, rtgs, timesteps, automaton=None, mask=None, state_emb=None):
        """state_emb: embedding delle osservazioni gia' calcolati (in valutazione si calcolano una
        volta per passo e si riusano); altrimenti si calcolano da states"""
        B, T = actions.shape[:2]
        if state_emb is None:
            state_emb = self.embed_states(states, mask)
        time_emb = self.embed_timestep(timesteps)
        state_emb = state_emb + time_emb
        if self.embed_automaton is not None:
            state_emb = state_emb + self.embed_automaton(automaton)
        action_emb = self.embed_action(actions) + time_emb
        rtg_emb = self.embed_rtg(rtgs.unsqueeze(-1)) + time_emb

        # ogni passo t diventa (R_t, s_t, a_t); l'azione a_t si predice dal token s_t,
        # che con la maschera causale vede R_t, s_t e tutto il passato, ma non a_t
        stacked = torch.stack((rtg_emb, state_emb, action_emb), dim=2).view(B, 3 * T, self.hidden_size)
        causal = torch.triu(torch.ones(3 * T, 3 * T, device=actions.device), diagonal=1).bool()
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
        logits = model(b["states"], b["actions"], b["rtgs"], b["timesteps"], b.get("automaton"), b["mask"])[b["mask"]]
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
          weight_decay=None, max_epochs=MAX_EPOCHS, out_dir=None, automaton=None, batch_size=None):
    context_len = CONTEXT_LEN if context_len is None else context_len
    fixed_val = FIXED_VAL if fixed_val is None else fixed_val
    weight_decay = WEIGHT_DECAY if weight_decay is None else weight_decay
    bs = BATCH_SIZE if batch_size is None else batch_size
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    paths = output_paths(state_type, reward_type, task, dataset, seed, out_dir)
    for key in ("model", "log"):
        paths[key].parent.mkdir(parents=True, exist_ok=True)
    h5_path = dataset_path(reward_type, task, dataset)

    with h5py.File(h5_path, "r") as f:
        keys = sorted(f.keys(), key=int)
    random.Random(seed).shuffle(keys)
    split = int(0.8 * len(keys))
    grounder = str(grounder_path(state_type, reward_type, task, dataset, seed)) if automaton == "grounder" else None
    tracker = make_tracker(automaton, state_type, reward_type, task, grounder)
    # caricare il grounder consuma numeri casuali (costruzione dei suoi layer): si rifissa il seed, cosi'
    # il DT parte dagli stessi pesi iniziali e vede i batch nello stesso ordine con qualunque automaton
    # (senza automa e con "true" non cambia nulla: tra il seed sopra e qui non si consuma nessun numero)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_set = DTDataset(h5_path, keys[:split], context_len, tracker=tracker)
    val_set = DTDataset(h5_path, keys[split:], context_len, all_windows=fixed_val, tracker=tracker)
    train_loader = DataLoader(train_set, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=bs, shuffle=False)

    num_automaton_states = tracker.num_states if tracker is not None else 0
    model = DecisionTransformer(state_type, num_automaton_states).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=weight_decay)
    config = {"state_type": state_type, "reward_type": reward_type, "task": task, "dataset": dataset,
              "seed": seed, "context_len": context_len, "fixed_val": fixed_val, "weight_decay": weight_decay,
              "batch_size": bs, "lr": LR, "hidden_size": HIDDEN_SIZE, "n_layers": N_LAYERS,
              "n_heads": N_HEADS, "dropout": DROPOUT, "max_epochs": max_epochs, "patience": PATIENCE,
              "automaton": automaton, "num_automaton_states": num_automaton_states, "grounder": grounder}

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
def eval_seeds(task, num_episodes=NUM_EVAL_EPISODES):
    """seed delle disposizioni di test: i primi num_episodes risolvibili a partire da EVAL_SEED_BASE.
    La risolvibilita' non dipende dal tipo di reward (stessi stati di vittoria e di fallimento),
    quindi la lista e' la stessa per distance e three_value. Si calcola una volta per task e si salva
    in Dataset/test_layouts/task<N>.json (per i task con molti stati dell'automa il controllo con il
    planner richiede decine di secondi)."""
    cache = ROOT / "Dataset" / "test_layouts" / f"task{task}.json"
    if cache.exists():
        saved = json.loads(cache.read_text())
        if saved["base"] == EVAL_SEED_BASE and len(saved["seeds"]) >= num_episodes:
            return saved["seeds"][:num_episodes]
    env = GridWorldEnv(formulas[task - 1], state_type="symbolic", reward_type="three_value")
    seeds, seed = [], EVAL_SEED_BASE
    while len(seeds) < num_episodes:
        env.reset(seed=seed)
        if shortest_path(env, np.random.default_rng(0)) is not None:
            seeds.append(seed)
        seed += 1
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"base": EVAL_SEED_BASE, "seeds": seeds}))
    os.replace(tmp, cache)
    return seeds


def evaluate(state_type, reward_type, task, dataset, seed=0, num_episodes=NUM_EVAL_EPISODES, gamma=0.99,
             target_return=1.0, out_dir=None):
    """Un episodio per ciascuna delle num_episodes disposizioni di test (eval_seeds): ambiente e policy
    sono deterministici, quindi ogni disposizione si gioca una volta.
    Gli episodi si giocano tutti insieme: a ogni passo il DT sceglie con una sola chiamata le azioni di
    tutti gli episodi ancora in corso (che hanno tutti la stessa lunghezza di storia, perche' partono
    insieme), e ogni episodio fa il suo passo nell'ambiente con le stesse regole di un episodio singolo."""
    paths = output_paths(state_type, reward_type, task, dataset, seed, out_dir)
    paths["csv"].parent.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(paths["model"], map_location=device)
    K = ckpt["config"]["context_len"]
    automaton = ckpt["config"].get("automaton")
    model = DecisionTransformer(state_type, ckpt["config"].get("num_automaton_states", 0)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    env = GridWorldEnv(formulas[task - 1], state_type="symbolic", reward_type=reward_type)
    # con "grounder" lo stato dell'automa dato al DT avanza con i simboli predetti dal grounder salvato
    # nella configurazione del modello; l'esito dell'episodio lo decide sempre l'ambiente
    tracker = make_tracker(automaton, state_type, reward_type, task, ckpt["config"].get("grounder"))
    test_seeds = eval_seeds(task, num_episodes)
    t_start = time.time()

    # stato di ogni episodio (l'ambiente e' uno solo: prima di ogni passo gli si carica lo stato
    # dell'episodio, e dopo il passo lo si rilegge)
    episodes = []
    for env_seed in test_seeds:
        env.reset(seed=env_seed)
        episodes.append({"seed": env_seed, "grid": env.grid.copy(), "pos": env._agent_location.copy(),
                         "q": env.curr_automaton_state, "step": 0,
                         "start": tuple(int(v) for v in env._agent_location),
                         "reward": 0.0, "disc": 0.0, "chosen": [], "done": False})
    n = len(episodes)
    # storia allineata come nel training: lo slot t contiene (R_t, s_t, a_t); l'azione dello slot
    # corrente e' un segnaposto (0), invisibile al token s_t per via della maschera causale
    # embedding delle osservazioni: ogni osservazione passa nella CNN una volta sola, quando arriva,
    # e il risultato si riusa nei passi successivi (le osservazioni passate non cambiano)
    def embed(codes):
        with torch.no_grad():
            out = [model.embed_states(torch.from_numpy(codes[i:i + EVAL_CHUNK]).to(device)[:, None])[:, 0]
                   for i in range(0, len(codes), EVAL_CHUNK)]
        return torch.cat(out)
    first = np.stack([cell_codes(e["grid"], e["pos"][None])[0] for e in episodes])
    state_embs = [embed(first)]                                    # lista di (n, h), uno per passo
    actions = [np.zeros(n, dtype=np.int64)]
    rtgs = [np.full(n, target_return, dtype=np.float32)]
    aut_states = []
    if tracker:
        # stato dell'automa visto dal DT ("qa", numerazione della libreria): dopo il simbolo della prima osservazione
        for e, sym in zip(episodes, tracker.symbols(first)):
            e["qa"] = tracker.step(0, sym)
        aut_states = [np.array([tracker.canonical[e["qa"]] for e in episodes])]

    t = 0
    active = np.arange(n)
    while len(active):
        emb_hist = torch.stack(state_embs[-K:], dim=1)            # (n, L, h)
        act_hist = np.stack(actions[-K:], axis=1)
        rtg_hist = np.stack(rtgs[-K:], axis=1)
        aut_hist = np.stack(aut_states[-K:], axis=1) if tracker else None
        L = emb_hist.shape[1]
        chosen = []
        for c in range(0, len(active), EVAL_CHUNK):
            sel = active[c:c + EVAL_CHUNK]
            a = torch.from_numpy(act_hist[sel]).to(device)
            r = torch.from_numpy(rtg_hist[sel]).to(device)
            ts = torch.arange(t - L + 1, t + 1, device=device).expand(len(sel), L)
            aut = None
            if tracker:
                aut = nn.functional.one_hot(torch.from_numpy(aut_hist[sel]).to(device), tracker.num_states).float()
            with torch.no_grad():
                logits = model(None, a, r, ts, aut, state_emb=emb_hist[torch.from_numpy(sel).to(device)])
            chosen.append(torch.argmax(logits[:, -1], dim=-1).cpu().numpy())
        chosen = np.concatenate(chosen)

        new_codes = first.copy()
        new_rtg = rtgs[-1].copy()
        new_aut = aut_states[-1].copy() if tracker else None
        actions[-1] = actions[-1].copy()
        still = []
        for i, action in zip(active, chosen):
            e = episodes[i]
            env.grid, env._agent_location = e["grid"], e["pos"].copy()
            env.curr_automaton_state, env.curr_step = e["q"], e["step"]
            _, reward, done, truncated, _ = env.step(int(action))
            e["pos"], e["q"], e["step"] = env._agent_location.copy(), env.curr_automaton_state, env.curr_step
            actions[-1][i] = action
            e["chosen"].append(int(action))
            e["reward"] += reward
            e["disc"] += (gamma ** t) * reward
            if done or truncated:
                continue
            still.append(i)
            new_codes[i] = cell_codes(e["grid"], e["pos"][None])[0]
            new_rtg[i] = rtgs[-1][i] - reward / RTG_SCALE
        t += 1
        active = np.array(still, dtype=np.int64)
        if tracker and len(active):
            for i, sym in zip(active, tracker.symbols(new_codes[active])):
                episodes[i]["qa"] = tracker.step(episodes[i]["qa"], sym)
                new_aut[i] = tracker.canonical[episodes[i]["qa"]]
        # solo le osservazioni nuove degli episodi ancora in corso passano nella CNN
        new_emb = state_embs[-1].clone()
        if len(active):
            new_emb[torch.from_numpy(active).to(device)] = embed(new_codes[active])
        state_embs.append(new_emb)
        actions.append(np.zeros(n, dtype=np.int64))
        rtgs.append(new_rtg)
        if tracker:
            aut_states.append(new_aut)

    rows = []
    for e in episodes:
        # esito: vittoria = automa in stato accettante; fallimento = automa in uno stato di
        # fallimento (reward -100); altrimenti l'episodio e' finito per il limite di passi
        if env.automaton.acceptance[e["q"]]:
            outcome = "success"
        elif env.automaton.rewards[e["q"]] == -100:
            outcome = "failure"
        else:
            outcome = "timeout"
        rows.append({
            "env_seed": e["seed"],
            "start_pos": str(e["start"]),
            "outcome": outcome,
            "success": int(outcome == "success"),
            "reward_puro": e["reward"],
            "discounted_reward": e["disc"],
            "episode_length": e["step"],
            "action": str(e["chosen"]),
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
        f"Episodes: {res['episodes']} (disposizioni di test risolvibili, seed {test_seeds[0]}..{test_seeds[-1]}) | "
        f"target_return={target_return} | gamma={gamma}",
        f"Success rate: {res['success_rate']:.1f}% | failure: {res['failure_rate']:.1f}% | timeout: {res['timeout_rate']:.1f}%",
        f"Reward: {res['reward_mean']:.2f} +- {res['reward_std']:.2f}",
        f"Discounted reward: {res['disc_reward_mean']:.2f} +- {res['disc_reward_std']:.2f}",
        f"Episode length: {res['length_mean']:.2f}",
        f"Evaluation time: {(time.time() - t_start) / 60:.1f} min",
    ]
    paths["summary"].write_text("\n".join(summary) + "\n")
    return res
