"""Grounder (Neural Reward Machine) per la griglia 7x7.

Stesso metodo della 4x4 (Grid4/NeuralRewardMachines/RL/A2C_nrm.py):
- il grounder e' un classificatore che riceve l'osservazione intera (mappa (5, 7, 7) per symbolic,
  immagine (3, 112, 112) per image) e restituisce una distribuzione sui 5 simboli (P, L, D, G, vuoto),
  il simbolo della casella in cui si trova l'agente;
- si allena SOLO dai reward: la sequenza dei simboli predetti passa nell'automa della formula
  (ProbabilisticAutoma inizializzato dal DFA, normalizzato, congelato), che prevede a ogni passo la
  classe di reward (il potenziale dello stato dell'automa); loss = NLL della classe vera, con pesi per
  classe 1/frequenza;
- etichette: classe di reward di ogni osservazione della traiettoria, compresa quella finale;
- split 80/10/10 delle traiettorie (train / validation / test); con distance train bilanciato tra
  traiettorie a reward basso e alto; si tiene il classifier della epoch con la miglior balanced
  accuracy sul validation; numeri finali sul test.
Le osservazioni non si tengono in memoria: mappa o immagine si compongono sulla GPU dai codici delle
caselle (contenuto + 5 se c'e' l'agente), identiche a quelle dell'ambiente.

Dati: Grid7/Dataset/<reward>/task<N>/dataset_<dataset>.h5
Output: Grid7/Grounding/<State>/<reward>/task<N>/seed<S>/
        grounder/grounder_<dataset>.pth   pesi del classifier (epoch migliore sul validation)
        log/log_<dataset>.txt             config, epoch per epoch, risultati sul test
"""
import copy
import itertools
import os
import random
import sys
import time
from pathlib import Path

# algoritmi deterministici di cuBLAS; memoria della GPU in segmenti espandibili (evita la frammentazione)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import h5py
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "NeuralRewardMachines"))

from LTL_tasks import formulas
from RL.Env.Environment import GridWorldEnv, ImageRenderer, OBJECT_TYPES
from RL.NRM.DeepAutoma import ProbabilisticAutoma

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# stesso seed, stesso modello
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

GRID_SIZE = 7
NUM_CONTENTS = ImageRenderer.NUM_CONTENTS   # P, L, D, G, vuota; codice = contenuto + 5 * agente
NUM_SYMBOLS = NUM_CONTENTS

# iperparametri
EPOCHS = 100
PATIENCE = 20          # early stopping sulla balanced accuracy del validation
LR = 1e-3
MIN_LR = 1e-5
CLASS_WEIGHT_POWER = 1.0          # peso di classe = 1 / frequenza^potenza
# traiettorie per batch di training, uguale per symbolic e image (con image limitato dalla memoria della GPU)
CHUNK_SIZE = {"symbolic": 32, "image": 32}
EVAL_CHUNK_SIZE = {"symbolic": 500, "image": 32}   # traiettorie per blocco di valutazione


def dataset_path(reward_type, task, dataset):
    return ROOT / "Dataset" / reward_type / f"task{task}" / f"dataset_{dataset}.h5"


def run_dir(state_type, reward_type, task, seed):
    return ROOT / "Grounding" / state_type.capitalize() / reward_type / f"task{task}" / f"seed{seed}"


def grounder_path(state_type, reward_type, task, dataset, seed):
    return run_dir(state_type, reward_type, task, seed) / "grounder" / f"grounder_{dataset}.pth"


def log_path(state_type, reward_type, task, dataset, seed):
    return run_dir(state_type, reward_type, task, seed) / "log" / f"log_{dataset}.txt"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Osservazioni
# ---------------------------------------------------------------------------
def cell_codes(grid, positions):
    """codici delle caselle per ogni posizione dell'agente: (T, 7, 7), indicizzati [x, y] come grid"""
    codes = np.repeat(np.asarray(grid, dtype=np.int64)[None], len(positions), axis=0)
    t = np.arange(len(positions))
    codes[t, positions[:, 0], positions[:, 1]] += NUM_CONTENTS
    return codes


class Observation(nn.Module):
    """Dai codici delle caselle (B, 7, 7) all'osservazione dell'ambiente:
    image (B, 3, 112, 112) composta con i tasselli dell'ImageRenderer (riga = y, colonna = x);
    symbolic (B, 5, 7, 7) indicizzata [x, y]: 4 canali per tipo di oggetto + 1 per l'agente"""

    def __init__(self, state_type):
        super().__init__()
        self.state_type = state_type
        if state_type == "image":
            renderer = ImageRenderer(GRID_SIZE)
            self.register_buffer("tiles", torch.from_numpy(renderer.tiles), persistent=False)
            self.cell = renderer.cell

    def forward(self, codes):
        codes = codes.long()
        if self.state_type == "image":
            B, n, c = codes.shape[0], GRID_SIZE, self.cell
            t = self.tiles[codes.transpose(1, 2)]                          # (B, y, x, 3, c, c)
            return t.permute(0, 3, 1, 4, 2, 5).reshape(B, 3, n * c, n * c)
        content, agent = codes % NUM_CONTENTS, codes >= NUM_CONTENTS
        maps = [(content == obj) for obj in OBJECT_TYPES] + [agent]
        return torch.stack(maps, dim=1).float()


def true_symbols(codes):
    """simbolo vero della casella dell'agente per ogni osservazione (codici (N, 7, 7))"""
    flat = np.asarray(codes, dtype=np.int64).reshape(len(codes), -1)
    return flat[np.arange(len(codes)), flat.argmax(axis=1)] - NUM_CONTENTS   # la casella dell'agente ha codice >= 5


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------
class GlobalMaxPool(nn.Module):
    """massimo di ogni canale su tutte le posizioni: (B, C, H, W) -> (B, C). Come AdaptiveMaxPool2d(1),
    ma con amax, che sulla GPU ha il gradiente deterministico (AdaptiveMaxPool2d no)"""

    def forward(self, x):
        return x.amax(dim=(2, 3))


class Grounder(nn.Module):
    """Osservazione intera -> distribuzione sui 5 simboli. CNN con max pooling globale:
    image:    4 conv 3x3 stride 2 + BatchNorm (16-32-64-64 filtri), fino a 7x7 (una posizione per casella)
    symbolic: 2 conv 1x1 + BatchNorm (32-64 filtri): ogni posizione vede solo la propria casella
    poi: max pooling globale - Linear(64, 128) - ReLU - Dropout(0.2) - Linear(128, 5) - Softmax"""

    def __init__(self, state_type):
        super().__init__()
        self.state_type = state_type
        self.observation = Observation(state_type)
        if state_type == "image":
            chans, kernel, stride = [3, 16, 32, 64, 64], 3, 2
        else:
            chans, kernel, stride = [len(OBJECT_TYPES) + 1, 32, 64], 1, 1
        layers = []
        for c_in, c_out in zip(chans[:-1], chans[1:]):
            layers += [nn.Conv2d(c_in, c_out, kernel_size=kernel, stride=stride, padding=kernel // 2),
                       nn.BatchNorm2d(c_out), nn.ReLU()]
        self.features = nn.Sequential(*layers, GlobalMaxPool())
        self.classifier = nn.Sequential(nn.Linear(chans[-1], 128), nn.ReLU(), nn.Dropout(0.2),
                                        nn.Linear(128, NUM_SYMBOLS), nn.Softmax(dim=-1))

    def forward(self, codes):
        """codici (N, 7, 7) -> probabilita' dei simboli (N, 5)"""
        return self.classifier(self.features(self.observation(codes)))


def load_grounder(path, state_type):
    model = Grounder(state_type)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.to(device).eval()


def predict_symbols(model, codes, chunk=2048):
    """simbolo predetto (argmax) per ogni osservazione; codici (N, 7, 7) numpy o tensore"""
    codes = torch.as_tensor(np.asarray(codes), dtype=torch.long)
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(codes), chunk):
            out.append(model(codes[i:i + chunk].to(device)).argmax(dim=-1).cpu())
    return torch.cat(out).numpy() if out else np.zeros(0, dtype=np.int64)


# ---------------------------------------------------------------------------
# Dati
# ---------------------------------------------------------------------------
def trajectory_labels(rewards, rew_dictionary):
    """Classe di reward di ogni osservazione della traiettoria, compresa quella finale (come nella 4x4).
    rewards[t] e' il delta ottenuto con l'azione t, quindi il potenziale dello stato osservato a t e' la
    somma dei reward fino a t-1; l'osservazione finale ha il potenziale finale."""
    potenziali = np.concatenate([[0.0], np.cumsum(rewards)])
    values = np.array(list(rew_dictionary))
    classes = np.array([rew_dictionary[v] for v in rew_dictionary])
    return classes[np.abs(potenziali[:, None] - values[None]).argmin(axis=1)]


def load_dataset(h5_path, rew_dictionary):
    """per ogni traiettoria: codici di tutte le osservazioni (T+1, 7, 7), etichette (T+1,), reward totale"""
    data = {}
    with h5py.File(h5_path, "r") as f:
        for k in sorted(f.keys(), key=int):
            g = f[k]
            grid, positions = g["grid"][:].astype(np.int64), g["positions"][:].astype(np.int64)
            rewards = g["rewards"][:]
            data[k] = {"codes": cell_codes(grid, positions).astype(np.int8),
                       "labels": trajectory_labels(rewards, rew_dictionary),
                       "total_reward": float(np.sum(rewards))}
    return data


def make_batch(data, keys):
    """codici dei soli passi reali (N, 7, 7), etichette (B, L) e maschera dei passi reali (B, L);
    L = lunghezza massima nel batch. I passi reali sono in ordine riga per riga, come mask."""
    lengths = [len(data[k]["labels"]) for k in keys]
    L = max(lengths)
    codes = torch.from_numpy(np.concatenate([data[k]["codes"] for k in keys]))
    labels = torch.zeros(len(keys), L, dtype=torch.long)
    mask = torch.zeros(len(keys), L, dtype=torch.bool)
    for i, k in enumerate(keys):
        labels[i, :lengths[i]] = torch.from_numpy(data[k]["labels"])
        mask[i, :lengths[i]] = True
    return codes, labels, mask


# ---------------------------------------------------------------------------
# Grounder + automa
# ---------------------------------------------------------------------------
def build_automaton(env):
    """automa della formula come ProbabilisticAutoma (initFromDfa, normalizzato, congelato), con gli stati
    rinumerati da una visita in ampiezza: la libreria numera gli stati in modo diverso a ogni esecuzione."""
    num_states, num_symbols, num_outputs, transitions, outputs = env.get_automaton_specs()
    order, i = [0], 0
    while i < len(order):
        for sym in range(num_symbols):
            nxt = transitions[order[i]][sym]
            if nxt not in order:
                order.append(nxt)
        i += 1
    if len(order) != num_states:
        raise ValueError("automa con stati non raggiungibili dallo stato iniziale")
    canonical = {q: c for c, q in enumerate(order)}
    transitions = {canonical[q]: {sym: canonical[transitions[q][sym]] for sym in range(num_symbols)} for q in order}
    outputs = [outputs[q] for q in order]
    automa = ProbabilisticAutoma(num_symbols, num_states, num_outputs)
    automa.initFromDfa(transitions, outputs)
    automa.normalized = True
    automa.trans_prob = automa.trans_prob.float().to(device)
    automa.rew_matrix = automa.rew_matrix.float().to(device)
    return automa


def predict_rewards(classifier, automa, codes, mask):
    """distribuzione sulle classi di reward (B, L, R) per ogni passo; il classifier vede solo i passi
    reali, il padding riceve un vettore nullo ed e' escluso da loss e metriche."""
    sym_real = classifier(codes.to(device))
    sym = torch.zeros(*mask.shape, NUM_SYMBOLS, device=device)
    sym[mask.to(device)] = sym_real
    if len(mask) < 2:
        raise ValueError("ProbabilisticAutoma.step richiede almeno 2 traiettorie per batch")
    _, pred_rew = automa(sym, 1.0)
    return pred_rew


def chunk_groups(keys, chunk):
    """keys a gruppi di chunk; l'automa vuole almeno 2 traiettorie per batch, quindi un eventuale
    ultimo gruppo da 1 si unisce al precedente"""
    groups = [keys[s:s + chunk] for s in range(0, len(keys), chunk)]
    if len(groups) > 1 and len(groups[-1]) == 1:
        groups[-2] = groups[-2] + groups.pop()
    return groups


def evaluate_counts(classifier, automa, data, keys, chunk):
    """conteggi di classi di reward predette correttamente (totali e per classe) e coppie
    (simbolo vero, simbolo predetto) per tutti i passi reali delle traiettorie keys"""
    classifier.eval()
    correct = total = 0
    class_correct, class_total = {}, {}
    sym_true, sym_pred = [], []
    with torch.no_grad():
        for batch in chunk_groups(keys, chunk):
            codes, labels, mask = make_batch(data, batch)
            pred = predict_rewards(classifier, automa, codes, mask).argmax(dim=-1).cpu()
            ok = (pred == labels) & mask
            correct += ok.sum().item()
            total += mask.sum().item()
            for c in labels[mask].unique().tolist():
                cm = mask & (labels == c)
                class_total[c] = class_total.get(c, 0) + cm.sum().item()
                class_correct[c] = class_correct.get(c, 0) + (ok & cm).sum().item()
            sym_true.append(true_symbols(codes.numpy()))
            sym_pred.append(classifier(codes.to(device)).argmax(dim=-1).cpu().numpy())
    return {"correct": correct, "total": total, "class_correct": class_correct, "class_total": class_total,
            "sym_true": np.concatenate(sym_true), "sym_pred": np.concatenate(sym_pred)}


def balanced(counts):
    recalls = [counts["class_correct"][c] / counts["class_total"][c] for c in counts["class_total"]]
    return 100.0 * sum(recalls) / len(recalls)


def formula_symmetries(automaton, num_formula_symbols, num_symbols):
    """Permutazioni dei simboli della formula che lasciano invariato il reward per ogni sequenza
    possibile (es. in "visita pickaxe e lava" scambiare pickaxe e lava non cambia nulla): dai soli
    reward non si possono distinguere."""
    symmetries = []
    for perm in itertools.permutations(range(num_formula_symbols)):
        pi = list(perm) + list(range(num_formula_symbols, num_symbols))
        seen, frontier, equivalent = {(0, 0)}, [(0, 0)], True
        while frontier and equivalent:
            q1, q2 = frontier.pop()
            if automaton.rewards[q1] != automaton.rewards[q2]:
                equivalent = False
                break
            for sym in range(num_symbols):
                nxt = (automaton.transitions[q1][sym], automaton.transitions[q2][pi[sym]])
                if nxt not in seen:
                    seen.add(nxt)
                    frontier.append(nxt)
        if equivalent:
            symmetries.append(pi)
    return symmetries


def symbol_accuracy(sym_true, sym_pred, env):
    """accuracy su tutti i simboli e sui soli simboli della formula, a meno di cio' che dai reward non si
    puo' distinguere (come nella 4x4): gli oggetti che la formula non usa si comportano come caselle
    vuote e contano come vuoto; se la formula e' simmetrica si prende la permutazione dei simboli
    predetti piu' favorevole tra quelle simmetriche"""
    num_formula_symbols, empty = env.formula[1], NUM_SYMBOLS - 1
    reduce = np.vectorize(lambda s: s if s < num_formula_symbols else empty)
    symmetries = formula_symmetries(env.automaton, num_formula_symbols, NUM_SYMBOLS)
    acc_all = 100.0 * np.mean(sym_true == sym_pred)
    true_r = reduce(sym_true)
    acc_formula = max(100.0 * np.mean(reduce(np.array(pi)[sym_pred]) == true_r) for pi in symmetries)
    return acc_all, acc_formula, symmetries


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def split_keys(data, reward_type, rng):
    """80/10/10 come nella 4x4. three_value: tutte le traiettorie mescolate; distance: meta' a reward
    basso e meta' a reward alto, divise 80/10/10 ciascuna (il train resta bilanciato)"""
    keys = list(data)
    if reward_type == "three_value":
        rng.shuffle(keys)
        n = len(keys)
        return {"high": keys[:int(n * 0.8)], "low": [], "val": keys[int(n * 0.8):int(n * 0.9)],
                "test": keys[int(n * 0.9):]}
    keys.sort(key=lambda k: data[k]["total_reward"])
    half = len(keys) // 2
    low, high = keys[:half], keys[half:]
    rng.shuffle(low)
    rng.shuffle(high)
    split = lambda ks: (ks[:int(len(ks) * 0.8)], ks[int(len(ks) * 0.8):int(len(ks) * 0.9)], ks[int(len(ks) * 0.9):])
    (low_tr, low_va, low_te), (high_tr, high_va, high_te) = split(low), split(high)
    val, test = low_va + high_va, low_te + high_te
    rng.shuffle(val)
    rng.shuffle(test)
    return {"high": high_tr, "low": low_tr, "val": val, "test": test}


def train_batches(split, reward_type, chunk, rng):
    """gruppi di traiettorie per una epoch (come nella 4x4): con distance ogni batch e' meta' a reward
    basso e meta' alto"""
    if reward_type == "distance":
        rng.shuffle(split["low"])
        rng.shuffle(split["high"])
        half, groups = chunk // 2, []
        for s in range(0, min(len(split["low"]), len(split["high"])), half):
            group = split["low"][s:s + half] + split["high"][s:s + half]
            rng.shuffle(group)
            groups.append(group)
        if len(groups) > 1 and len(groups[-1]) == 1:
            groups[-2] += groups.pop()
        return groups
    rng.shuffle(split["high"])
    return chunk_groups(split["high"], chunk)


def train_grounder(state_type, reward_type, task, dataset, seed=0, epochs=EPOCHS, patience=PATIENCE, out_dir=None):
    """allena il grounder e lo valuta sul test; restituisce un dict con i risultati"""
    set_seed(seed)
    rng = random.Random(seed)
    env = GridWorldEnv(formulas[task - 1], state_type="symbolic", reward_type=reward_type)
    model_file = grounder_path(state_type, reward_type, task, dataset, seed)
    log_file = log_path(state_type, reward_type, task, dataset, seed)
    if out_dir is not None:
        model_file = Path(out_dir) / "grounder" / model_file.name
        log_file = Path(out_dir) / "log" / log_file.name
    model_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    h5 = dataset_path(reward_type, task, dataset)
    data = load_dataset(h5, env.rew_dictionary)
    split = split_keys(data, reward_type, rng)
    train_keys = split["low"] + split["high"]

    automa = build_automaton(env)
    classifier = Grounder(state_type).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=MIN_LR)

    # pesi per classe sul train: 1 / frequenza^potenza, normalizzati a media 1
    num_classes = len(env.rew_dictionary)
    counts = np.bincount(np.concatenate([data[k]["labels"] for k in train_keys]), minlength=num_classes)
    freq = counts / counts.sum()
    class_weights = torch.tensor([1.0 / f ** CLASS_WEIGHT_POWER if f > 0 else 1.0 for f in freq], dtype=torch.float32)
    class_weights = (class_weights / class_weights.mean()).to(device)
    inv_rew = {v: k for k, v in env.rew_dictionary.items()}

    chunk, eval_chunk = CHUNK_SIZE[state_type], EVAL_CHUNK_SIZE[state_type]
    t_start = time.time()
    with open(log_file, "w") as log:
        def write(msg):
            print(msg, flush=True)
            log.write(msg + "\n")
            log.flush()

        write(f"Dataset: {h5}")
        write(f"Config: state_type={state_type}, reward_type={reward_type}, task={task}, dataset={dataset}, "
              f"seed={seed}, epochs={epochs}, patience={patience}, lr={LR}, chunk={chunk}, class_weight_power={CLASS_WEIGHT_POWER}, "
              f"normalized_automaton=True")
        write(f"Device: {device} | parameters: {sum(p.numel() for p in classifier.parameters()):,}")
        write(f"Traiettorie train: {len(train_keys)} ({len(split['low'])} low, {len(split['high'])} high) | "
              f"val: {len(split['val'])} | test: {len(split['test'])}")
        write("Normalized automaton: True | Class weight power: 1.0 | " + " | ".join(
            f"classe {inv_rew[c]}: frequenza {100 * freq[c]:.2f}%, peso {class_weights[c]:.3f}" for c in range(num_classes)))

        best_bal, best_epoch, best_state = -1.0, 0, None
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            classifier.train()
            losses, tr_correct, tr_total = [], 0, 0
            for group in train_batches(split, reward_type, chunk, rng):
                codes, labels, mask = make_batch(data, group)
                labels, mask = labels.to(device), mask.to(device)
                optimizer.zero_grad()
                pred_rew = predict_rewards(classifier, automa, codes, mask)
                per_pos = nn.functional.nll_loss(torch.log(pred_rew.clamp_min(1e-8)).view(-1, num_classes),
                                                 labels.view(-1), reduction="none", weight=class_weights)
                m = mask.view(-1).float()
                loss = (per_pos * m).sum() / m.sum()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                tr_correct += ((pred_rew.argmax(-1) == labels) & mask).sum().item()
                tr_total += mask.sum().item()
            mean_loss = float(np.mean(losses))
            # un solo passo di scheduler per epoch, sulla loss media dell'epoch (come nella 4x4)
            scheduler.step(mean_loss)

            val = evaluate_counts(classifier, automa, data, split["val"], eval_chunk)
            acc_val, bal_val = 100.0 * val["correct"] / val["total"], balanced(val)
            if bal_val > best_bal:
                best_bal, best_epoch = bal_val, epoch
                best_state = copy.deepcopy(classifier.state_dict())
            write(f"Epoch {epoch} | Mean Loss: {mean_loss:.4f} | Train Acc: {100.0 * tr_correct / tr_total:.2f} | "
                  f"Val Acc: {acc_val:.2f} (balanced: {bal_val:.2f}) | LR: {optimizer.param_groups[0]['lr']:.6f} | "
                  f"{time.time() - t0:.1f}s")
            if epoch - best_epoch >= patience:
                write(f"Early stopping alla epoch {epoch} (nessun miglioramento sul validation da {patience} epoch)")
                break

        classifier.load_state_dict(best_state)
        write(f"\nBEST EPOCH (su validation, balanced): {best_epoch} - Val Acc: {best_bal:.2f}%")
        tmp = model_file.with_suffix(".pth.tmp")
        torch.save(classifier.state_dict(), tmp)
        os.replace(tmp, model_file)

        # test finale
        test = evaluate_counts(classifier, automa, data, split["test"], eval_chunk)
        acc_test, bal_test = 100.0 * test["correct"] / test["total"], balanced(test)
        recall = {c: 100.0 * test["class_correct"][c] / test["class_total"][c] for c in sorted(test["class_total"])}
        sym_all, sym_formula, symmetries = symbol_accuracy(test["sym_true"], test["sym_pred"], env)
        names = env.dictionary_symbols
        write(f"\nACCURATEZZA FINALE GLOBALE SUL TEST SET: {acc_test:.2f}%")
        write(f"BALANCED ACCURACY FINALE GLOBALE SUL TEST SET: {bal_test:.2f}%")
        write("RECALL PER CLASSE SUL TEST SET: " + " | ".join(
            f"classe {inv_rew[c]}: {r:.2f}% ({test['class_total'][c]} passi)" for c, r in recall.items()))
        write(f"SYMBOL ACCURACY SUL TEST SET: {sym_all:.2f}%")
        write(f"SYMBOL ACCURACY SUL TEST SET (simboli della formula, oggetti non usati = vuoto, a meno delle "
              f"simmetrie della formula): {sym_formula:.2f}%")
        swaps = ["(" + " ".join(f"{names[a]}->{names[b]}" for a, b in enumerate(pi) if a != b) + ")"
                 for pi in symmetries if pi != sorted(pi)]
        write(f"SIMMETRIE DELLA FORMULA: {', '.join(swaps) if swaps else 'nessuna'}")
        conf = np.zeros((NUM_SYMBOLS, NUM_SYMBOLS), dtype=np.int64)
        np.add.at(conf, (test["sym_true"], test["sym_pred"]), 1)
        write("MATRICE DI CONFUSIONE DEI SIMBOLI SUL TEST SET (righe = vero, colonne = predetto, "
              f"{'/'.join(names)}; passi):")
        for i in range(NUM_SYMBOLS):
            write(f"  {names[i]}: " + " ".join(f"{n:7d}" for n in conf[i]))
        write(f"Training time: {(time.time() - t_start) / 60:.1f} min")

    return {"balanced": bal_test, "accuracy": acc_test, "symbol_all": sym_all, "symbol_formula": sym_formula,
            "best_epoch": best_epoch, "model": model_file}
