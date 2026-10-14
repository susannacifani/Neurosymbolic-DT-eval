import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from statistics import mean

from .NN_models import ActorCritic, RNN, Net
from .NRM.NeuralRewardMachine import NeuralRewardMachine
from .NRM.utils import eval_acceptance
use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")
print(device)
torch.autograd.set_detect_anomaly(False)

from pathlib import Path
import h5py

# cartella NRM: i dataset stanno in NRM/Dataset/<State>/<reward>/task<N>/
ROOT = Path(__file__).resolve().parents[2]
import random
import copy
import itertools

# max number of episodes
max_episodes   = 10000
rnn_outputs = 5
num_layers  = 2
hidden_size = 120 
rnn_hidden_size = 50 
slide_wind = 100
lr=0.0004
num_steps = 5
TT_policy = 5
TT_grounder = 40 
grounder_epochs = 100
TTT = 10

def trajectory_labels(grp, rew_dictionary):
    """Classe di reward di ogni osservazione della traiettoria, compresa quella finale.
    rewards[t] e' il delta ottenuto con l'azione t, quindi il potenziale dello stato osservato
    a t e' la somma dei reward fino a t-1; l'osservazione finale (dopo l'ultima azione) ha il
    potenziale finale, che e' l'unico punto in cui compaiono vittoria e fallimento."""
    potenziali = np.concatenate([[0.0], np.cumsum(grp['rewards'][:])])
    return [rew_dictionary[min(rew_dictionary, key=lambda v: abs(v - p))] for p in potenziali]


def load_trajectory(grp, state_type, rew_dictionary):
    obs_np = np.concatenate([grp['observations'][:], grp['final_observation'][:][None]])
    if state_type == "symbolic":
        traj = list(torch.tensor(obs_np, dtype=torch.float32))
    else:
        traj = [torch.tensor(frame, dtype=torch.float32) for frame in obs_np]
    return traj, trajectory_labels(grp, rew_dictionary)


def build_cache(f, keys, rew_dictionary):
    """Legge dal file una sola volta tutte le traiettorie indicate.
    Le osservazioni distinte sono pochissime (una per casella della griglia), quindi ognuna
    viene salvata una volta sola in 'frames' e per ogni traiettoria si tiene solo l'indice
    dell'osservazione a ogni passo (compresa quella finale) e le etichette di reward.
    Gli input della rete restano identici pixel per pixel a quelli letti dal file."""
    index_of, frames, cache = {}, [], {}
    for k in keys:
        grp = f[k]
        obs_np = np.concatenate([grp['observations'][:], grp['final_observation'][:][None]]).astype(np.float32)
        idx = []
        for o in obs_np:
            key = o.tobytes()
            if key not in index_of:
                index_of[key] = len(frames)
                frames.append(torch.from_numpy(o.copy()))
            idx.append(index_of[key])
        cache[k] = (idx, trajectory_labels(grp, rew_dictionary))
    return torch.stack(frames), cache


def make_batch(frames, cache, keys, max_len):
    """Batch (traiettorie, max_len, ...) di osservazioni, con etichette e lunghezze reali.
    Il padding ripete l'ultimo passo reale, come prima."""
    idx, labels, lengths = [], [], []
    for k in keys:
        traj_idx, traj_labels = cache[k]
        L = len(traj_idx)
        lengths.append(L)
        idx.append(traj_idx + [traj_idx[-1]] * (max_len - L))
        labels.append(traj_labels + [traj_labels[-1]] * (max_len - L))
    return frames[torch.tensor(idx)], labels, lengths


def formula_symmetries(automaton, num_formula_symbols, num_symbols):
    """Permutazioni dei simboli della formula che lasciano invariato il reward per ogni
    sequenza possibile (es. in "visita pickaxe e lava" scambiare pickaxe e lava non cambia
    nulla). Da soli reward questi scambi non si possono distinguere. Si verifica con una
    visita in ampiezza sulle coppie di stati (automa originale, automa con simboli permutati)."""
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


def symbol_grounding_eval(classifier, frames, cache, keys, env, state_type):
    """Symbol accuracy: per ogni passo reale delle traiettorie 'keys' confronta il simbolo
    predetto dal grounder con il simbolo vero della casella (preso dall'ambiente).
    Restituisce:
    - accuracy su tutti i simboli
    - accuracy sui soli simboli della formula, a meno di cio' che dai reward non si puo'
      distinguere: gli oggetti che la formula non usa si comportano come caselle vuote e
      vengono contati come vuoto, e se la formula e' simmetrica (es. pickaxe <-> lava) si
      prende la permutazione dei simboli predetti piu' favorevole tra quelle simmetriche
    - la mappa casella -> (simbolo vero, simbolo predetto)
    - le simmetrie della formula trovate"""
    cell_of = {}
    for x in range(env.size):
        for y in range(env.size):
            obs = env.image_locations[x, y] if state_type == "image" else np.array([x, y])
            cell_of[np.asarray(obs, dtype=np.float32).tobytes()] = (x, y)

    saved_location = getattr(env, "_agent_location", None)
    cells, true_sym = [], []
    for frame in frames:
        x, y = cell_of[frame.numpy().tobytes()]
        env._agent_location = np.array([x, y])
        cells.append((x, y))
        true_sym.append(env._current_symbol())
    env._agent_location = saved_location

    classifier.eval()
    with torch.no_grad():
        pred_sym = classifier(frames.float().to(device)).argmax(dim=-1).cpu().tolist()

    num_formula_symbols = env.formula[1]
    num_symbols = len(env.dictionary_symbols)
    empty = num_symbols - 1
    reduce = lambda sym: sym if sym < num_formula_symbols else empty
    symmetries = formula_symmetries(env.automaton, num_formula_symbols, num_symbols)
    total = correct_all = 0
    correct_formula = [0] * len(symmetries)
    for k in keys:
        for i in cache[k][0]:
            total += 1
            correct_all += pred_sym[i] == true_sym[i]
            for j, pi in enumerate(symmetries):
                correct_formula[j] += reduce(pi[pred_sym[i]]) == reduce(true_sym[i])
    symbol_map = {cells[i]: (true_sym[i], pred_sym[i]) for i in range(len(frames))}
    return 100.0 * correct_all / total, 100.0 * max(correct_formula) / total, symbol_map, symmetries


def recurrent_A2C(task_id, dataset_el, state_type, reward_type, env, path, experiment, method, feature_extraction, epochs=100,
                  class_weight_power=1.0, normalized_automaton=True):

    num_outputs = env.action_space.n
    params = []
    
    if feature_extraction:
        cnn = Net().to(device)
        cnn.float()
        CNN_output_size = 16
        num_inputs = CNN_output_size
        params += list(cnn.parameters())
    else:
        num_inputs = env.state_space_size

    if method == "rnn":
        rnn=RNN(num_inputs, rnn_hidden_size, num_layers).to(device)
        rnn.float()
        params += list(rnn.parameters())
    elif method == "nrm":
        num_of_states, num_of_symbols, num_automaton_outputs, transition_function, automaton_rewards = env.get_automaton_specs()

        if env.state_type == "symbolic":
            dataset = "minecraft_location"
        elif env.state_type == "image":
            dataset = "minecraft_image"
            
        grounder = NeuralRewardMachine(num_of_states, num_of_symbols, num_automaton_outputs, 
                                    num_exp=experiment, log_dir=path, dataset=dataset)
        grounder.deepAutoma.initFromDfa(transition_function, automaton_rewards)
        grounder.deepAutoma.normalized = normalized_automaton
        grounder.deepAutoma.float()
        grounder.deepAutoma.to(device)
        grounder.classifier.float()
        grounder.classifier.to(device)

        print("Caricamento dataset...")

        if state_type == "image":
            folder_state = "Image"
        elif state_type == "symbolic":
            folder_state = "Symbolic"
            
        if reward_type == "three_value":
            folder_reward = "three_value"
        elif reward_type == "distance":
            folder_reward = "distance"
            
        folder_task = f"task{task_id+1}"
        exp = dataset_el
        
        file = f'dataset_{exp}.h5'
        H5_PATH = ROOT / "Dataset" / folder_state / folder_reward / folder_task / file
        print(f"Pesco dataset da: ", H5_PATH)
        # stessa organizzazione del DT: path/log/ per i log, path/grounder/ per i modelli
        log_file_path = Path(path) / "log" / f"log_{exp}.txt"
        path_model = Path(path) / "grounder" / f"grounder_{exp}.pth"
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        path_model.parent.mkdir(parents=True, exist_ok=True)
        open(log_file_path, "w").close()

        with h5py.File(H5_PATH, 'r') as f:
            all_keys = list(f.keys())
            print(f"Dataset originale: {len(all_keys)} traiettorie.")

            # =========================================================
            # FILTRAGGIO E SPLIT 80/10/10 (TRAIN / VALIDATION / TEST) BILANCIATO
            # Train: usato per aggiornare i pesi.
            # Validation: mai usato per i pesi, solo per scegliere la epoch migliore.
            # Test: toccato una volta sola, alla fine, per il numero finale.
            # =========================================================
            if folder_reward == "three_value":
                # si usano tutte le traiettorie, anche quelle senza esito (reward totale 0):
                # dicono al grounder quali caselle NON fanno scattare vittoria o fallimento.
                # Lo sbilanciamento tra classi e' gestito dai pesi per classe nella loss.
                chiavi_filtrate = list(all_keys)

                random.shuffle(chiavi_filtrate)

                n = len(chiavi_filtrate)
                split_train = int(n * 0.8)
                split_val = int(n * 0.9)
                train_keys = chiavi_filtrate[:split_train]
                val_keys = chiavi_filtrate[split_train:split_val]
                test_keys = chiavi_filtrate[split_val:]

                print(f"Dataset Sparse 3-Value:")
                print(f"- Train Set: {len(train_keys)} traiettorie")
                print(f"- Validation Set: {len(val_keys)} traiettorie")
                print(f"- Test Set: {len(test_keys)} traiettorie")

                high_keys = train_keys
                low_keys = []

            elif folder_reward == "distance":
                chiavi_con_reward = []
                for k in all_keys:
                    reward_totale = np.sum(f[k]['rewards'][:])
                    chiavi_con_reward.append((k, reward_totale))

                chiavi_con_reward.sort(key=lambda x: x[1])
                meta = len(chiavi_con_reward) // 2

                low_keys_full = [x[0] for x in chiavi_con_reward[:meta]]
                high_keys_full = [x[0] for x in chiavi_con_reward[meta:]]

                random.shuffle(low_keys_full)
                random.shuffle(high_keys_full)

                n_low = len(low_keys_full)
                n_high = len(high_keys_full)
                split_train_low, split_val_low = int(n_low * 0.8), int(n_low * 0.9)
                split_train_high, split_val_high = int(n_high * 0.8), int(n_high * 0.9)

                low_keys = low_keys_full[:split_train_low]
                high_keys = high_keys_full[:split_train_high]

                val_keys = low_keys_full[split_train_low:split_val_low] + high_keys_full[split_train_high:split_val_high]
                random.shuffle(val_keys)

                test_keys = low_keys_full[split_val_low:] + high_keys_full[split_val_high:]
                random.shuffle(test_keys)

                print(f"Dataset Dense (Distance) Bilanciato:")
                print(f"- Train Set: {len(low_keys) + len(high_keys)} traiettorie ({len(low_keys)} Low, {len(high_keys)} High)")
                print(f"- Validation Set: {len(val_keys)} traiettorie")
                print(f"- Test Set: {len(test_keys)} traiettorie")

            keys_for_padding = low_keys + high_keys + val_keys + test_keys
            # +1: in coda a ogni traiettoria c'e' anche l'osservazione finale
            max_len = max([f[k].attrs['length'] for k in keys_for_padding]) + 1
            print(f"Padding impostato a {max_len} step.")

            # lettura unica dal file di tutte le traiettorie usate (train, validation e test)
            frames, cache = build_cache(f, keys_for_padding, env.rew_dictionary)
            print(f"Osservazioni distinte nel dataset: {len(frames)}")

            # class weighting per compensare classi di reward sbilanciate (es. three_value,
            # dove "vittoria" e' un evento raro): peso = 1/frequenza^class_weight_power
            # (0.5 = 1/sqrt(frequenza), 1.0 = 1/frequenza), normalizzato in modo che il peso
            # medio resti 1 (non altera la scala complessiva della loss)
            class_counts = {}
            for k in (low_keys + high_keys):
                for c in cache[k][1]:
                    class_counts[c] = class_counts.get(c, 0) + 1

            num_classes = len(env.rew_dictionary)
            total_frames = sum(class_counts.values())
            class_weights = torch.ones(num_classes, dtype=torch.float32)
            for c in range(num_classes):
                freq = class_counts.get(c, 0) / total_frames if total_frames > 0 else 1.0 / num_classes
                class_weights[c] = 1.0 / (freq ** class_weight_power) if freq > 0 else 1.0
            class_weights = class_weights / class_weights.mean()
            grounder.class_weights = class_weights
            print(f"Class weights (per compensare sbilanciamento): {class_weights.tolist()}")
            inv_rew = {v: k for k, v in env.rew_dictionary.items()}
            with open(log_file_path, "a") as f_log:
                f_log.write(f"Normalized automaton: {normalized_automaton} | Class weight power: {class_weight_power} | "
                            + " | ".join(f"classe {inv_rew[c]}: frequenza {100 * class_counts.get(c, 0) / max(total_frames, 1):.2f}%, peso {class_weights[c]:.3f}"
                                         for c in range(num_classes)) + "\n")

            # Batch di validation costruito una sola volta (le chiavi non cambiano tra le epoch)
            val_image_traj, val_rew_traj, val_lengths = make_batch(frames, cache, val_keys, max_len)

            EPOCHS = epochs
            if state_type == "symbolic":
                CHUNK_SIZE = 200
            else: 
                CHUNK_SIZE = 100
            HALF_CHUNK = CHUNK_SIZE // 2 

            # ---------------------------------------------------------
            # TRAINING LOOP
            # ---------------------------------------------------------
            best_acc = -1.0
            best_classifier_state = None

            for epoch in range(1, EPOCHS + 1):
                print(f"\n================ EPOCH {epoch}/{EPOCHS} ================")
                loss_di_questa_epoca = []

                # if folder_reward == "distance":
                #     random.shuffle(low_keys)
                #     random.shuffle(high_keys)
                #     num_batches = min(len(low_keys), len(high_keys)) // HALF_CHUNK
                # else: 
                #     random.shuffle(high_keys)
                #     num_batches = len(high_keys) // CHUNK_SIZE

                # for i in range(num_batches):
                #     if folder_reward == "distance":
                #         batch_low = low_keys[i * HALF_CHUNK : (i + 1) * HALF_CHUNK]
                #         batch_high = high_keys[i * HALF_CHUNK : (i + 1) * HALF_CHUNK]
                #         keys_chunk = batch_low + batch_high
                #         random.shuffle(keys_chunk) 
                #     else:
                #         keys_chunk = high_keys[i * CHUNK_SIZE : (i + 1) * CHUNK_SIZE]
                    

                if folder_reward == "distance":
                    random.shuffle(low_keys)
                    random.shuffle(high_keys)
                    batch_key_groups = []
                    max_pairs = min(len(low_keys), len(high_keys))

                    for start_idx in range(0, max_pairs, HALF_CHUNK):
                        batch_low = low_keys[start_idx : start_idx + HALF_CHUNK]
                        batch_high = high_keys[start_idx : start_idx + HALF_CHUNK]
                        keys_chunk = batch_low + batch_high

                        if keys_chunk:
                            random.shuffle(keys_chunk)
                            batch_key_groups.append(keys_chunk)
                else:
                    random.shuffle(high_keys)
                    batch_key_groups = [
                        high_keys[start_idx : start_idx + CHUNK_SIZE]
                        for start_idx in range(0, len(high_keys), CHUNK_SIZE)
                        if high_keys[start_idx : start_idx + CHUNK_SIZE]
                    ]

                if not batch_key_groups:
                    raise ValueError(
                        f"Nessun batch costruibile per task {task_id + 1}, dataset '{dataset_el}', "
                        f"reward_type '{reward_type}'. Train trajectories disponibili: "
                        f"{len(high_keys) if folder_reward != 'distance' else min(len(low_keys), len(high_keys))}."
                    )

                for keys_chunk in batch_key_groups:

                    

                    image_traj, rew_traj, lengths = make_batch(frames, cache, keys_chunk, max_len)

                    grounder.set_dataset(image_traj, rew_traj, lengths=lengths)
                    batch_loss = grounder.train_symbol_grounding(num_of_epochs=1)
                    loss_di_questa_epoca.append(float(batch_loss))

                mean_epoch_loss = sum(loss_di_questa_epoca) / len(loss_di_questa_epoca)
                # un solo step di scheduler per epoch reale, sulla loss media dell'epoch
                # (train_symbol_grounding viene chiamata una volta per chunk, non per epoch)
                grounder.sheduler.step(mean_epoch_loss)
                # accuracy calcolata solo sugli step reali (il padding e' escluso a monte,
                # dentro eval_acceptance, tramite le lengths)
                # qui acc_train riflette solo l'ultimo chunk di training usato in questa epoch
                # (serve solo come indicazione di quanto il modello si adatta ai dati appena visti)
                acc_train, _, bal_train = grounder.eval_all(automa_implementation='logic_circuit', temperature=1, return_balanced=True)

                # valutazione sul validation set: mai usato per aggiornare i pesi, serve solo
                # per scegliere la epoch migliore in modo onesto (non sui dati di training)
                grounder.set_dataset(val_image_traj, val_rew_traj, lengths=val_lengths)
                acc_val, _, bal_val = grounder.eval_all(automa_implementation='logic_circuit', temperature=1, return_balanced=True)

                # tiene il checkpoint del classifier alla epoch con la miglior validation accuracy:
                # serve perche' un crollo di loss a meta' training puo' non riprendersi piu' entro
                # la fine delle epoch, e altrimenti il modello valutato/salvato sarebbe quello
                # dell'ultima epoch qualunque cosa sia successo nel mezzo.
                # Si usa la balanced accuracy (non quella grezza) per scegliere l'epoch migliore:
                # con classi di reward sbilanciate (es. three_value) l'accuracy grezza premia un
                # modello che ignora la classe rara, mentre la balanced accuracy no.
                if bal_val > best_acc:
                    best_acc = bal_val
                    best_classifier_state = copy.deepcopy(grounder.classifier.state_dict())

                current_lr = grounder.optimizer.param_groups[0]['lr']

                with open(log_file_path, "a") as f_log:
                    f_log.write(f"Epoch {epoch} | Mean Loss: {mean_epoch_loss:.4f} | Train Acc: {acc_train:.2f} (balanced: {bal_train:.2f}) | Val Acc: {acc_val:.2f} (balanced: {bal_val:.2f}) | LR: {current_lr:.6f}\n")
                print(f"--> Epoch {epoch} | Loss: {mean_epoch_loss:.4f} | Train: {acc_train:.2f}% (bal: {bal_train:.2f}%) | Val: {acc_val:.2f}% (bal: {bal_val:.2f}%) | LR: {current_lr:.6f}")

        if best_classifier_state is not None:
            grounder.classifier.load_state_dict(best_classifier_state)
            print(f"\n=== Ripristinato il classifier della epoch migliore su validation (Val Acc balanced: {best_acc:.2f}%) ===")
            with open(log_file_path, "a") as f_log:
                f_log.write(f"\nBEST EPOCH (su validation, balanced) - Val Acc: {best_acc:.2f}%\n")

        torch.save(grounder.classifier.state_dict(), path_model)
        print(f"\n=== Modello salvato con successo in {path_model} ===")

        # =================================================================
        # TEST FINALE GLOBALE (FASE 6) SUL TEST SET (10%)
        # =================================================================
        print("\n=== Avvio Valutazione Finale Globale ===")
        # conteggi aggregati su tutti i blocchi del test: accuracy e recall per classe esatte
        # anche quando l'ultimo blocco e' piu' piccolo degli altri
        test_correct, test_total = 0, 0
        class_correct, class_total = {}, {}

        if state_type == "symbolic":
            EVAL_CHUNK_SIZE = 500 
        else:
            EVAL_CHUNK_SIZE = 100 

        print(f"Valutazione sul Test Set (Dati mai visti): {len(test_keys)} traiettorie")
        for i in range(0, len(test_keys), EVAL_CHUNK_SIZE):
            keys_chunk = test_keys[i : i + EVAL_CHUNK_SIZE]
            eval_image_traj, eval_rew_traj, eval_lengths = make_batch(frames, cache, keys_chunk, max_len)

            grounder.set_dataset(eval_image_traj, eval_rew_traj, lengths=eval_lengths)
            counts = grounder.eval_counts(automa_implementation='logic_circuit', temperature=1)
            test_correct += counts["correct"]
            test_total += counts["total"]
            for c, n in counts["class_total"].items():
                class_total[c] = class_total.get(c, 0) + n
                class_correct[c] = class_correct.get(c, 0) + counts["class_correct"][c]
            print(f"Valutato blocco {i} - {i+len(keys_chunk)}: {100 * counts['correct'] / counts['total']:.2f}%")

        accuratezza_globale = 100.0 * test_correct / test_total if test_total > 0 else 0.0
        recall_per_classe = {c: 100.0 * class_correct[c] / class_total[c] for c in sorted(class_total)}
        balanced_accuratezza_globale = sum(recall_per_classe.values()) / len(recall_per_classe) if recall_per_classe else 0.0
        inv_rew = {v: k for k, v in env.rew_dictionary.items()}

        print(f"\nACCURATEZZA FINALE SUL TEST SET: {accuratezza_globale:.2f}% (balanced: {balanced_accuratezza_globale:.2f}%)")

        with open(log_file_path, "a") as f_log:
            f_log.write(f"\nACCURATEZZA FINALE GLOBALE SUL TEST SET: {accuratezza_globale:.2f}%\n")
            f_log.write(f"BALANCED ACCURACY FINALE GLOBALE SUL TEST SET: {balanced_accuratezza_globale:.2f}%\n")
            f_log.write("RECALL PER CLASSE SUL TEST SET: " + " | ".join(
                f"classe {inv_rew[c]}: {r:.2f}% ({class_total[c]} passi)" for c, r in recall_per_classe.items()) + "\n")

        # symbol accuracy: il grounder riconosce il simbolo giusto in ogni casella?
        sym_acc_all, sym_acc_formula, symbol_map, symmetries = symbol_grounding_eval(
            grounder.classifier, frames, cache, test_keys, env, state_type)
        names = env.dictionary_symbols
        print(f"SYMBOL ACCURACY SUL TEST SET: {sym_acc_all:.2f}% (simboli della formula: {sym_acc_formula:.2f}%)")
        with open(log_file_path, "a") as f_log:
            f_log.write(f"SYMBOL ACCURACY SUL TEST SET: {sym_acc_all:.2f}%\n")
            f_log.write(f"SYMBOL ACCURACY SUL TEST SET (simboli della formula, oggetti non usati = vuoto, a meno delle simmetrie della formula): {sym_acc_formula:.2f}%\n")
            swaps = ["(" + " ".join(f"{names[a]}->{names[b]}" for a, b in enumerate(pi) if a != b) + ")" for pi in symmetries if pi != sorted(pi)]
            f_log.write(f"SIMMETRIE DELLA FORMULA: {', '.join(swaps) if swaps else 'nessuna'}\n")
            f_log.write(f"MAPPA DEI SIMBOLI (vero->predetto, {'/'.join(names)}; righe x=0..{env.size - 1}, colonne y=0..{env.size - 1}):\n")
            for x in range(env.size):
                f_log.write("  " + "  ".join(f"{names[symbol_map[x, y][0]]}->{names[symbol_map[x, y][1]]}" for y in range(env.size)) + "\n")
