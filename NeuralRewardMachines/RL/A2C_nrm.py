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
import random
import copy

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

def recurrent_A2C(task_id, dataset_el, state_type, reward_type, env, path, experiment, method, feature_extraction):

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
        H5_PATH = Path("Resultss") / folder_state / folder_reward / folder_task / "datasets" / file
        print(f"Pesco dataset da: ", H5_PATH)
        log_file_path = Path(path) / f"log_{exp}.txt"
        path_model = Path(path) / f"grounder_{exp}.pth"
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
                chiavi_filtrate = []
                for k in all_keys:
                    if np.sum(f[k]['rewards'][:]) != 0:
                        chiavi_filtrate.append(k)

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
            max_len = max([f[k].attrs['length'] for k in keys_for_padding])
            print(f"Padding impostato a {max_len} step.")

            # class weighting per compensare classi di reward sbilanciate (es. three_value,
            # dove "vittoria" e' un evento rarissimo): peso = 1/sqrt(frequenza), normalizzato
            # in modo che il peso medio resti 1 (non altera la scala complessiva della loss)
            class_counts = {}
            for k in (low_keys + high_keys):
                rew_np = f[k]['rewards'][:]
                potenziali = np.cumsum(rew_np) - rew_np
                for p in potenziali:
                    c = env.rew_dictionary[min(env.rew_dictionary, key=lambda v: abs(v - p))]
                    class_counts[c] = class_counts.get(c, 0) + 1

            num_classes = len(env.rew_dictionary)
            total_frames = sum(class_counts.values())
            class_weights = torch.ones(num_classes, dtype=torch.float32)
            for c in range(num_classes):
                freq = class_counts.get(c, 0) / total_frames if total_frames > 0 else 1.0 / num_classes
                class_weights[c] = 1.0 / (freq ** 0.5) if freq > 0 else 1.0
            class_weights = class_weights / class_weights.mean()
            grounder.class_weights = class_weights
            print(f"Class weights (per compensare sbilanciamento): {class_weights.tolist()}")

            # Batch di validation costruito una sola volta (le chiavi non cambiano tra le epoch)
            val_image_traj, val_rew_traj, val_lengths = [], [], []
            for k in val_keys:
                obs_np = f[k]['observations'][:]
                rew_np = f[k]['rewards'][:]

                if state_type == "symbolic":
                    curr_traj = list(torch.tensor(obs_np, dtype=torch.float32))
                else:
                    curr_traj = [torch.tensor(frame, dtype=torch.float32) for frame in obs_np]

                potenziali = np.cumsum(rew_np) - rew_np
                curr_rew = [env.rew_dictionary[min(env.rew_dictionary, key=lambda v: abs(v - p))] for p in potenziali]

                L = len(curr_traj)
                val_lengths.append(L)
                if L < max_len:
                    curr_traj.extend([curr_traj[-1]] * (max_len - L))
                    curr_rew.extend([curr_rew[-1]] * (max_len - L))

                val_image_traj.append(curr_traj)
                val_rew_traj.append(curr_rew)

            EPOCHS = 100
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

                    

                    image_traj = []
                    rew_traj = []
                    lengths = []

                    for k in keys_chunk:
                        # Lettura diretta da HDF5 (No Cache) per preservare la RAM
                        obs_np = f[k]['observations'][:]
                        rew_np = f[k]['rewards'][:]

                        if state_type == "symbolic":
                            tensore_intero = torch.tensor(obs_np, dtype=torch.float32)
                            curr_traj = list(tensore_intero)
                        else:
                            # Ripristinato caricamento conservativo frame-per-frame
                            curr_traj = [torch.tensor(frame, dtype=torch.float32) for frame in obs_np]

                        # cumsum esclusiva: rewards[t] e' il delta verso lo stato t+1 (vedi dataset_generation.py),
                        # quindi il potenziale dello stato osservato a t e' la somma dei reward FINO a t-1
                        potenziali = np.cumsum(rew_np) - rew_np
                        curr_rew = [env.rew_dictionary[min(env.rew_dictionary, key=lambda v: abs(v - p))] for p in potenziali]

                        L = len(curr_traj)
                        lengths.append(L)
                        if L < max_len:
                            curr_traj.extend([curr_traj[-1]] * (max_len - L))
                            curr_rew.extend([curr_rew[-1]] * (max_len - L))

                        image_traj.append(curr_traj)
                        rew_traj.append(curr_rew)

                    grounder.set_dataset(image_traj, rew_traj, lengths=lengths)
                    batch_loss = grounder.train_symbol_grounding(num_of_epochs=1)
                    loss_di_questa_epoca.append(float(batch_loss))

                    del image_traj, rew_traj
                    import gc; gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

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
        accuratezze_finali = []
        balanced_accuratezze_finali = []

        with h5py.File(H5_PATH, 'r') as f_test:
            if state_type == "symbolic":
                EVAL_CHUNK_SIZE = 500 
            else:
                EVAL_CHUNK_SIZE = 100 
            
            print(f"Valutazione sul Test Set (Dati mai visti): {len(test_keys)} traiettorie")
            for i in range(0, len(test_keys), EVAL_CHUNK_SIZE):
                keys_chunk = test_keys[i : i + EVAL_CHUNK_SIZE]
                
                eval_image_traj = []
                eval_rew_traj = []
                eval_lengths = []

                for k in keys_chunk:
                    # Lettura diretta da disco
                    obs_np = f_test[k]['observations'][:]
                    rew_np = f_test[k]['rewards'][:]

                    if state_type == "symbolic":
                        curr_traj_tensor = torch.tensor(obs_np, dtype=torch.float32)
                        curr_traj = list(curr_traj_tensor)
                    else:
                        # Ripristinato caricamento conservativo frame-per-frame
                        curr_traj = [torch.tensor(frame, dtype=torch.float32) for frame in obs_np]

                    potenziali = np.cumsum(rew_np) - rew_np
                    curr_rew = [env.rew_dictionary[min(env.rew_dictionary, key=lambda v: abs(v - p))] for p in potenziali]

                    L = len(curr_traj)
                    eval_lengths.append(L)
                    if L < max_len:
                        curr_traj.extend([curr_traj[-1]] * (max_len - L))
                        curr_rew.extend([curr_rew[-1]] * (max_len - L))

                    eval_image_traj.append(curr_traj)
                    eval_rew_traj.append(curr_rew)

                grounder.set_dataset(eval_image_traj, eval_rew_traj, lengths=eval_lengths)
                acc_chunk, _, bal_chunk = grounder.eval_all(automa_implementation='logic_circuit', temperature=1, return_balanced=True)
                accuratezze_finali.append(acc_chunk)
                balanced_accuratezze_finali.append(bal_chunk)
                print(f"Valutato blocco {i} - {i+len(keys_chunk)}: {acc_chunk:.2f}% (balanced: {bal_chunk:.2f}%)")

                del eval_image_traj, eval_rew_traj
                import gc; gc.collect()

        if len(accuratezze_finali) > 0:
            accuratezza_globale = sum(accuratezze_finali) / len(accuratezze_finali)
            balanced_accuratezza_globale = sum(balanced_accuratezze_finali) / len(balanced_accuratezze_finali)
        else:
            accuratezza_globale = 0.0
            balanced_accuratezza_globale = 0.0

        print(f"\nACCURATEZZA FINALE SUL TEST SET: {accuratezza_globale:.2f}% (balanced: {balanced_accuratezza_globale:.2f}%)")

        with open(log_file_path, "a") as f_log:
            f_log.write(f"\nACCURATEZZA FINALE GLOBALE SUL TEST SET: {accuratezza_globale:.2f}%\n")
            f_log.write(f"BALANCED ACCURACY FINALE GLOBALE SUL TEST SET: {balanced_accuratezza_globale:.2f}%\n")
