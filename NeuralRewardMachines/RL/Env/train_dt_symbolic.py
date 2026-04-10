import torch
import torch.nn as nn
import numpy as np
import h5py
import random
from torch.utils.data import Dataset, DataLoader
import os
from utils import setup_folders

# --- 1. DATASET ---
class DTDataset(Dataset):
    def __init__(self, h5_path, keys, K=10):
        self.K = K
        self.data = []
        
        with h5py.File(h5_path, 'r') as f:
            for key in keys:
                grp = f[key]
                traj_len = grp.attrs['length']
                
                # Normalized coordinates
                pos = grp['observations'][:].astype(np.float32) / 3.0
                
                obs = pos
                
                acts = grp['actions'][:]
                rtgs = grp['returns_to_go'][:].astype(np.float32) / 100.0
                
                self.data.append({
                    'length': traj_len,
                    'observations': obs,
                    'actions': acts,
                    'returns_to_go': rtgs
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        traj = self.data[idx]
        traj_len = traj['length']
        
        start_idx = np.random.randint(0, traj_len - self.K + 1) if traj_len > self.K else 0
        end_idx = min(start_idx + self.K, traj_len)
        
        states = traj['observations'][start_idx:end_idx]
        actions = traj['actions'][start_idx:end_idx]
        rtgs = traj['returns_to_go'][start_idx:end_idx]
        
        current_k = len(actions)
        timesteps = np.arange(start_idx, start_idx + current_k)
        
        pad_len = self.K - current_k
        if pad_len > 0:
            states = np.pad(states, ((pad_len, 0), (0, 0)), mode='constant')
            actions = np.pad(actions, (pad_len, 0), mode='constant')
            rtgs = np.pad(rtgs, (pad_len, 0), mode='constant')
            timesteps = np.pad(timesteps, (pad_len, 0), mode='constant')
            attention_mask = np.concatenate([np.zeros(pad_len), np.ones(current_k)])
        else:
            attention_mask = np.ones(self.K)

        return {
            'states': torch.tensor(states, dtype=torch.float32), 
            'actions': torch.tensor(actions, dtype=torch.long),
            'rtgs': torch.tensor(rtgs, dtype=torch.float32),
            'timesteps': torch.tensor(timesteps, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.bool)
        }

# --- 2. DT Model ---
class DecisionTransformer(nn.Module):
    def __init__(self, state_dim, action_dim=4, hidden_size=128, max_ep_len=100):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Embeddings
        self.embed_state = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh()
        )
        self.embed_rtg = nn.Linear(1, hidden_size)
        self.embed_action = nn.Embedding(action_dim, hidden_size)
        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, 
            nhead=4, 
            dim_feedforward=4*hidden_size, 
            dropout=0.1, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.predict_action = nn.Linear(hidden_size, action_dim)

    def forward(self, states, actions, rtgs, timesteps):
        B, T, C = states.shape 
        
        state_emb = self.embed_state(states)
        time_emb = self.embed_timestep(timesteps)
        
        state_emb = state_emb + time_emb
        action_emb = self.embed_action(actions) + time_emb
        rtg_emb = self.embed_rtg(rtgs.unsqueeze(-1)) + time_emb
        
        # Stack: (R1, S1, A1, R2, S2, A2...)
        stacked = torch.stack((rtg_emb, state_emb, action_emb), dim=2).view(B, 3 * T, self.hidden_size)
        
        # Maschera causale per impedire di guardare al futuro
        mask = torch.triu(torch.ones(3 * T, 3 * T), diagonal=1).to(states.device).bool()
        
        out = self.transformer(stacked, mask=mask)
        return self.predict_action(out[:, 1::3])

# --- 3. TRAINING LOOP ---
def train(dt_name, h5_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {h5_path} -> {device}")
    
    with h5py.File(h5_path, 'r') as f:
        keys = list(f.keys())
        state_dim = 2
        print(f"Detected State Dim: {state_dim}")

    random.shuffle(keys)
    split = int(0.8 * len(keys))
    train_keys, val_keys = keys[:split], keys[split:]

    train_loader = DataLoader(DTDataset(h5_path, train_keys), batch_size=64, shuffle=True)
    val_loader = DataLoader(DTDataset(h5_path, val_keys), batch_size=64, shuffle=False)
    
    model = DecisionTransformer(state_dim=state_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    patience, counter = 10, 0

    for epoch in range(500):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            
            states = batch['states'].to(device)
            actions = batch['actions'].to(device)
            rtgs = batch['rtgs'].to(device)
            timesteps = batch['timesteps'].to(device)
            mask = batch['attention_mask'].to(device)
            
            logits = model(states, actions, rtgs, timesteps)
            
            loss = loss_fn(logits[mask], actions[mask])
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validazione
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                logits = model(batch['states'].to(device), batch['actions'].to(device), 
                               batch['rtgs'].to(device), batch['timesteps'].to(device))
                mask = batch['attention_mask'].to(device)
                val_loss += loss_fn(logits[mask], batch['actions'].to(device)[mask]).item()
        
        avg_val = val_loss / len(val_loader)
        print(f"Epoch {epoch+1:03d} | Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            counter = 0
            torch.save(model.state_dict(), dt_name)
            print("Model saved!")
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}.")
                break


def train_all(state_type, reward_type, task_id):
    paths = setup_folders(state_type, reward_type, task_id)

    dt_path = paths["models"] / 'dt_medium_expert_model.pth'
    h5_path = paths["datasets"] / 'dataset_medium_expert.h5'
    if not os.path.exists(dt_path): train(dt_path, h5_path)

    dt_path = paths["models"] / 'dt_medium_model.pth'
    h5_path = paths["datasets"] / 'dataset_medium.h5'
    if not os.path.exists(dt_path): train(dt_path, h5_path)

    dt_path = paths["models"] / 'dt_medium_replay_model.pth'
    h5_path = paths["datasets"] / 'dataset_medium_replay.h5'
    if not os.path.exists(dt_path): train(dt_path, h5_path)