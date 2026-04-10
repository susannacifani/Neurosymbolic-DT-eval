import torch
import torch.nn as nn
import numpy as np
import h5py
import random
from torch.utils.data import Dataset, DataLoader
from utils import setup_folders
import os

# 1. DATASET
class DTDataset(Dataset):
    def __init__(self, h5_path, keys, K=10):
        self.h5_path = h5_path
        self.keys = keys
        self.K = K

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as f:
            grp = f[self.keys[idx]]
            traj_len = grp.attrs['length']
            
            start_idx = np.random.randint(0, traj_len - self.K + 1) if traj_len > self.K else 0
            end_idx = min(start_idx + self.K, traj_len)
            
            images = grp['observations'][start_idx:end_idx].astype(np.float32) / 255.0
            actions = grp['actions'][start_idx:end_idx]
            rtgs = grp['returns_to_go'][start_idx:end_idx].astype(np.float32) / 100.0
            
            current_k = len(actions) 
            timesteps = np.arange(start_idx, start_idx + current_k)
            

            pad_len = self.K - current_k
            if pad_len > 0:
                images = np.pad(images, ((pad_len, 0), (0,0), (0,0), (0,0)), mode='constant')
                actions = np.pad(actions, (pad_len, 0), mode='constant')
                rtgs = np.pad(rtgs, (pad_len, 0), mode='constant')
                timesteps = np.pad(timesteps, (pad_len, 0), mode='constant')
                
                attention_mask = np.concatenate([np.zeros(pad_len), np.ones(current_k)])
            else:
                attention_mask = np.ones(self.K)

            mask_expanded = np.repeat(attention_mask, 3) 

        return {
            'images': torch.tensor(images, dtype=torch.float32), 
            'actions': torch.tensor(actions, dtype=torch.long),
            'rtgs': torch.tensor(rtgs, dtype=torch.float32),
            'timesteps': torch.tensor(timesteps, dtype=torch.long),
            'attention_mask': torch.tensor(mask_expanded, dtype=torch.bool)
        }

# 2. MODEL 
class DecisionTransformer(nn.Module):
    def __init__(self, action_dim, hidden_size=128, max_ep_len=1000):
        super().__init__()
        self.hidden_size = hidden_size
        self.embed_image = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2), nn.ReLU(),
            nn.Flatten(), nn.Linear(3136, hidden_size)
        )
        self.embed_rtg = nn.Linear(1, hidden_size)
        self.embed_action = nn.Embedding(action_dim, hidden_size)
        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, 
                                                   dim_feedforward=4*hidden_size, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.predict_action = nn.Linear(hidden_size, action_dim)

    def forward(self, images, actions, rtgs, timesteps):
        B, T, C, H, W = images.shape
        state_emb = self.embed_image(images.view(B * T, C, H, W)).view(B, T, self.hidden_size)
        time_emb = self.embed_timestep(timesteps)
        
        state_emb += time_emb
        action_emb = self.embed_action(actions) + time_emb
        rtg_emb = self.embed_rtg(rtgs.unsqueeze(-1)) + time_emb
        
        stacked = torch.stack((rtg_emb, state_emb, action_emb), dim=2).view(B, 3 * T, self.hidden_size)
        seq_len = 3 * T
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).to(images.device).bool()
        
        out = self.transformer(stacked, mask=mask)
        return self.predict_action(out[:, 1::3]) 

# 3. TRAINING 
def train(dt_name, dataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")
    
    h5_path = dataset
    with h5py.File(h5_path, 'r') as f:
        keys = list(f.keys())
        random.shuffle(keys)
        split = int(0.8 * len(keys))
        train_keys, val_keys = keys[:split], keys[split:]

    train_loader = DataLoader(DTDataset(h5_path, train_keys), batch_size=32, shuffle=True)
    val_loader = DataLoader(DTDataset(h5_path, val_keys), batch_size=32, shuffle=False)
    
    model = DecisionTransformer(action_dim=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    patience, counter = 10, 0

    for epoch in range(500):
        model.train()
        train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            logits = model(batch['images'].to(device), batch['actions'].to(device), 
                           batch['rtgs'].to(device), batch['timesteps'].to(device))
            
            action_mask = batch['attention_mask'][:, 2::3].to(device)
            loss = loss_fn(logits[action_mask], batch['actions'].to(device)[action_mask])
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                logits = model(batch['images'].to(device), batch['actions'].to(device), 
                               batch['rtgs'].to(device), batch['timesteps'].to(device))
                action_mask = batch['attention_mask'][:, 2::3].to(device)
                val_loss += loss_fn(logits[action_mask], batch['actions'].to(device)[action_mask]).item()
        
        avg_val = val_loss / len(val_loader)
        print(f"Epoch {epoch+1:03d} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            counter = 0
            torch.save(model.state_dict(), dt_name)
        else:
            counter += 1
            print(f"   --> Nessun miglioramento ({counter}/{patience})")
            if counter >= patience:
                print(f"Early stopping epoch {epoch+1}.")
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


