"""
Supervised training of PhysicsEncoder + DynamicsPredictor
on collected Agent B transition data.

Input:  obs_t (117,), action_t (2,)
Target: next_obs_t[109:111] = next vy, wz
"""
import sys
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(BASE, 'src'))

from agent_c_files.physics_modules import PhysicsEncoder, DynamicsPredictor

DATA_PATH    = os.path.join(BASE, 'src', 'agent_c_files', 'encoder_data.npz')
ENCODER_PATH = os.path.join(BASE, 'checkpoints', 'physics_encoder.pt')

# hyperparams
BATCH_SIZE = 512
EPOCHS     = 50
LR         = 1e-3
VAL_SPLIT  = 0.1


def train():
    device = torch.device('cpu')

    # load data
    print("Loading data...")
    data     = np.load(DATA_PATH)
    obs      = torch.tensor(data['obs'],      dtype=torch.float32)
    actions  = torch.tensor(data['actions'],  dtype=torch.float32)
    next_obs = torch.tensor(data['next_obs'], dtype=torch.float32)

    # inputs to encoder/predictor
    params  = obs[:, 111:117]    # normalized physics params (6,)
    vxvywz  = obs[:, 108:111]    # current vx, vy, wz (3,)

    # targets: next vy, wz
    targets = next_obs[:, 109:111]  # (N, 2)

    dataset = TensorDataset(params, vxvywz, actions, targets)

    val_size   = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    # models
    encoder   = PhysicsEncoder().to(device)
    predictor = DynamicsPredictor().to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(predictor.parameters()), lr=LR
    )
    loss_fn = nn.MSELoss()

    best_val_loss = float('inf')

    print("Training encoder...")
    for epoch in range(EPOCHS):
        # train
        encoder.train()
        predictor.train()
        train_losses = []

        for params_b, vxvywz_b, actions_b, targets_b in train_loader:
            embedding = encoder(params_b)
            predicted = predictor(embedding, vxvywz_b, actions_b)
            loss      = loss_fn(predicted, targets_b)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        # validate
        encoder.eval()
        predictor.eval()
        val_losses = []

        with torch.no_grad():
            for params_b, vxvywz_b, actions_b, targets_b in val_loader:
                embedding = encoder(params_b)
                predicted = predictor(embedding, vxvywz_b, actions_b)
                val_losses.append(loss_fn(predicted, targets_b).item())

        train_loss = np.mean(train_losses)
        val_loss   = np.mean(val_losses)

        print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
              f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        # save best encoder
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(encoder.state_dict(), ENCODER_PATH)
            print(f"  → saved best encoder (val_loss={val_loss:.6f})")

    print(f"\nDone. Best val_loss={best_val_loss:.6f}")
    print(f"Encoder saved to {ENCODER_PATH}")


if __name__ == '__main__':
    train()