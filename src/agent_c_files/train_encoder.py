"""
Supervised training with weighted loss to improve Δwz prediction.
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
from dataclasses import dataclass

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(BASE, 'src'))

from agent_c_files.physics_modules import (
    PhysicsEncoder, 
    DynamicsPredictor,
    EncoderConfig,
)

DT = 0.01

@dataclass
class Config:
    data_path: str = os.path.join(BASE, 'src', 'agent_c_files', 'encoder_data.npz')
    encoder_save_path: str = os.path.join(BASE, 'checkpoints', 'physics_encoder.pt')
    
    batch_size: int = 512
    epochs: int = 80
    learning_rate: float = 1e-3
    val_split: float = 0.15
    
    embedding_dim: int = 16  # Larger embedding
    hidden_dim: int = 128    # Larger network
    dropout: float = 0.15
    
    wz_weight: float = 10.0  # Weight for Δwz loss
    
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


config = Config()


def load_data(config):
    print("Loading data...")
    data = np.load(config.data_path)
    
    obs = torch.tensor(data['obs'], dtype=torch.float32)
    actions = torch.tensor(data['actions'], dtype=torch.float32)
    next_obs = torch.tensor(data['next_obs'], dtype=torch.float32)
    
    print(f"Loaded {len(obs)} transitions")
    
    # Acceleration and Δwz
    acceleration = (next_obs[:, 108:109] - obs[:, 108:109]) / DT
    delta_wz = next_obs[:, 110:111] - obs[:, 110:111]
    targets = torch.cat([acceleration, delta_wz], dim=-1)
    
    print(f"Acceleration range: [{acceleration.min():.2f}, {acceleration.max():.2f}], mean: {acceleration.mean():.2f}")
    print(f"Δwz range: [{delta_wz.min():.4f}, {delta_wz.max():.4f}], mean: {delta_wz.mean():.4f}")
    
    # Filter
    steering = torch.abs(actions[:, 0])
    speed = torch.abs(obs[:, 108])
    mask = (steering > 0.01) | (speed > 1.0)
    
    straight_mask = ~mask
    straight_indices = torch.where(straight_mask)[0]
    if len(straight_indices) > 0:
        n_straight = min(len(straight_indices), int(0.1 * mask.sum()))
        straight_sample = straight_indices[torch.randperm(len(straight_indices))[:n_straight]]
        mask[straight_sample] = True
    
    params = obs[mask, 111:117]
    vxvywz = obs[mask, 108:111]
    actions = actions[mask]
    targets = targets[mask]
    
    print(f"Using {mask.sum()} transitions ({mask.sum()/len(mask)*100:.1f}%)")
    
    return params, vxvywz, actions, targets


def create_loaders(params, vxvywz, actions, targets, config):
    dataset = TensorDataset(params, vxvywz, actions, targets)
    
    val_size = int(len(dataset) * config.val_split)
    train_size = len(dataset) - val_size
    
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)
    
    print(f"Train: {train_size}, Val: {val_size}")
    return train_loader, val_loader


def train_epoch(encoder, predictor, loader, optimizer, loss_fn, device, wz_weight):
    encoder.train()
    predictor.train()
    
    total_loss = 0
    count = 0
    
    for params_b, vxvywz_b, actions_b, targets_b in loader:
        params_b = params_b.to(device)
        vxvywz_b = vxvywz_b.to(device)
        actions_b = actions_b.to(device)
        targets_b = targets_b.to(device)
        
        embedding = encoder(params_b)
        predicted = predictor(embedding, vxvywz_b, actions_b)
        
        # Separate losses with weighting
        loss_accel = loss_fn(predicted[:, 0:1], targets_b[:, 0:1])
        loss_wz = loss_fn(predicted[:, 1:2], targets_b[:, 1:2])
        loss = loss_accel + wz_weight * loss_wz
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(predictor.parameters()), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        count += 1
    
    return total_loss / count, loss_accel.item(), loss_wz.item()


def validate(encoder, predictor, loader, loss_fn, device, wz_weight):
    encoder.eval()
    predictor.eval()
    
    total_loss = 0
    count = 0
    
    with torch.no_grad():
        for params_b, vxvywz_b, actions_b, targets_b in loader:
            params_b = params_b.to(device)
            vxvywz_b = vxvywz_b.to(device)
            actions_b = actions_b.to(device)
            targets_b = targets_b.to(device)
            
            embedding = encoder(params_b)
            predicted = predictor(embedding, vxvywz_b, actions_b)
            
            loss_accel = loss_fn(predicted[:, 0:1], targets_b[:, 0:1])
            loss_wz = loss_fn(predicted[:, 1:2], targets_b[:, 1:2])
            loss = loss_accel + wz_weight * loss_wz
            
            total_loss += loss.item()
            count += 1
    
    return total_loss / count, loss_accel.item(), loss_wz.item()


def test_visualization(encoder, predictor, loader, device):
    encoder.eval()
    predictor.eval()
    
    preds = []
    targets_list = []
    
    with torch.no_grad():
        for params_b, vxvywz_b, actions_b, targets_b in loader:
            params_b = params_b.to(device)
            vxvywz_b = vxvywz_b.to(device)
            actions_b = actions_b.to(device)
            
            embedding = encoder(params_b)
            predicted = predictor(embedding, vxvywz_b, actions_b)
            
            preds.append(predicted.cpu().numpy())
            targets_list.append(targets_b.numpy())
            
            if len(np.concatenate(preds)) > 10000:
                break
    
    preds = np.concatenate(preds, axis=0)[:10000]
    targets_list = np.concatenate(targets_list, axis=0)[:10000]
    
    print("\n" + "="*50)
    print("Prediction Quality")
    print("="*50)
    
    # Acceleration
    mse_accel = np.mean((preds[:,0] - targets_list[:,0])**2)
    r2_accel = 1 - mse_accel / np.var(targets_list[:,0])
    print(f"Acceleration - MSE: {mse_accel:.6f}, R²: {r2_accel:.4f}")
    
    # Δwz
    mse_wz = np.mean((preds[:,1] - targets_list[:,1])**2)
    r2_wz = 1 - mse_wz / np.var(targets_list[:,1])
    print(f"Δwz - MSE: {mse_wz:.6f}, R²: {r2_wz:.4f}")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Acceleration
    max_val = max(abs(targets_list[:,0].max()), abs(preds[:,0].max()))
    axes[0].scatter(targets_list[:,0], preds[:,0], alpha=0.3, s=1)
    axes[0].plot([-max_val, max_val], [-max_val, max_val], 'r--', linewidth=2)
    axes[0].set_xlabel('True Acceleration (m/s²)')
    axes[0].set_ylabel('Predicted Acceleration (m/s²)')
    axes[0].set_title(f'Acceleration Prediction (R²={r2_accel:.3f})')
    axes[0].grid(True, alpha=0.3)
    
    # Δwz
    max_val = max(abs(targets_list[:,1].max()), abs(preds[:,1].max()))
    axes[1].scatter(targets_list[:,1], preds[:,1], alpha=0.3, s=1)
    axes[1].plot([-max_val, max_val], [-max_val, max_val], 'r--', linewidth=2)
    axes[1].set_xlabel('True Δwz')
    axes[1].set_ylabel('Predicted Δwz')
    axes[1].set_title(f'Δwz Prediction (R²={r2_wz:.3f})')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(config.encoder_save_path), 'encoder_test.png'), dpi=150)
    print(f"Plot saved to: {os.path.join(os.path.dirname(config.encoder_save_path), 'encoder_test.png')}")
    plt.show()
    
    return {'mse_accel': mse_accel, 'r2_accel': r2_accel, 'mse_wz': mse_wz, 'r2_wz': r2_wz}


def main():
    device = torch.device(config.device)
    print(f"Using device: {device}")
    print(f"Δwz loss weight: {config.wz_weight}")
    
    params, vxvywz, actions, targets = load_data(config)
    train_loader, val_loader = create_loaders(params, vxvywz, actions, targets, config)
    
    encoder_config = EncoderConfig(
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        num_layers=2,
        dropout=config.dropout,
        use_batch_norm=False,
    )
    
    encoder = PhysicsEncoder(encoder_config).to(device)
    predictor = DynamicsPredictor(
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim * 2,
    ).to(device)
    
    print(f"Encoder params: {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"Predictor params: {sum(p.numel() for p in predictor.parameters()):,}")
    
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=config.learning_rate,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    loss_fn = nn.MSELoss()
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 12
    
    print("\n" + "="*50)
    print("Training...")
    print("="*50)
    
    for epoch in range(config.epochs):
        train_loss, train_accel, train_wz = train_epoch(
            encoder, predictor, train_loader, optimizer, loss_fn, device, config.wz_weight
        )
        val_loss, val_accel, val_wz = validate(
            encoder, predictor, val_loader, loss_fn, device, config.wz_weight
        )
        scheduler.step(val_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3d}/{config.epochs} | Train: {train_loss:.6f} (Accel: {train_accel:.6f}, Wz: {train_wz:.6f}) | Val: {val_loss:.6f} (Accel: {val_accel:.6f}, Wz: {val_wz:.6f}) | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(encoder.state_dict(), config.encoder_save_path)
            print(f"  → Saved best encoder (val_loss: {val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    print(f"\nBest val_loss: {best_val_loss:.6f}")
    print(f"Encoder saved to: {config.encoder_save_path}")
    
    results = test_visualization(encoder, predictor, val_loader, device)
    
    print("\n" + "="*50)
    print("Final Results")
    print("="*50)
    print(f"Acceleration R²: {results['r2_accel']:.4f}")
    print(f"Δwz R²: {results['r2_wz']:.4f}")


if __name__ == '__main__':
    os.makedirs(os.path.dirname(config.encoder_save_path), exist_ok=True)
    main()