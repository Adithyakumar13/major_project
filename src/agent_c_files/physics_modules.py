"""
Physics Modules for Agent C
Simple, clean encoder and predictor for physics-aware autonomous racing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


@dataclass
class EncoderConfig:
    """Configuration for physics encoder."""
    physics_dim: int = 6
    embedding_dim: int = 16
    hidden_dim: int = 128
    num_layers: int = 2  # Simpler: 2 layers
    dropout: float = 0.0  # No dropout for clean inference
    use_batch_norm: bool = False  # Always False for inference


class PhysicsEncoder(nn.Module):
    """
    Simple MLP encoder: 6 physics params -> 16-dim embedding.
    No BatchNorm for clean single-sample inference.
    """
    
    def __init__(self, config: Optional[EncoderConfig] = None):
        super().__init__()
        
        if config is None:
            config = EncoderConfig()
        
        self.config = config
        self.embedding_dim = config.embedding_dim
        
        # Simple MLP: 6 -> 128 -> 16
        self.net = nn.Sequential(
            nn.Linear(6, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.embedding_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: (batch, 6) -> (batch, embedding_dim)"""
        return self.net(x)
    
    def get_embedding_size(self) -> int:
        return self.embedding_dim


class DynamicsPredictor(nn.Module):
    """
    Predicts Δvx and Δwz given embedding + state + action.
    """
    
    def __init__(
        self,
        embedding_dim: int = 16,
        state_dim: int = 3,
        action_dim: int = 2,
        hidden_dim: int = 256,
    ):
        super().__init__()
        
        # Input: embedding + state (vx, vy, wz) + action (steering, speed)
        input_dim = embedding_dim + state_dim + action_dim
        
        # Simple MLP: input -> 256 -> 2 (Δvx, Δwz)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # Output: Δvx, Δwz
        )
    
    def forward(
        self,
        embedding: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            embedding: (batch, embedding_dim) physics embedding
            state: (batch, 3) [vx, vy, wz]
            action: (batch, 2) [steering, speed_command]
        
        Returns:
            (batch, 2) [Δvx, Δwz]
        """
        x = torch.cat([embedding, state, action], dim=-1)
        return self.net(x)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model: nn.Module, name: str = "Model"):
    """Print model summary."""
    print(f"\n{'='*50}")
    print(f"{name} Summary")
    print(f"{'='*50}")
    print(f"Architecture: {model.__class__.__name__}")
    print(f"Parameters: {count_parameters(model):,}")
    print(f"{'='*50}\n")