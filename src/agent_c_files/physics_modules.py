import torch
import torch.nn as nn


class PhysicsEncoder(nn.Module):
    """
    Encodes normalized physics params (6,) into embedding (8,).
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DynamicsPredictor(nn.Module):
    """
    Predicts next vy, wz given:
        embedding(8) + vx,vy,wz(3) + action(2) = 13
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(13, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, embedding: torch.Tensor,
                vxvywz: torch.Tensor,
                action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([embedding, vxvywz, action], dim=-1)
        return self.net(x)