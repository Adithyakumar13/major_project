"""
SB3 features extractor that uses the pretrained frozen PhysicsEncoder.
Plugs directly into ActorCriticPolicy via policy_kwargs.
"""
import os
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# import from shared module
import sys
sys.path.append(os.path.join(BASE, 'src'))
from agent_c_files.physics_modules import PhysicsEncoder

ENCODER_PATH = os.path.join(BASE, 'checkpoints', 'physics_encoder.pt')


class FrozenPhysicsExtractor(BaseFeaturesExtractor):
    """
    obs layout (117,):
        [0:108]   lidar
        [108:111] vx, vy, wz
        [111:117] normalized physics params

    output: lidar_feat(64) + frozen_embedding(16) = 80
    """

    def __init__(self, observation_space: spaces.Box,
                 encoder_path: str = ENCODER_PATH):
        # change features_dim to 64 + 8 = 72
        super().__init__(observation_space, features_dim=72)

        # lidar encoder — trained with PPO
        self.lidar_encoder = nn.Sequential(
            nn.Linear(108, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        # physics encoder — pretrained and frozen
        self.physics_encoder = PhysicsEncoder()
        self.physics_encoder.load_state_dict(
            torch.load(encoder_path, map_location='cpu')
        )

        # freeze all encoder weights
        for param in self.physics_encoder.parameters():
            param.requires_grad = False

        print(f"Loaded frozen physics encoder from {encoder_path}")

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        lidar  = obs[:, :108]
        params = obs[:, 111:117]

        lidar_feat = self.lidar_encoder(lidar)
        embedding  = self.physics_encoder(params)  # frozen

        return torch.cat([lidar_feat, embedding], dim=-1)  # (batch, 72)