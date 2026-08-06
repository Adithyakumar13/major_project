import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
from typing import Optional


# ==============================================================
# Configuration
# ==============================================================

class FeatureExtractorConfig:
    """Configuration for the feature extractor."""
    lidar_input_dim: int = 108
    lidar_hidden_dim: int = 128
    lidar_output_dim: int = 64
    physics_embedding_dim: int = 16
    use_velocity_features: bool = True
    use_dropout: bool = False
    dropout_rate: float = 0.1


# ==============================================================
# Feature Extractor
# ==============================================================

class FrozenPhysicsExtractor(BaseFeaturesExtractor):
    """
    Feature extractor that receives pre-computed physics embedding.
    
    Obs layout (127,):
        [0:108]   LiDAR scans (downsampled)
        [108:111] vx, vy, wz (velocities)
        [111:127] physics embedding (16 dims, pre-computed from encoder)
    
    Output: Combined features for policy/value networks.
    """
    
    def __init__(
        self,
        observation_space: spaces.Box,
        config: Optional[FeatureExtractorConfig] = None,
        verbose: bool = True,
    ):
        if config is None:
            config = FeatureExtractorConfig()
        
        self.config = config
        self.verbose = verbose
        
        # Calculate feature dimensions FIRST
        features_dim = config.lidar_output_dim  # LiDAR features
        
        if config.use_velocity_features:
            features_dim += 8  # Velocity features
        
        features_dim += config.physics_embedding_dim  # Physics embedding
        
        # Call parent with the actual features_dim
        super().__init__(observation_space, features_dim=features_dim)
        
        # Build LiDAR encoder
        self.lidar_encoder = nn.Sequential(
            nn.Linear(config.lidar_input_dim, config.lidar_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.lidar_hidden_dim, config.lidar_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.lidar_hidden_dim // 2, config.lidar_output_dim),
            nn.ReLU(),
        )
        
        # Create velocity encoder if needed
        if config.use_velocity_features:
            self.velocity_encoder = nn.Sequential(
                nn.Linear(3, 16),
                nn.ReLU(),
                nn.Linear(16, 8),
            )
        else:
            self.velocity_encoder = None
        
        # Optional: final projection layer
        if config.use_dropout:
            self.dropout = nn.Dropout(config.dropout_rate)
        else:
            self.dropout = nn.Identity()
        
        # Print summary
        if self.verbose:
            self._print_summary()
    
    def _print_summary(self):
        """Print a summary of the feature extractor."""
        print("\n" + "="*60)
        print("FrozenPhysicsExtractor Summary")
        print("="*60)
        # Use _observation_space (stored by BaseFeaturesExtractor)
        print(f"Observation space: {self._observation_space.shape}")
        print(f"Feature dimension: {self.features_dim}")
        print(f"  - LiDAR features: {self.config.lidar_output_dim}")
        if self.config.use_velocity_features:
            print(f"  - Velocity features: 8")
        print(f"  - Physics embedding: {self.config.physics_embedding_dim} (passed through)")
        print("="*60 + "\n")
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Extract features from observation.
        
        Args:
            obs: (batch_size, 127) observation tensor
                [0:108]   LiDAR
                [108:111] velocities (vx, vy, wz)
                [111:127] physics embedding (pre-computed)
        
        Returns:
            features: (batch_size, features_dim) combined features
        """
        # Split observation
        lidar = obs[:, :108]                    # (batch, 108)
        velocities = obs[:, 108:111]            # (batch, 3)
        physics_embedding = obs[:, 111:127]     # (batch, 16) - pre-computed!
        
        # Process LiDAR
        lidar_features = self.lidar_encoder(lidar)  # (batch, lidar_dim)
        
        # Process velocities (if enabled)
        if self.velocity_encoder is not None:
            velocity_features = self.velocity_encoder(velocities)  # (batch, 8)
        else:
            velocity_features = None
        
        # Combine features - physics embedding is passed through directly
        features = [lidar_features, physics_embedding]
        
        if velocity_features is not None:
            features.append(velocity_features)
        
        combined = torch.cat(features, dim=-1)
        
        # Apply dropout (if enabled)
        combined = self.dropout(combined)
        
        # Final clipping for stability
        combined = torch.clamp(combined, -10.0, 10.0)
        
        return combined