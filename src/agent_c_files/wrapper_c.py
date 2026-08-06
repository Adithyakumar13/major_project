import os
import numpy as np
import gym
import f110_gym
import gymnasium
from gymnasium import spaces
from typing import Optional, Dict, Any, Tuple
import torch
import torch.nn as nn

from physics_modules import PhysicsEncoder, EncoderConfig


# ==============================================================
# Physics configuration (same as Agent B)
# ==============================================================

PARAM_RANGES = {
    "mu":   (0.5, 1.2),
    "m":    (2.5, 5.5),
    "lf":   (0.12, 0.22),
    "lr":   (0.12, 0.22),
    "C_Sf": (4.0, 7.0),
    "C_Sr": (4.0, 7.0),
}

BASE_PARAMS = {
    "mu":   1.0489,
    "m":    3.74,
    "lf":   0.15875,
    "lr":   0.17145,
    "C_Sf": 4.718,
    "C_Sr": 5.4562,
    "I":    0.04712,
}


def sample_params(rng: Optional[np.random.Generator] = None) -> Dict[str, float]:
    """Sample random physics parameters."""
    if rng is None:
        rng = np.random.default_rng()
    
    params = {}
    for key, (low, high) in PARAM_RANGES.items():
        params[key] = float(rng.uniform(low, high))
    
    # Scale inertia with mass
    params["I"] = BASE_PARAMS["I"] * (params["m"] / BASE_PARAMS["m"])
    return params


def normalize_params(params: Dict[str, float]) -> np.ndarray:
    """Normalize physics parameters to [0, 1]."""
    normalized = []
    for key, (low, high) in PARAM_RANGES.items():
        normalized.append((params[key] - low) / (high - low))
    return np.array(normalized, dtype=np.float32)


class F110WrapperC(gymnasium.Env):
    """
    Agent C environment with physics encoding.
    
    Observation:
        108 LiDAR values
        vx, vy, wz (3)
        physics embedding (embedding_dim, default=16)
        
        Total dimension = 108 + 3 + embedding_dim
    
    Action:
        [steering_angle, speed_command]
    
    The physics embedding is computed from the current physics parameters
    using a pre-trained encoder. The encoder is frozen and only used
    for inference (no gradients).
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        map_path: str,
        waypoint_path: str,
        map_ext: str = ".png",
        reset_to_waypoint_start: bool = False,
        encoder_path: Optional[str] = None,
        embedding_dim: int = 16,
        seed: Optional[int] = None,
        use_encoder: bool = True,
    ):
        """
        Initialize the environment.
        
        Args:
            map_path: Path to the map directory
            waypoint_path: Path to waypoints CSV file
            map_ext: Map file extension
            reset_to_waypoint_start: Whether to reset to waypoint 0
            encoder_path: Path to pre-trained encoder weights
            embedding_dim: Dimension of physics embedding
            seed: Random seed
            use_encoder: Whether to use physics encoder (if False, uses raw params)
        """
        super().__init__()
        
        # Create base F1TENTH environment
        self.env = gym.make(
            "f110-v0",
            map=map_path,
            map_ext=map_ext,
            num_agents=1,
            disable_env_checker=True,
        )
        
        # Load waypoints
        self.waypoint_path = waypoint_path
        self.waypoints = self._load_waypoints(waypoint_path)
        
        self.s_waypoints = self.waypoints["s"]
        self.x_waypoints = self.waypoints["x"]
        self.y_waypoints = self.waypoints["y"]
        self.psi_waypoints = self.waypoints["psi"]
        self.kappa_waypoints = self.waypoints["kappa"]
        self.vx_reference = self.waypoints["vx_ref"]
        self.ax_reference = self.waypoints["ax_ref"]
        
        self.track_length = float(self.s_waypoints[-1])
        if self.track_length <= 0.0:
            raise ValueError("Waypoint track length must be positive.")
        
        self.reset_to_waypoint_start = reset_to_waypoint_start
        self.use_encoder = use_encoder
        self.embedding_dim = embedding_dim
        
        # RNG
        self.rng = np.random.default_rng(seed)
        
        # Current physics parameters
        self.current_params = BASE_PARAMS.copy()
        
        # Physics encoder (only used in _get_physics_embedding)
        self.encoder = None
        if use_encoder:
            config = EncoderConfig(
                embedding_dim=embedding_dim,
                use_batch_norm=False,  # Important: no BatchNorm for inference
            )
            self.encoder = PhysicsEncoder(config)
            
            if encoder_path is not None and os.path.exists(encoder_path):
                self._load_encoder(encoder_path)
            else:
                print(f"Warning: Encoder not loaded. Using random encoder.")
        
        # Observation space: LiDAR (108) + velocities (3) + physics embedding
        lidar_dim = 108
        velocity_dim = 3
        obs_dim = lidar_dim + velocity_dim + embedding_dim  # 108 + 3 + 16 = 127
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        
        # Action space (same as Agent B)
        self.action_space = spaces.Box(
            low=np.array([-0.4189, 0.0], dtype=np.float32),
            high=np.array([0.4189, 10.0], dtype=np.float32),
            dtype=np.float32,
        )
        
        # State tracking
        self.previous_s = None
        self.previous_index = None
        self.previous_action = np.zeros(2, dtype=np.float32)
        
        print(f"F110WrapperC initialized:")
        print(f"  LiDAR dim: {lidar_dim}")
        print(f"  Velocity dim: {velocity_dim}")
        print(f"  Embedding dim: {embedding_dim}")
        print(f"  Total obs dim: {obs_dim}")
        print(f"  Use encoder: {use_encoder}")
    
    # ==============================================================
    # Waypoint loading
    # ==============================================================
    
    def _load_waypoints(self, waypoint_path: str) -> Dict[str, np.ndarray]:
        """Load waypoints from CSV file."""
        if not os.path.exists(waypoint_path):
            raise FileNotFoundError(f"Waypoint file not found: {waypoint_path}")
        
        data = np.loadtxt(
            waypoint_path,
            delimiter=";",
            comments="#",
            skiprows=3,
            dtype=np.float32,
        )
        
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        if data.shape[1] < 7:
            raise ValueError(
                "Waypoint file must contain at least 7 columns: "
                "s, x, y, psi, kappa, vx, ax."
            )
        
        if len(data) < 2:
            raise ValueError("At least two waypoints are required.")
        
        return {
            "s": data[:, 0],
            "x": data[:, 1],
            "y": data[:, 2],
            "psi": data[:, 3],
            "kappa": data[:, 4],
            "vx_ref": data[:, 5],
            "ax_ref": data[:, 6],
        }
    
    # ==============================================================
    # Encoder handling
    # ==============================================================
    
    def _load_encoder(self, encoder_path: str):
        """Load pre-trained encoder weights."""
        try:
            checkpoint = torch.load(encoder_path, map_location="cpu", weights_only=False)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.encoder.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.encoder.load_state_dict(checkpoint)
            self.encoder.eval()
            print(f"Encoder loaded from: {encoder_path}")
        except Exception as e:
            print(f"Failed to load encoder: {e}")
            print("Using random encoder.")
    
    def _get_physics_embedding(self) -> np.ndarray:
        """
        Get physics embedding for current parameters.
        
        Returns:
            embedding: (embedding_dim,) physics embedding
        """
        if not self.use_encoder or self.encoder is None:
            # Fallback: use raw normalized params (padded to embedding_dim)
            raw = normalize_params(self.current_params)
            if len(raw) < self.embedding_dim:
                # Pad with zeros to match embedding dimension
                return np.pad(raw, (0, self.embedding_dim - len(raw)), 'constant').astype(np.float32)
            return raw.astype(np.float32)
        
        # Convert physics params to tensor
        params_tensor = torch.FloatTensor(
            normalize_params(self.current_params)
        ).unsqueeze(0)  # Shape: (1, 6)
        
        # Get embedding from encoder (frozen)
        with torch.no_grad():
            self.encoder.eval()  # Ensure eval mode
            embedding = self.encoder(params_tensor).squeeze(0).cpu().numpy()
        
        # Clip to prevent extreme values
        embedding = np.clip(embedding, -5.0, 5.0)
        
        # Check for NaN
        if np.isnan(embedding).any():
            print(f"WARNING: NaN in embedding! Falling back to raw params.")
            raw = normalize_params(self.current_params)
            return np.pad(raw, (0, self.embedding_dim - len(raw)), 'constant').astype(np.float32)
        
        return embedding.astype(np.float32)
    
    # ==============================================================
    # Physics handling
    # ==============================================================
    
    def _update_vehicle_params(self):
        """Apply current physics parameters to the simulator."""
        params = self.env.sim.agents[0].params.copy()
        params.update(self.current_params)
        
        # Try different update methods for compatibility
        if hasattr(self.env, "update_params"):
            try:
                self.env.update_params(params, index=0)
                return
            except TypeError:
                try:
                    self.env.update_params(params, agent_idx=0)
                    return
                except TypeError:
                    pass
        
        if hasattr(self.env.sim, "update_params"):
            try:
                self.env.sim.update_params(params, index=0)
                return
            except TypeError:
                try:
                    self.env.sim.update_params(params, agent_idx=0)
                    return
                except TypeError:
                    pass
        
        # Fallback
        self.env.sim.agents[0].params.update(params)
    
    # ==============================================================
    # Reset and step
    # ==============================================================
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment."""
        super().reset(seed=seed)
        
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        # Sample new physics
        self.current_params = sample_params(rng=self.rng)
        
        # Set start pose
        if self.reset_to_waypoint_start:
            start_pose = np.array([[
                self.x_waypoints[0],
                self.y_waypoints[0],
                self.psi_waypoints[0],
            ]], dtype=np.float32)
        else:
            start_pose = np.array([[0.0, 0.0, np.pi / 2.0]], dtype=np.float32)
        
        # Reset simulator
        raw_obs, _, _, info = self.env.reset(poses=start_pose)
        
        # Apply physics parameters
        self._update_vehicle_params()
        
        # Get waypoint features
        features = self._get_waypoint_features(raw_obs)
        
        self.previous_s = features["s"]
        self.previous_index = features["index"]
        self.previous_action = np.zeros(2, dtype=np.float32)
        
        # Build info dict
        info = dict(info)
        info["physics_params"] = normalize_params(self.current_params).copy()
        info["physics_embedding"] = self._get_physics_embedding().copy()
        info["waypoint_index"] = features["index"]
        info["waypoint_s"] = features["s"]
        info["lateral_error"] = features["lateral_error"]
        info["heading_error"] = features["heading_error"]
        info["target_speed"] = features["target_speed"]
        
        return self._process_obs(raw_obs), info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take a step in the environment."""
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        
        if action.shape[0] != 2:
            raise ValueError(f"Expected action shape (2,), got {action.shape}")
        
        action = np.clip(
            action,
            self.action_space.low,
            self.action_space.high,
        )
        
        # Step simulator
        raw_obs, _, _, info = self.env.step(
            np.array([[action[0], action[1]]], dtype=np.float32)
        )
        
        collision = bool(raw_obs["collisions"][0])
        lap_complete = bool(raw_obs["lap_counts"][0] >= 1)
        
        terminated = collision or lap_complete
        truncated = False
        
        reward, reward_info = self._compute_reward(
            raw_obs=raw_obs,
            collision=collision,
            lap_complete=lap_complete,
            action=action,
        )
        
        # Build info dict
        info = dict(info)
        info["physics_params"] = normalize_params(self.current_params).copy()
        info["physics_embedding"] = self._get_physics_embedding().copy()
        info["lap_time"] = float(raw_obs["lap_times"][0])
        info["lap_count"] = int(raw_obs["lap_counts"][0])
        info["collision"] = collision
        info.update(reward_info)
        
        return (
            self._process_obs(raw_obs),
            float(reward),
            terminated,
            truncated,
            info,
        )
    
    # ==============================================================
    # Observation processing
    # ==============================================================
    
    def _process_obs(self, raw_obs: Dict) -> np.ndarray:
        """
        Process raw observation.
        
        Returns:
            obs: [108 LiDAR, vx, vy, wz, physics_embedding]
        """
        # LiDAR (downsampled from 1080 to 108)
        lidar = np.asarray(raw_obs["scans"][0][::10], dtype=np.float32)
        lidar = np.clip(lidar, 0.0, 30.0)
        
        # Velocities
        vx = np.array([raw_obs["linear_vels_x"][0]], dtype=np.float32)
        vy = np.array([raw_obs["linear_vels_y"][0]], dtype=np.float32)
        wz = np.array([raw_obs["ang_vels_z"][0]], dtype=np.float32)
        
        # Clip velocities
        vx = np.clip(vx, -10.0, 10.0)
        vy = np.clip(vy, -5.0, 5.0)
        wz = np.clip(wz, -5.0, 5.0)
        
        # Physics embedding (from encoder)
        physics = self._get_physics_embedding()
        
        # Combine
        obs = np.concatenate([lidar, vx, vy, wz, physics])
        
        # Clean up
        obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
        obs = np.clip(obs, -10.0, 10.0)
        
        return obs.astype(np.float32)
    
    # ==============================================================
    # Waypoint calculations
    # ==============================================================
    
    @staticmethod
    def _wrap_angle(angle: float) -> float:
        """Wrap angle to [-pi, pi]."""
        return np.arctan2(np.sin(angle), np.cos(angle))
    
    def _get_vehicle_pose(self, raw_obs: Dict) -> Tuple[float, float, float]:
        """Extract vehicle pose from observation."""
        x = float(raw_obs["poses_x"][0])
        y = float(raw_obs["poses_y"][0])
        psi = float(raw_obs["poses_theta"][0])
        return x, y, psi
    
    def _nearest_waypoint(
        self,
        x: float,
        y: float,
        search_window: int = 40,
    ) -> Tuple[int, float]:
        """Find nearest waypoint."""
        n_waypoints = len(self.x_waypoints)
        
        if self.previous_index is None:
            candidate_indices = np.arange(n_waypoints)
        else:
            offsets = np.arange(-search_window, search_window + 1)
            candidate_indices = (self.previous_index + offsets) % n_waypoints
        
        dx = self.x_waypoints[candidate_indices] - x
        dy = self.y_waypoints[candidate_indices] - y
        squared_distances = dx * dx + dy * dy
        
        local_min_index = int(np.argmin(squared_distances))
        index = int(candidate_indices[local_min_index])
        distance = float(np.sqrt(squared_distances[local_min_index]))
        
        return index, distance
    
    def _get_waypoint_features(self, raw_obs: Dict) -> Dict[str, float]:
        """Get waypoint-related features."""
        x, y, psi = self._get_vehicle_pose(raw_obs)
        index, distance = self._nearest_waypoint(x, y)
        
        waypoint_x = float(self.x_waypoints[index])
        waypoint_y = float(self.y_waypoints[index])
        waypoint_psi = float(self.psi_waypoints[index])
        waypoint_s = float(self.s_waypoints[index])
        target_speed = float(self.vx_reference[index])
        curvature = float(self.kappa_waypoints[index])
        
        # Lateral error
        dx = x - waypoint_x
        dy = y - waypoint_y
        normal_x = -np.sin(waypoint_psi)
        normal_y = np.cos(waypoint_psi)
        lateral_error = normal_x * dx + normal_y * dy
        
        # Heading error
        heading_error = self._wrap_angle(psi - waypoint_psi)
        
        return {
            "index": index,
            "distance": distance,
            "s": waypoint_s,
            "lateral_error": float(lateral_error),
            "heading_error": float(heading_error),
            "target_speed": target_speed,
            "curvature": curvature,
        }
    
    # ==============================================================
    # Reward function (matching Agent B)
    # ==============================================================
    
    def _compute_reward(
        self,
        raw_obs: Dict,
        collision: bool,
        lap_complete: bool,
        action: np.ndarray,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute reward (identical to Agent B's reward structure).
        """
        if collision:
            return -50.0, {
                "reward_progress": 0.0,
                "reward_lateral": 0.0,
                "reward_heading": 0.0,
                "reward_speed": 0.0,
                "reward_slip": 0.0,
                "reward_yaw": 0.0,
                "reward_steering": 0.0,
                "progress_delta": 0.0,
                "waypoint_index": self.previous_index,
                "waypoint_s": self.previous_s,
            }
        
        if lap_complete:
            return 100.0, {
                "reward_progress": 0.0,
                "reward_lateral": 0.0,
                "reward_heading": 0.0,
                "reward_speed": 0.0,
                "reward_slip": 0.0,
                "reward_yaw": 0.0,
                "reward_steering": 0.0,
                "progress_delta": 0.0,
                "waypoint_index": self.previous_index,
                "waypoint_s": self.previous_s,
            }
        
        features = self._get_waypoint_features(raw_obs)
        
        current_s = features["s"]
        previous_s = self.previous_s if self.previous_s is not None else current_s
        
        # Progress (with lap wrapping)
        delta_s = current_s - previous_s
        if delta_s < -0.5 * self.track_length:
            delta_s += self.track_length
        elif delta_s > 0.5 * self.track_length:
            delta_s -= self.track_length
        
        # Clamp delta_s to prevent extreme values
        delta_s = np.clip(delta_s, -5.0, 5.0)
        
        # Vehicle state
        vx = float(raw_obs["linear_vels_x"][0])
        vy = float(raw_obs["linear_vels_y"][0])
        wz = float(raw_obs["ang_vels_z"][0])
        steering = float(action[0])
        
        lateral_error = np.clip(features["lateral_error"], -5.0, 5.0)
        heading_error = np.clip(features["heading_error"], -0.5, 0.5)
        target_speed = features["target_speed"]
        
        # Clamp speed difference
        speed_diff = np.clip(vx - target_speed, -5.0, 5.0)
        
        # Reward components (scaled down for stability)
        reward_progress = 10.0 * delta_s
        reward_lateral = -0.05 * abs(lateral_error)
        reward_heading = -0.02 * abs(heading_error)
        reward_speed = -0.005 * abs(speed_diff)
        reward_slip = -0.01 * abs(vy)
        reward_yaw = -0.002 * abs(wz)
        reward_steering = -0.001 * steering * steering
        time_penalty = -0.001
        
        total_reward = (
            reward_progress
            + reward_lateral
            + reward_heading
            + reward_speed
            + reward_slip
            + reward_yaw
            + reward_steering
            + time_penalty
        )
        
        # Clip total reward to prevent extreme values
        total_reward = np.clip(total_reward, -50.0, 50.0)
        
        # Update state
        self.previous_s = current_s
        self.previous_index = features["index"]
        self.previous_action = action.copy()
        
        reward_info = {
            "reward_progress": float(reward_progress),
            "reward_lateral": float(reward_lateral),
            "reward_heading": float(reward_heading),
            "reward_speed": float(reward_speed),
            "reward_slip": float(reward_slip),
            "reward_yaw": float(reward_yaw),
            "reward_steering": float(reward_steering),
            "progress_delta": float(delta_s),
            "waypoint_index": int(features["index"]),
            "waypoint_s": float(current_s),
            "lateral_error": float(lateral_error),
            "heading_error": float(heading_error),
            "target_speed": float(target_speed),
            "vehicle_speed": float(vx),
            "physics_mu": float(self.current_params["mu"]),
            "physics_mass": float(self.current_params["m"]),
        }
        
        return float(total_reward), reward_info
    
    # ==============================================================
    # Rendering and cleanup
    # ==============================================================
    
    def render(self, mode: str = "human"):
        """Render the environment."""
        self.env.render(mode=mode)
    
    def close(self):
        """Close the environment."""
        self.env.close()