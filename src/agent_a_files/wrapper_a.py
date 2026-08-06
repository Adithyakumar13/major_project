import os
import numpy as np
import gym
import f110_gym
import gymnasium
from gymnasium import spaces


class F110Wrapper(gymnasium.Env):
    """
    F1TENTH Gym wrapper with waypoint-based reward shaping.

    Observation:
        108 LiDAR values
        vx
        vy
        wz

        Total dimension = 111

    Action:
        [steering_angle, speed_command]
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        map_path,
        waypoint_path,
        map_ext=".png",
        reset_to_waypoint_start=False,
    ):
        super().__init__()

        self.env = gym.make(
            "f110-v0",
            map=map_path,
            map_ext=map_ext,
            num_agents=1,
            disable_env_checker=True,
        )

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

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(111,),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=np.array([-0.4189, 0.0], dtype=np.float32),
            high=np.array([0.4189, 10.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.previous_s = None
        self.previous_index = None
        self.previous_action = np.zeros(2, dtype=np.float32)

    # ------------------------------------------------------------------
    # Waypoint loading
    # ------------------------------------------------------------------

    def _load_waypoints(self, waypoint_path):
        """
        Expected file format:

        # comment
        # comment
        # s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2
        0.0; 1.0; 2.0; 0.0; 0.1; 5.0; 0.0
        """

        if not os.path.exists(waypoint_path):
            raise FileNotFoundError(
                f"Waypoint file not found: {waypoint_path}"
            )

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

    # ------------------------------------------------------------------
    # Environment reset and step
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.reset_to_waypoint_start:
            start_pose = np.array(
                [[
                    self.x_waypoints[0],
                    self.y_waypoints[0],
                    self.psi_waypoints[0],
                ]],
                dtype=np.float32,
            )
        else:
            # Use this only if the simulator and waypoint file use
            # the same coordinate frame with this starting pose.
            start_pose = np.array(
                [[0.0, 0.0, np.pi / 2.0]],
                dtype=np.float32,
            )

        raw_obs, _, _, info = self.env.reset(
            poses=start_pose
        )

        features = self._get_waypoint_features(raw_obs)

        self.previous_s = features["s"]
        self.previous_index = features["index"]
        self.previous_action = np.zeros(2, dtype=np.float32)

        info = dict(info)
        info["waypoint_index"] = features["index"]
        info["waypoint_s"] = features["s"]
        info["lateral_error"] = features["lateral_error"]
        info["heading_error"] = features["heading_error"]
        info["target_speed"] = features["target_speed"]

        return self._process_obs(raw_obs), info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)

        if action.shape[0] != 2:
            raise ValueError(
                f"Expected action with shape (2,), got {action.shape}"
            )

        action = np.clip(
            action,
            self.action_space.low,
            self.action_space.high,
        )

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

        info = dict(info)

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

    # ------------------------------------------------------------------
    # Observation processing
    # ------------------------------------------------------------------

    def _process_obs(self, raw_obs):
        """
        Creates a 111-dimensional observation:

        108 downsampled LiDAR values
        vx
        vy
        wz
        """

        lidar = np.asarray(
            raw_obs["scans"][0][::10],
            dtype=np.float32,
        )

        vx = np.array(
            [raw_obs["linear_vels_x"][0]],
            dtype=np.float32,
        )

        vy = np.array(
            [raw_obs["linear_vels_y"][0]],
            dtype=np.float32,
        )

        wz = np.array(
            [raw_obs["ang_vels_z"][0]],
            dtype=np.float32,
        )

        obs = np.concatenate([lidar, vx, vy, wz])

        obs = np.nan_to_num(
            obs,
            nan=0.0,
            posinf=30.0,
            neginf=-30.0,
        )

        return obs.astype(np.float32)

    # ------------------------------------------------------------------
    # Waypoint calculations
    # ------------------------------------------------------------------

    @staticmethod
    def _wrap_angle(angle):
        return np.arctan2(
            np.sin(angle),
            np.cos(angle),
        )

    def _get_vehicle_pose(self, raw_obs):
        required_keys = [
            "poses_x",
            "poses_y",
            "poses_theta",
        ]

        missing_keys = [
            key for key in required_keys
            if key not in raw_obs
        ]

        if missing_keys:
            raise KeyError(
                f"Missing pose keys: {missing_keys}. "
                f"Available keys: {list(raw_obs.keys())}"
            )

        x = float(raw_obs["poses_x"][0])
        y = float(raw_obs["poses_y"][0])
        psi = float(raw_obs["poses_theta"][0])

        return x, y, psi

    def _nearest_waypoint(self, x, y, search_window=40):
        """
        Search locally around the previous waypoint.

        Local searching prevents the nearest point from jumping to
        another part of the track near hairpins or overlapping sections.
        """

        n_waypoints = len(self.x_waypoints)

        if self.previous_index is None:
            candidate_indices = np.arange(n_waypoints)
        else:
            offsets = np.arange(
                -search_window,
                search_window + 1,
            )

            candidate_indices = (
                self.previous_index + offsets
            ) % n_waypoints

        dx = self.x_waypoints[candidate_indices] - x
        dy = self.y_waypoints[candidate_indices] - y

        squared_distances = dx * dx + dy * dy
        local_min_index = int(np.argmin(squared_distances))

        index = int(candidate_indices[local_min_index])
        distance = float(
            np.sqrt(squared_distances[local_min_index])
        )

        return index, distance

    def _get_waypoint_features(self, raw_obs):
        """
        Returns information about the closest local waypoint.
        """

        x, y, psi = self._get_vehicle_pose(raw_obs)

        index, distance = self._nearest_waypoint(x, y)

        waypoint_x = float(self.x_waypoints[index])
        waypoint_y = float(self.y_waypoints[index])
        waypoint_psi = float(self.psi_waypoints[index])
        waypoint_s = float(self.s_waypoints[index])
        target_speed = float(self.vx_reference[index])
        curvature = float(self.kappa_waypoints[index])

        dx = x - waypoint_x
        dy = y - waypoint_y

        # Left-side normal of the waypoint tangent.
        normal_x = -np.sin(waypoint_psi)
        normal_y = np.cos(waypoint_psi)

        signed_lateral_error = (
            normal_x * dx +
            normal_y * dy
        )

        heading_error = self._wrap_angle(
            psi - waypoint_psi
        )

        return {
            "index": index,
            "distance": distance,
            "s": waypoint_s,
            "lateral_error": float(signed_lateral_error),
            "heading_error": float(heading_error),
            "target_speed": target_speed,
            "curvature": curvature,
        }

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(
        self,
        raw_obs,
        collision,
        lap_complete,
        action,
    ):
        """
        Reward components:

        1. Progress along the waypoint path.
        2. Lateral tracking error.
        3. Heading error.
        4. Soft reference-speed error.
        5. Lateral velocity penalty.
        6. Yaw-rate penalty.
        7. Steering effort penalty.
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
                "waypoint_index": self.previous_index,
                "waypoint_s": self.previous_s,
            }

        features = self._get_waypoint_features(raw_obs)

        current_s = features["s"]
        previous_s = self.previous_s

        if previous_s is None:
            previous_s = current_s

        delta_s = current_s - previous_s

        # Handle crossing the start/finish line.
        if delta_s < -0.5 * self.track_length:
            delta_s += self.track_length
        elif delta_s > 0.5 * self.track_length:
            delta_s -= self.track_length

        vx = float(raw_obs["linear_vels_x"][0])
        vy = float(raw_obs["linear_vels_y"][0])
        wz = float(raw_obs["ang_vels_z"][0])

        steering = float(action[0])

        lateral_error = features["lateral_error"]
        heading_error = features["heading_error"]
        target_speed = features["target_speed"]

        # Main objective: make progress along the racing line.
        reward_progress = 10.0 * delta_s

        # Keep the car near the waypoint racing line.
        reward_lateral = -0.05 * abs(lateral_error)

        # Encourage alignment with the local waypoint tangent.
        reward_heading = -0.02 * abs(heading_error)

        # Soft speed tracking. This should not dominate progress.
        reward_speed = -0.005 * abs(vx - target_speed)

        # Penalize excessive lateral motion and yaw motion.
        reward_slip = -0.01 * abs(vy)
        reward_yaw = -0.002 * abs(wz)

        # Mild steering effort penalty.
        reward_steering = -0.001 * steering * steering

        time_penalty = -0.001

        reward = (
            reward_progress
            + reward_lateral
            + reward_heading
            + reward_speed
            + reward_slip
            + reward_yaw
            + reward_steering
            + time_penalty
        )

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
            "waypoint_index": int(features["index"]),
            "waypoint_s": float(current_s),
            "lateral_error": float(lateral_error),
            "heading_error": float(heading_error),
            "target_speed": float(target_speed),
            "progress_delta": float(delta_s),
            "vehicle_speed": float(vx),
        }

        return float(reward), reward_info

    # ------------------------------------------------------------------
    # Rendering and cleanup
    # ------------------------------------------------------------------

    def render(self):
        return self.env.render(mode="human")

    def close(self):
        self.env.close()