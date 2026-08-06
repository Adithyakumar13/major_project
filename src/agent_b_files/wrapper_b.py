import os
import numpy as np
import gym
import f110_gym
import gymnasium
from gymnasium import spaces


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


def sample_params(rng=None):
    """
    Sample one physics configuration for an episode.
    """

    if rng is None:
        rng = np.random.default_rng()

    params = {}

    for key, (low, high) in PARAM_RANGES.items():
        params[key] = float(
            rng.uniform(low, high)
        )

    # Assumed linear scaling of yaw inertia with mass.
    params["I"] = (
        BASE_PARAMS["I"]
        * params["m"]
        / BASE_PARAMS["m"]
    )

    return params


def normalize_params(params):
    """
    Normalize randomized physics parameters to [0, 1].
    """

    normalized = []

    for key, (low, high) in PARAM_RANGES.items():
        value = (
            params[key] - low
        ) / (high - low)

        normalized.append(value)

    return np.asarray(
        normalized,
        dtype=np.float32,
    )


class F110WrapperB(gymnasium.Env):
    """
    Agent B environment.

    Observation:
        108 LiDAR values
        vx, vy, wz
        6 normalized physics parameters

        Total dimension = 117

    Action:
        [steering_angle, speed_command]
    """

    metadata = {
        "render_modes": ["human"],
    }

    def __init__(
        self,
        map_path,
        waypoint_path,
        map_ext=".png",
        reset_to_waypoint_start=False,
        seed=None,
    ):
        super().__init__()

        self.env = gym.make(
            "f110-v0",
            map=map_path,
            map_ext=map_ext,
            num_agents=1,
            disable_env_checker=True,
        )

        self.rng = np.random.default_rng(seed)

        self.waypoint_path = waypoint_path
        self.waypoints = self._load_waypoints(
            waypoint_path
        )

        self.s_waypoints = self.waypoints["s"]
        self.x_waypoints = self.waypoints["x"]
        self.y_waypoints = self.waypoints["y"]
        self.psi_waypoints = self.waypoints["psi"]
        self.kappa_waypoints = self.waypoints["kappa"]
        self.vx_reference = self.waypoints["vx_ref"]
        self.ax_reference = self.waypoints["ax_ref"]

        self.track_length = float(
            self.s_waypoints[-1]
        )

        if self.track_length <= 0.0:
            raise ValueError(
                "Waypoint track length must be positive."
            )

        self.reset_to_waypoint_start = (
            reset_to_waypoint_start
        )

        self.current_params = BASE_PARAMS.copy()

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(117,),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=np.array(
                [-0.4189, 0.0],
                dtype=np.float32,
            ),
            high=np.array(
                [0.4189, 10.0],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        self.previous_s = None
        self.previous_index = None
        self.previous_action = np.zeros(
            2,
            dtype=np.float32,
        )

    # ==============================================================
    # Waypoint loading
    # ==============================================================

    def _load_waypoints(self, waypoint_path):
        """
        Expected format:

        # comment
        # comment
        # s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2
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
            raise ValueError(
                "At least two waypoints are required."
            )

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
    # Physics handling
    # ==============================================================

    def _update_vehicle_params(self):
        """
        Apply current_params using the available API.

        Different F1TENTH versions expose parameter updates
        through slightly different methods.
        """

        params = self.env.sim.agents[0].params.copy()
        params.update(self.current_params)

        # Preferred API in many F1TENTH versions.
        if hasattr(self.env, "update_params"):
            try:
                self.env.update_params(
                    params,
                    index=0,
                )
                return
            except TypeError:
                try:
                    self.env.update_params(
                        params,
                        agent_idx=0,
                    )
                    return
                except TypeError:
                    pass

        # Alternative simulator-level API.
        if hasattr(self.env.sim, "update_params"):
            try:
                self.env.sim.update_params(
                    params,
                    index=0,
                )
                return
            except TypeError:
                try:
                    self.env.sim.update_params(
                        params,
                        agent_idx=0,
                    )
                    return
                except TypeError:
                    pass

        # Fallback for older installations.
        self.env.sim.agents[0].params.update(params)

    # ==============================================================
    # Reset and step
    # ==============================================================

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.current_params = sample_params(
            rng=self.rng
        )

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
            start_pose = np.array(
                [[0.0, 0.0, np.pi / 2.0]],
                dtype=np.float32,
            )

        raw_obs, _, _, info = self.env.reset(
            poses=start_pose
        )

        # Apply randomized physics after reset so that the
        # simulator is not allowed to overwrite the parameters.
        self._update_vehicle_params()

        features = self._get_waypoint_features(
            raw_obs
        )

        self.previous_s = features["s"]
        self.previous_index = features["index"]
        self.previous_action = np.zeros(
            2,
            dtype=np.float32,
        )

        info = dict(info)

        info["physics_params"] = normalize_params(
            self.current_params
        ).copy()

        info["waypoint_index"] = features["index"]
        info["waypoint_s"] = features["s"]
        info["lateral_error"] = features[
            "lateral_error"
        ]
        info["heading_error"] = features[
            "heading_error"
        ]
        info["target_speed"] = features[
            "target_speed"
        ]

        return self._process_obs(raw_obs), info

    def step(self, action):
        action = np.asarray(
            action,
            dtype=np.float32,
        ).reshape(-1)

        if action.shape[0] != 2:
            raise ValueError(
                f"Expected action shape (2,), "
                f"got {action.shape}"
            )

        action = np.clip(
            action,
            self.action_space.low,
            self.action_space.high,
        )

        raw_obs, _, _, info = self.env.step(
            np.array(
                [[action[0], action[1]]],
                dtype=np.float32,
            )
        )

        collision = bool(
            raw_obs["collisions"][0]
        )

        lap_complete = bool(
            raw_obs["lap_counts"][0] >= 1
        )

        terminated = collision or lap_complete
        truncated = False

        reward, reward_info = (
            self._compute_reward(
                raw_obs=raw_obs,
                collision=collision,
                lap_complete=lap_complete,
                action=action,
            )
        )

        info = dict(info)

        info["physics_params"] = normalize_params(
            self.current_params
        ).copy()

        info["lap_time"] = float(
            raw_obs["lap_times"][0]
        )

        info["lap_count"] = int(
            raw_obs["lap_counts"][0]
        )

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

    def _process_obs(self, raw_obs):
        """
        108 LiDAR values + 3 velocity values
        + 6 normalized physics values = 117.
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

        physics = normalize_params(
            self.current_params
        )

        obs = np.concatenate(
            [
                lidar,
                vx,
                vy,
                wz,
                physics,
            ]
        )

        obs = np.nan_to_num(
            obs,
            nan=0.0,
            posinf=30.0,
            neginf=-30.0,
        )

        return obs.astype(np.float32)

    # ==============================================================
    # Pose and waypoint calculations
    # ==============================================================

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
            key
            for key in required_keys
            if key not in raw_obs
        ]

        if missing_keys:
            raise KeyError(
                f"Missing pose keys: {missing_keys}. "
                f"Available keys: {list(raw_obs.keys())}"
            )

        x = float(
            raw_obs["poses_x"][0]
        )

        y = float(
            raw_obs["poses_y"][0]
        )

        psi = float(
            raw_obs["poses_theta"][0]
        )

        return x, y, psi

    def _nearest_waypoint(
        self,
        x,
        y,
        search_window=40,
    ):
        n_waypoints = len(
            self.x_waypoints
        )

        if self.previous_index is None:
            candidate_indices = np.arange(
                n_waypoints
            )
        else:
            offsets = np.arange(
                -search_window,
                search_window + 1,
            )

            candidate_indices = (
                self.previous_index + offsets
            ) % n_waypoints

        dx = (
            self.x_waypoints[candidate_indices]
            - x
        )

        dy = (
            self.y_waypoints[candidate_indices]
            - y
        )

        squared_distances = (
            dx * dx + dy * dy
        )

        local_min_index = int(
            np.argmin(squared_distances)
        )

        index = int(
            candidate_indices[local_min_index]
        )

        distance = float(
            np.sqrt(
                squared_distances[
                    local_min_index
                ]
            )
        )

        return index, distance

    def _get_waypoint_features(self, raw_obs):
        x, y, psi = self._get_vehicle_pose(
            raw_obs
        )

        index, distance = (
            self._nearest_waypoint(x, y)
        )

        waypoint_x = float(
            self.x_waypoints[index]
        )

        waypoint_y = float(
            self.y_waypoints[index]
        )

        waypoint_psi = float(
            self.psi_waypoints[index]
        )

        waypoint_s = float(
            self.s_waypoints[index]
        )

        target_speed = float(
            self.vx_reference[index]
        )

        curvature = float(
            self.kappa_waypoints[index]
        )

        dx = x - waypoint_x
        dy = y - waypoint_y

        normal_x = -np.sin(waypoint_psi)
        normal_y = np.cos(waypoint_psi)

        lateral_error = (
            normal_x * dx
            + normal_y * dy
        )

        heading_error = self._wrap_angle(
            psi - waypoint_psi
        )

        return {
            "index": index,
            "distance": distance,
            "s": waypoint_s,
            "lateral_error": float(
                lateral_error
            ),
            "heading_error": float(
                heading_error
            ),
            "target_speed": target_speed,
            "curvature": curvature,
        }

    # ==============================================================
    # Reward
    # ==============================================================

    def _compute_reward(
        self,
        raw_obs,
        collision,
        lap_complete,
        action,
    ):
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

        features = self._get_waypoint_features(
            raw_obs
        )

        current_s = features["s"]

        if self.previous_s is None:
            previous_s = current_s
        else:
            previous_s = self.previous_s

        delta_s = current_s - previous_s

        # Correct for crossing the start/finish line.
        if delta_s < -0.5 * self.track_length:
            delta_s += self.track_length

        elif delta_s > 0.5 * self.track_length:
            delta_s -= self.track_length

        vx = float(
            raw_obs["linear_vels_x"][0]
        )

        vy = float(
            raw_obs["linear_vels_y"][0]
        )

        wz = float(
            raw_obs["ang_vels_z"][0]
        )

        steering = float(action[0])

        lateral_error = features[
            "lateral_error"
        ]

        heading_error = features[
            "heading_error"
        ]

        target_speed = features[
            "target_speed"
        ]

        # Main racing objective.
        reward_progress = 10.0 * delta_s

        # Track-following terms.
        reward_lateral = (
            -0.05 * abs(lateral_error)
        )

        reward_heading = (
            -0.02 * abs(heading_error)
        )

        # Soft reference-speed tracking.
        reward_speed = (
            -0.005 * abs(vx - target_speed)
        )

        # Stability terms.
        reward_slip = -0.01 * abs(vy)
        reward_yaw = -0.002 * abs(wz)

        # Small steering effort penalty.
        reward_steering = (
            -0.001 * steering * steering
        )

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
            "reward_progress": float(
                reward_progress
            ),
            "reward_lateral": float(
                reward_lateral
            ),
            "reward_heading": float(
                reward_heading
            ),
            "reward_speed": float(
                reward_speed
            ),
            "reward_slip": float(
                reward_slip
            ),
            "reward_yaw": float(
                reward_yaw
            ),
            "reward_steering": float(
                reward_steering
            ),
            "progress_delta": float(
                delta_s
            ),
            "waypoint_index": int(
                features["index"]
            ),
            "waypoint_s": float(
                current_s
            ),
            "lateral_error": float(
                lateral_error
            ),
            "heading_error": float(
                heading_error
            ),
            "target_speed": float(
                target_speed
            ),
            "vehicle_speed": float(
                vx
            ),
            "physics_mu": float(
                self.current_params["mu"]
            ),
            "physics_mass": float(
                self.current_params["m"]
            ),
        }

        return float(reward), reward_info

    # ==============================================================
    # Rendering and cleanup
    # ==============================================================

    def render(self):
        return self.env.render(
            mode="human"
        )

    def close(self):
        self.env.close()