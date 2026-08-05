import os
import numpy as np
import gym
import f110_gym
import gymnasium
from gymnasium import spaces

PARAM_RANGES = {
    'mu':   (0.5,  1.2),
    'm':    (2.5,  5.5),
    'lf':   (0.12, 0.22),
    'lr':   (0.12, 0.22),
    'C_Sf': (4.0,  7.0),
    'C_Sr': (4.0,  7.0),
}

BASE_PARAMS = {
    'mu':   1.0489,
    'm':    3.74,
    'lf':   0.15875,
    'lr':   0.17145,
    'C_Sf': 4.718,
    'C_Sr': 5.4562,
    'I':    0.04712,
}

def sample_params():
    params = {}
    for key, (lo, hi) in PARAM_RANGES.items():
        params[key] = float(np.random.uniform(lo, hi))
    params['I'] = BASE_PARAMS['I'] * (params['m'] / BASE_PARAMS['m'])
    return params

def normalize_params(params):
    normalized = []
    for key, (lo, hi) in PARAM_RANGES.items():
        normalized.append((params[key] - lo) / (hi - lo))
    return np.array(normalized, dtype=np.float32)


class F110WrapperC(gymnasium.Env):

    def __init__(self, map_path, map_ext='.png'):
        super().__init__()

        self.env = gym.make('f110-v0',
                            map=map_path,
                            map_ext=map_ext,
                            num_agents=1,
                            disable_env_checker=True)

        self.current_params = BASE_PARAMS.copy()

        # identical to wrapper B — physics params in obs
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(117,), dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=np.array([-0.4189, 0.0],  dtype=np.float32),
            high=np.array([0.4189, 10.0], dtype=np.float32),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        self.current_params = sample_params()
        self.env.sim.agents[0].params.update(self.current_params)

        raw_obs, _, _, info = self.env.reset(
            poses=np.array([[0.0, 0.0, np.pi/2]])
        )
        return self._process_obs(raw_obs), info

    def step(self, action):
        raw_obs, _, _, info = self.env.step(
                        np.array([[action[0], action[1]]])
                    )
            
        collision = bool(raw_obs["collisions"][0])
        lap_complete = raw_obs["lap_counts"][0] >= 1
            
        terminated = collision or lap_complete
        truncated = False
            
        reward = self._compute_reward(raw_obs, collision, lap_complete)
            
        info["lap_time"] = raw_obs["lap_times"][0]
        info["lap_count"] = raw_obs["lap_counts"][0]
        info["collision"] = collision
            
        return (
            self._process_obs(raw_obs),
            reward,
            terminated,
            truncated,
            info,
        )

    def _process_obs(self, raw_obs):
        lidar  = raw_obs['scans'][0][::10]
        vx     = [raw_obs['linear_vels_x'][0]]
        vy     = [raw_obs['linear_vels_y'][0]]
        wz     = [raw_obs['ang_vels_z'][0]]
        params = normalize_params(self.current_params)
        obs    = np.concatenate([lidar, vx, vy, wz, params])
        obs    = np.nan_to_num(obs, nan=0.0, posinf=30.0, neginf=-30.0)
        obs    = np.clip(obs, -3.4e38, 3.4e38)
        return obs.astype(np.float32)

    def _compute_reward(self, raw_obs, collision, lap_complete):
        if collision:
            return -10.0
        if lap_complete:
            return 100.0
        return 0.1 * float(raw_obs['linear_vels_x'][0]) - 0.001

    def render(self):
        self.env.render(mode='human')

    def close(self):
        self.env.close()