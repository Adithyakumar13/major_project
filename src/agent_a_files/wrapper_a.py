import os
import numpy as np
import gym
import f110_gym
import gymnasium
from gymnasium import spaces


class F110Wrapper(gymnasium.Env):

    def __init__(self, map_path, map_ext='.png'):
        super().__init__()

        self.env = gym.make('f110-v0',
                            map=map_path,
                            map_ext=map_ext,
                            num_agents=1,
                            disable_env_checker=True)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(111,), dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=np.array([-0.4189, 0.0],  dtype=np.float32),
            high=np.array([0.4189, 10.0],  dtype=np.float32),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        #for example_map
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
        lidar = raw_obs['scans'][0][::10]       # (108,)
        vx    = [raw_obs['linear_vels_x'][0]]   # (1,)
        vy    = [raw_obs['linear_vels_y'][0]]   # (1,)
        wz    = [raw_obs['ang_vels_z'][0]]      # (1,)
        return np.concatenate([lidar, vx, vy, wz]).astype(np.float32)

    def _compute_reward(self, raw_obs, collision, lap_complete):

        if collision:
            return -10.0

        if lap_complete:
            return 100.0

        vx = raw_obs["linear_vels_x"][0]
        return 0.1 * float(vx) - 0.001

    def render(self):
        self.env.render(mode='human')

    def close(self):
        self.env.close()