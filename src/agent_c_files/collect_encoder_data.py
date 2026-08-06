"""
Roll out trained Agent B to collect transition data for encoder training.
Saves (obs_t, action_t, next_obs_t) tuples to disk.
"""
import sys
import os
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(BASE, 'src'))

from agent_b_files.wrapper_b import F110WrapperB
from stable_baselines3 import PPO

MAP_PATH      = os.path.join(BASE, 'examples', 'example_map')
CHECKPOINT_B  = os.path.join(BASE, 'checkpoints', 'agent_b')
SAVE_PATH     = os.path.join(BASE, 'src', 'agent_c_files', 'encoder_data.npz')

N_TRANSITIONS = 200_000


def collect():
    env   = F110WrapperB(map_path=MAP_PATH, map_ext='.png')
    model = PPO.load(CHECKPOINT_B, env=env)

    obs_list      = []
    action_list   = []
    next_obs_list = []

    obs, _ = env.reset()
    collected = 0

    print(f"Collecting {N_TRANSITIONS} transitions...")

    while collected < N_TRANSITIONS:
        action, _ = model.predict(obs, deterministic=False)
        next_obs, reward, terminated, truncated, info = env.step(action)

        if not (terminated or truncated):
            obs_list.append(obs.copy())
            action_list.append(action.copy())
            next_obs_list.append(next_obs.copy())
            collected += 1


        if collected % 10_000 == 0:
            print(f"  {collected}/{N_TRANSITIONS}")

        if terminated or truncated:
            obs, _ = env.reset()
        else:
            obs = next_obs

    env.close()

    np.savez(
        SAVE_PATH,
        obs=np.array(obs_list,      dtype=np.float32),
        actions=np.array(action_list,   dtype=np.float32),
        next_obs=np.array(next_obs_list, dtype=np.float32),
    )
    print(f"Saved {collected} transitions to {SAVE_PATH}")


if __name__ == '__main__':
    collect()