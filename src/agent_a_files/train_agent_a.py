import sys
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE, 'src'))

from agent_a_files.wrapper_a import F110Wrapper
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

MAP_PATH   = os.path.join(BASE, 'examples', 'example_map')
LOG_DIR    = os.path.join(BASE, 'logs','agent_a')
CHECKPOINT = os.path.join(BASE, 'checkpoints', 'agent_a_reward_2')

def make_env(rank):
    def _init():
        env = F110Wrapper(map_path=MAP_PATH, map_ext='.png')
        env = Monitor(env)
        return env
    return _init

if __name__ == '__main__':
    N_ENVS = 8
    vec_env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=LOG_DIR,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        learning_rate=3e-4,
    )

    model.learn(total_timesteps=2_000_000)
    model.save(CHECKPOINT)
    print(f"Done. Saved to {CHECKPOINT}")