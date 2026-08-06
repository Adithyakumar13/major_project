import sys
import os

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(BASE, 'src'))

from agent_c_files.wrapper_c import F110WrapperC
from agent_c_files.frozen_physics_extractor import FrozenPhysicsExtractor

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.policies import ActorCriticPolicy

MAP_PATH   = os.path.join(BASE, 'examples', 'example_map')
LOG_DIR    = os.path.join(BASE, 'logs', 'agent_c')
CHECKPOINT = os.path.join(BASE, 'checkpoints', 'agent_c')


def make_env(rank):
    def _init():
        env = F110WrapperC(
            map_path=MAP_PATH,
            map_ext='.png'
        )
        env = Monitor(env)
        return env
    return _init


if __name__ == '__main__':

    N_ENVS = 8
    vec_env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])

    # Continue training if checkpoint exists
    if os.path.exists(CHECKPOINT + ".zip"):
        print(f"Loading existing model from {CHECKPOINT}.zip")
        model = PPO.load(CHECKPOINT, env=vec_env)
        model.set_env(vec_env)
    else:
        print("No checkpoint found. Creating new Agent C.")
        model = PPO(
            ActorCriticPolicy,
            vec_env,
            verbose=1,
            tensorboard_log=LOG_DIR,
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            learning_rate=3e-4,
            policy_kwargs=dict(
                features_extractor_class=FrozenPhysicsExtractor,
                features_extractor_kwargs={},
                net_arch=[64, 64],
            )
        )

    print("Training for another 2,000,000 timesteps...")

    model.learn(
        total_timesteps=2_000_000,
        reset_num_timesteps=False
    )

    model.save(CHECKPOINT)

    print(f"Done. Saved to {CHECKPOINT}.zip")