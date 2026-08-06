import os
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

BASE = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

sys.path.append(
    os.path.join(BASE, "src")
)

from agent_c_files.wrapper_c import F110WrapperC
from agent_c_files.frozen_physics_extractor import FrozenPhysicsExtractor


MAP_PATH = os.path.join(
    BASE,
    "examples",
    "example_map",
)

WAYPOINT_PATH = os.path.join(
    BASE,
    "examples",
    "example_waypoints.csv",
)

ENCODER_PATH = os.path.join(
    BASE,
    "checkpoints",
    "physics_encoder.pt",
)

LOG_DIR = os.path.join(
    BASE,
    "logs",
    "agent_c",
)

CHECKPOINT = os.path.join(
    BASE,
    "checkpoints",
    "agent_c",
)


def make_env(rank):
    def _init():
        env = F110WrapperC(
            map_path=MAP_PATH,
            waypoint_path=WAYPOINT_PATH,
            map_ext=".png",
            reset_to_waypoint_start=False,
            encoder_path=ENCODER_PATH,  # Encoder is loaded in wrapper
            use_encoder=True,
        )

        env = Monitor(env)
        return env

    return _init


if __name__ == "__main__":
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(
        os.path.dirname(CHECKPOINT),
        exist_ok=True,
    )

    N_ENVS = 8

    vec_env = SubprocVecEnv(
        [
            make_env(rank)
            for rank in range(N_ENVS)
        ],
        start_method="spawn",
    )

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        tensorboard_log=LOG_DIR,

        n_steps=2048,
        batch_size=256,
        n_epochs=10,

        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,

        device="auto",

        policy_kwargs=dict(
            features_extractor_class=FrozenPhysicsExtractor,
            features_extractor_kwargs={},  # Empty - no args needed
            net_arch=[512, 256, 128],
        ),
    )

    model.learn(
        total_timesteps=2_000_000,
        progress_bar=True,
    )

    model.save(CHECKPOINT)

    vec_env.close()

    print(f"Done. Saved to: {CHECKPOINT}")