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

from agent_b_files.wrapper_b import F110WrapperB


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

LOG_DIR = os.path.join(
    BASE,
    "logs",
    "agent_b",
)

CHECKPOINT = os.path.join(
    BASE,
    "checkpoints",
    "agent_b_waypoint_reward",
)


def make_env(rank):
    def _init():
        env = F110WrapperB(
            map_path=MAP_PATH,
            waypoint_path=WAYPOINT_PATH,
            map_ext=".png",

            # This must match the coordinate-frame setup
            # used for Agent A.
            reset_to_waypoint_start=False,
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
    )

    model.learn(
        total_timesteps=2_000_000,
        progress_bar=True,
    )

    model.save(CHECKPOINT)

    vec_env.close()

    print(f"Done. Saved to: {CHECKPOINT}")