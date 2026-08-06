import os
import sys
import numpy as np

from stable_baselines3 import PPO


BASE = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

sys.path.append(os.path.join(BASE, "src"))

from agent_a_files.wrapper_a import F110Wrapper


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

CHECKPOINT = os.path.join(
    BASE,
    "checkpoints",
    "agent_a_waypoint_reward",
)


env = F110Wrapper(
    map_path=MAP_PATH,
    waypoint_path=WAYPOINT_PATH,
    map_ext=".png",

    # This must match the setting used during training.
    reset_to_waypoint_start=False,
)

model = PPO.load(
    CHECKPOINT,
    env=env,
)

print("Observation space:", env.observation_space)
print("Action space:", env.action_space)

for episode in range(3):
    obs, info = env.reset()

    terminated = False
    truncated = False
    total_reward = 0.0
    steps = 0
    max_steps = 10_000

    while not (terminated or truncated) and steps < max_steps:
        action, _ = model.predict(
            obs,
            deterministic=True,
        )

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += float(reward)
        steps += 1

        env.render()

    print(
        f"Episode {episode + 1}: "
        f"steps={steps}, "
        f"reward={total_reward:.2f}, "
        f"lap_time={info.get('lap_time', float('nan')):.2f}s, "
        f"lap_count={info.get('lap_count', -1)}, "
        f"collision={info.get('collision', None)}, "
        f"terminated={terminated}, "
        f"truncated={truncated}"
    )

env.close()