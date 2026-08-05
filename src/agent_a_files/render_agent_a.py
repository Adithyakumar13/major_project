import sys
import os

BASE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
sys.path.append(os.path.join(BASE, 'src'))

from agent_a_files.wrapper_a import F110Wrapper
from stable_baselines3 import PPO

MAP_PATH   = os.path.join(BASE, 'examples', 'example_map')
CHECKPOINT = os.path.join(BASE, 'checkpoints', 'agent_a_reward_2')

env = F110Wrapper(map_path=MAP_PATH, map_ext='.png')
model = PPO.load(CHECKPOINT, env=env)

for episode in range(3):
    obs, info = env.reset()
    terminated = False
    total_reward = 0
    steps = 0
    max_steps = 10000  # safety cutoff

    while not terminated and steps < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print("terminated =", terminated)
            print("truncated  =", truncated)
            print("lap_time   =", info["lap_time"])
            print("lap_count  =", info["lap_count"])
            print("collision  =", info["collision"])
            
        total_reward += reward
        steps += 1
        env.render()

    print(
    f"Episode {episode}: "
    f"steps={steps}, "
    f"reward={total_reward:.2f}, "
    f"lap_time={info['lap_time']:.2f}s"
    )

env.close()