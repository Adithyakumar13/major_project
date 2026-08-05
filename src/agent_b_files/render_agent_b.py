import sys
import os

BASE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
sys.path.append(os.path.join(BASE, 'src'))
sys.path.append(os.path.join(BASE, "src"))

from agent_b_files.wrapper_b import F110WrapperB
from stable_baselines3 import PPO

MAP_PATH = os.path.join(BASE, "examples", "example_map")
CHECKPOINT = os.path.join(BASE, "checkpoints", "agent_b")

# Create environment
env = F110WrapperB(
    map_path=MAP_PATH,
    map_ext=".png"
)

# Load trained model
model = PPO.load(CHECKPOINT, env=env)

NUM_EPISODES = 3

for episode in range(NUM_EPISODES):

    obs, info = env.reset()

    terminated = False
    truncated = False

    total_reward = 0.0
    steps = 0

    while not (terminated or truncated):

        action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        steps += 1

        env.render()

    params = env.current_params

    print("\n========================================")
    print(f"Episode {episode + 1}")
    print("========================================")
    print(f"Steps        : {steps}")
    print(f"Reward       : {total_reward:.2f}")
    print(f"Lap Time     : {info['lap_time']:.2f} s")
    print(f"Lap Count    : {info['lap_count']}")
    print(f"Collision    : {info['collision']}")
    print()
    print("Vehicle Parameters")
    print("------------------")
    print(f"mu           : {params['mu']:.3f}")
    print(f"mass         : {params['m']:.3f}")
    print(f"lf           : {params['lf']:.3f}")
    print(f"lr           : {params['lr']:.3f}")
    print(f"C_Sf         : {params['C_Sf']:.3f}")
    print(f"C_Sr         : {params['C_Sr']:.3f}")

env.close()