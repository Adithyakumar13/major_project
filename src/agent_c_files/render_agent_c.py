import sys
import os

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(BASE, 'src'))

from agent_c_files.wrapper_c import F110WrapperC
from agent_c_files.frozen_physics_extractor import FrozenPhysicsExtractor
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

MAP_PATH   = os.path.join(BASE, 'examples', 'example_map')
CHECKPOINT = os.path.join(BASE, 'checkpoints', 'agent_c')

env   = F110WrapperC(map_path=MAP_PATH, map_ext='.png')
model = PPO.load(
    CHECKPOINT, env=env,
    custom_objects=dict(
        policy_kwargs=dict(
            features_extractor_class=FrozenPhysicsExtractor,
            features_extractor_kwargs={},
            net_arch=[64, 64],
        )
    )
)

for episode in range(3):
    obs, info = env.reset()
    terminated = False
    total_reward = 0
    steps = 0

    while not terminated and steps < 10000:
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