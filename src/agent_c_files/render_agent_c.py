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

CHECKPOINT = os.path.join(
    BASE,
    "checkpoints",
    "agent_c_waypoint_reward",
)


env = F110WrapperC(
    map_path=MAP_PATH,
    waypoint_path=WAYPOINT_PATH,
    map_ext=".png",
    reset_to_waypoint_start=False,
    encoder_path=ENCODER_PATH,
    use_encoder=True,
)

model = PPO.load(
    CHECKPOINT,
    env=env,
    custom_objects={
        "policy_kwargs": dict(
            features_extractor_class=FrozenPhysicsExtractor,
            features_extractor_kwargs={},  # Empty
            net_arch=[512, 256, 128],
        )
    }
)

print("Observation space:", env.observation_space)
print("Action space:", env.action_space)
print("Embedding dimension:", env.embedding_dim)

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

    # Get physics parameters for this episode
    params = env.current_params
    
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
    
    # Print physics parameters for this episode
    print(
        f"  Physics: mu={params['mu']:.3f}, "
        f"mass={params['m']:.3f}, "
        f"lf={params['lf']:.3f}, "
        f"lr={params['lr']:.3f}, "
        f"C_Sf={params['C_Sf']:.3f}, "
        f"C_Sr={params['C_Sr']:.3f}, "
        f"I={params['I']:.5f}"
    )
    
    # Also show normalized physics from info
    if 'physics_params' in info:
        phys_norm = info['physics_params']
        print(
            f"  Normalized: mu={phys_norm[0]:.3f}, "
            f"mass={phys_norm[1]:.3f}, "
            f"lf={phys_norm[2]:.3f}, "
            f"lr={phys_norm[3]:.3f}, "
            f"C_Sf={phys_norm[4]:.3f}, "
            f"C_Sr={phys_norm[5]:.3f}"
        )
    
    # Show physics embedding (from encoder)
    if 'physics_embedding' in info:
        embedding = info['physics_embedding']
        print(f"  Embedding: [{', '.join([f'{x:.3f}' for x in embedding[:4]])}...]")
    
    print()  # Empty line between episodes

env.close()