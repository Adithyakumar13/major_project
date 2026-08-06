"""
Collect transitions with stratified sampling for balanced dynamics.
"""

import sys
import os
import numpy as np
from tqdm import tqdm
import time

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(BASE, 'src'))

from agent_b_files.wrapper_b import F110WrapperB, PARAM_RANGES, BASE_PARAMS
from stable_baselines3 import PPO


# ==============================================================
# Configuration
# ==============================================================

MAP_PATH = os.path.join(BASE, 'examples', 'example_map')
WAYPOINT_PATH = os.path.join(BASE, 'examples', 'example_waypoints.csv')
CHECKPOINT_B = os.path.join(BASE, 'checkpoints', 'agent_b')
SAVE_PATH = os.path.join(BASE, 'src', 'agent_c_files', 'encoder_data.npz')

N_TRANSITIONS = 500_000
DT = 0.01  # Simulator timestep

# Physics regimes (mu, m, lf, lr, C_Sf, C_Sr)
PARAM_REGIMES = [
    (0.6,   5.0,  0.13,  0.13,  4.5,   4.5),   # Heavy, low friction
    (1.1,   3.0,  0.20,  0.20,  6.5,   6.5),   # Light, high friction
    (0.6,   3.0,  0.20,  0.13,  6.5,   4.5),   # Understeer
    (1.1,   5.0,  0.13,  0.20,  4.5,   6.5),   # Oversteer
    (0.85,  4.0,  0.16,  0.16,  5.5,   5.5),   # Nominal
]


def set_params(env, mu, m, lf, lr, C_Sf, C_Sr):
    """Set physics parameters."""
    I = 0.04712 * (m / 3.74)
    params = {'mu': mu, 'm': m, 'lf': lf, 'lr': lr, 'C_Sf': C_Sf, 'C_Sr': C_Sr, 'I': I}
    env.env.sim.agents[0].params.update(params)
    env.current_params.update(params)


def get_keep_prob(acceleration, steering, delta_wz, speed):
    """Stratified sampling based on acceleration magnitude."""
    abs_accel = abs(acceleration)
    
    # Priority 1: Strong acceleration/braking
    if abs_accel > 0.5:
        return 1.0
    elif abs_accel > 0.2:
        return 0.8
    elif abs_accel > 0.05:
        return 0.4
    else:
        # Near zero: only keep 2%
        keep_prob = 0.02
    
    # Priority 2: Cornering
    if abs(steering) > 0.02 or abs(delta_wz) > 0.03:
        keep_prob = max(keep_prob, 0.8)
    
    # Priority 3: High speed
    if speed > 5.0:
        keep_prob = max(keep_prob, 0.3)
    
    return keep_prob


def collect():
    """Main collection function."""
    env = F110WrapperB(map_path=MAP_PATH, waypoint_path=WAYPOINT_PATH, map_ext='.png')
    model = PPO.load(CHECKPOINT_B, env=env)
    
    obs_list, action_list, next_obs_list, physics_list = [], [], [], []
    rng = np.random.default_rng(int(time.time()))
    
    print(f"Collecting {N_TRANSITIONS:,} transitions with stratified sampling...")
    print("-" * 60)
    
    pbar = tqdm(total=N_TRANSITIONS, desc="Collecting")
    collected = 0
    
    while collected < N_TRANSITIONS:
        # Sample physics (80% fixed, 20% random)
        if rng.random() < 0.2:
            params = {key: float(rng.uniform(low, high)) for key, (low, high) in PARAM_RANGES.items()}
            set_params(env, **params)
        else:
            regime = PARAM_REGIMES[rng.integers(len(PARAM_REGIMES))]
            set_params(env, *regime)
            params = dict(zip(['mu', 'm', 'lf', 'lr', 'C_Sf', 'C_Sr'], regime))
        
        obs, _ = env.reset()
        episode_steps = 0
        
        while episode_steps < 5000 and collected < N_TRANSITIONS:
            # Get action with exploration
            action, _ = model.predict(obs, deterministic=False)
            
            # Strong throttle exploration (30% chance)
            if rng.random() < 0.3:
                action[1] += rng.normal(0, 0.08)
                action[1] = np.clip(action[1], 0, 10)
            
            # Steering exploration (20% chance)
            if rng.random() < 0.2:
                action[0] += rng.normal(0, 0.05)
                action[0] = np.clip(action[0], -0.4189, 0.4189)
            
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # Calculate dynamics
            acceleration = (next_obs[108] - obs[108]) / DT  # m/s²
            delta_wz = next_obs[110] - obs[110]
            
            # Stratified sampling
            keep_prob = get_keep_prob(acceleration, action[0], delta_wz, obs[108])
            
            if rng.random() < keep_prob:
                obs_list.append(obs.copy())
                action_list.append(action.copy())
                next_obs_list.append(next_obs.copy())
                physics_list.append(params.copy())
                collected += 1
                pbar.update(1)
            
            episode_steps += 1
            if terminated or truncated:
                break
            obs = next_obs
        
        if collected % 50000 == 0 and collected > 0:
            pbar.set_postfix({'collected': f"{collected:,}"})
    
    pbar.close()
    env.close()
    
    # Save data
    print(f"\nSaving {collected} transitions...")
    np.savez(
        SAVE_PATH,
        obs=np.array(obs_list, dtype=np.float32),
        actions=np.array(action_list, dtype=np.float32),
        next_obs=np.array(next_obs_list, dtype=np.float32),
        physics=np.array([list(p.values()) for p in physics_list], dtype=np.float32),
    )
    
    # Quick summary
    obs_arr = np.array(obs_list)
    next_obs_arr = np.array(next_obs_list)
    acceleration = (next_obs_arr[:, 108] - obs_arr[:, 108]) / DT
    delta_wz = next_obs_arr[:, 110] - obs_arr[:, 110]
    
    print("\n" + "="*60)
    print("Data Collection Summary")
    print("="*60)
    print(f"Total transitions: {len(obs_list):,}")
    print(f"Acceleration range: [{acceleration.min():.2f}, {acceleration.max():.2f}]")
    print(f"Acceleration mean: {acceleration.mean():.4f}, std: {acceleration.std():.4f}")
    print(f"Δwz range: [{delta_wz.min():.4f}, {delta_wz.max():.4f}]")
    print(f"Δwz mean: {delta_wz.mean():.4f}, std: {delta_wz.std():.4f}")
    
    # Check balance
    positive_ratio = (acceleration > 0.05).sum() / len(acceleration) * 100
    negative_ratio = (acceleration < -0.05).sum() / len(acceleration) * 100
    print(f"Positive acceleration (>0.05): {positive_ratio:.1f}%")
    print(f"Negative acceleration (<-0.05): {negative_ratio:.1f}%")
    print(f"Near zero acceleration: {100 - positive_ratio - negative_ratio:.1f}%")
    print("="*60)


if __name__ == '__main__':
    collect()