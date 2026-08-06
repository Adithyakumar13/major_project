"""
Collect transitions with deliberate physics variation.
For each episode, sample params from different regimes to ensure
the encoder sees diverse dynamic responses.
"""
import sys
import os
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(BASE, 'src'))

from agent_b_files.wrapper_b import F110WrapperB
from stable_baselines3 import PPO

MAP_PATH      = os.path.join(BASE, 'examples', 'example_map')
CHECKPOINT_B  = os.path.join(BASE, 'checkpoints', 'agent_b')
SAVE_PATH     = os.path.join(BASE, 'src', 'agent_c_files', 'encoder_data.npz')

N_TRANSITIONS = 500_000

# extreme param configs to ensure diverse dynamics
# agent B may not drive perfectly with these, that's fine
PARAM_REGIMES = [
    # (mu,  m,    lf,    lr,    C_Sf, C_Sr)  label
    (0.6,  5.0,  0.13,  0.13,  4.5,  4.5),  # heavy, low friction
    (1.1,  3.0,  0.20,  0.20,  6.5,  6.5),  # light, high friction
    (0.6,  3.0,  0.20,  0.13,  6.5,  4.5),  # understeer config
    (1.1,  5.0,  0.13,  0.20,  4.5,  6.5),  # oversteer config
    (0.85, 4.0,  0.16,  0.16,  5.5,  5.5),  # nominal
]


def set_params(env, mu, m, lf, lr, C_Sf, C_Sr):
    I = 0.04712 * (m / 3.74)
    env.env.sim.agents[0].params.update({
        'mu': mu, 'm': m, 'lf': lf, 'lr': lr,
        'C_Sf': C_Sf, 'C_Sr': C_Sr, 'I': I
    })
    # also update wrapper's current_params so _process_obs is correct
    env.current_params.update({
        'mu': mu, 'm': m, 'lf': lf, 'lr': lr,
        'C_Sf': C_Sf, 'C_Sr': C_Sr, 'I': I
    })


def collect():
    env   = F110WrapperB(map_path=MAP_PATH, map_ext='.png')
    model = PPO.load(CHECKPOINT_B, env=env)

    obs_list      = []
    action_list   = []
    next_obs_list = []

    collected  = 0
    per_regime = N_TRANSITIONS // len(PARAM_REGIMES)

    print(f"Collecting {N_TRANSITIONS} transitions across "
          f"{len(PARAM_REGIMES)} physics regimes...")

    for regime_idx, (mu, m, lf, lr, C_Sf, C_Sr) in enumerate(PARAM_REGIMES):
        print(f"\nRegime {regime_idx+1}/{len(PARAM_REGIMES)}: "
              f"mu={mu}, m={m}, lf={lf}, lr={lr}")

        regime_count = 0
        obs, _ = env.reset()
        set_params(env, mu, m, lf, lr, C_Sf, C_Sr)

        while regime_count < per_regime:
            action, _ = model.predict(obs, deterministic=False)
            next_obs, reward, terminated, truncated, info = env.step(action)

            # only keep transitions where agent is actually steering
            # this filters out boring straight-line segments
            if abs(action[0]) > 0.02 or abs(next_obs[110]) > 0.05:
                obs_list.append(obs.copy())
                action_list.append(action.copy())
                next_obs_list.append(next_obs.copy())
                regime_count += 1
                collected += 1

            if regime_count % 20_000 == 0 and regime_count > 0:
                print(f"  {regime_count}/{per_regime}")

            if terminated or truncated:
                obs, _ = env.reset()
                set_params(env, mu, m, lf, lr, C_Sf, C_Sr)
            else:
                obs = next_obs

    env.close()

    np.savez(
        SAVE_PATH,
        obs=np.array(obs_list,      dtype=np.float32),
        actions=np.array(action_list,   dtype=np.float32),
        next_obs=np.array(next_obs_list, dtype=np.float32),
    )
    print(f"\nSaved {collected} transitions to {SAVE_PATH}")


if __name__ == '__main__':
    collect()