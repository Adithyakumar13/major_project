import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.type_aliases import RolloutBufferSamples
from gymnasium import spaces
from typing import Generator

from dataclasses import dataclass
from torch import Tensor

@dataclass
class RolloutBufferSamplesWithNext:
    observations:      Tensor
    actions:           Tensor
    old_values:        Tensor
    old_log_prob:      Tensor
    advantages:        Tensor
    returns:           Tensor
    next_observations: Tensor
    non_terminal:      Tensor


# ── Encoder ──────────────────────────────────────────────────────────────────

class PhysicsEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class DynamicsPredictor(nn.Module):
    """
    Input:  embedding(8) + vx,vy,wz(3) + action(2) = 13
    Output: predicted next vy, wz (2,)
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(13, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, embedding, vxvywz, action):
        x = torch.cat([embedding, vxvywz, action], dim=-1)
        return self.net(x)


# ── Features extractor ───────────────────────────────────────────────────────

class PhysicsInformedExtractor(BaseFeaturesExtractor):
    """
    obs layout (117,):
        [0:108]   lidar
        [108:111] vx, vy, wz
        [111:117] normalized physics params
    output: lidar_feat(64) + embedding(8) = 72
    """
    def __init__(self, observation_space: spaces.Box):
        super().__init__(observation_space, features_dim=72)

        self.lidar_encoder = nn.Sequential(
            nn.Linear(108, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.physics_encoder  = PhysicsEncoder()
        self.dynamics_predictor = DynamicsPredictor()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        lidar  = obs[:, :108]
        params = obs[:, 111:117]
        lidar_feat = self.lidar_encoder(lidar)
        embedding  = self.physics_encoder(params)
        return torch.cat([lidar_feat, embedding], dim=-1)

    def encode_physics(self, obs: torch.Tensor) -> torch.Tensor:
        return self.physics_encoder(obs[:, 111:117])

    def predict_dynamics(self, obs: torch.Tensor,
                         action: torch.Tensor) -> torch.Tensor:
        embedding = self.encode_physics(obs)
        vxvywz    = obs[:, 108:111]
        return self.dynamics_predictor(embedding, vxvywz, action)


# ── Custom rollout buffer ─────────────────────────────────────────────────────

class RolloutBufferWithNextObs(RolloutBuffer):
    """
    Extends SB3 RolloutBuffer to store next_observations and
    a terminal mask (1 = valid transition, 0 = episode ended).
    """

    def reset(self):
        super().reset()
        self.next_observations = np.zeros(
            (self.buffer_size, self.n_envs, *self.obs_shape),
            dtype=np.float32
        )
        # 1 = non-terminal, 0 = terminal (no valid next state)
        self.non_terminal = np.zeros(
            (self.buffer_size, self.n_envs), dtype=np.float32
        )

    def add(self, obs, action, reward, episode_start, value, log_prob,
            next_obs=None, non_terminal=None):
        if next_obs is not None:
            self.next_observations[self.pos] = next_obs.copy()
        if non_terminal is not None:
            self.non_terminal[self.pos] = non_terminal.copy()
        super().add(obs, action, reward, episode_start, value, log_prob)

    def get(self, batch_size: int = None) -> Generator:
        assert self.full
        indices = np.random.permutation(self.buffer_size * self.n_envs)

        if not self.generator_ready:
            for tensor in ["observations", "actions", "values",
                           "log_probs", "advantages", "returns"]:
                self.__dict__[tensor] = self.swap_and_flatten(
                    self.__dict__[tensor]
                )
            self._next_obs_flat = self.swap_and_flatten(self.next_observations)
            self._non_terminal_flat = self.swap_and_flatten(
                self.non_terminal
            )
            self.generator_ready = True

        # fix 6: handle batch_size=None
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples_with_next(
                indices[start_idx:start_idx + batch_size]
            )
            start_idx += batch_size

    def _get_samples_with_next(self, batch_inds):
        return RolloutBufferSamplesWithNext(
            observations      = self.to_torch(self.observations[batch_inds]),
            actions           = self.to_torch(self.actions[batch_inds]),
            old_values        = self.to_torch(self.values[batch_inds].flatten()),
            old_log_prob      = self.to_torch(self.log_probs[batch_inds].flatten()),
            advantages        = self.to_torch(self.advantages[batch_inds].flatten()),
            returns           = self.to_torch(self.returns[batch_inds].flatten()),
            next_observations = self.to_torch(self._next_obs_flat[batch_inds]),
            non_terminal      = self.to_torch(self._non_terminal_flat[batch_inds]),
        )


# ── Custom PPO ───────────────────────────────────────────────────────────────

class PPOC(PPO):
    """
    PPO + physics-informed auxiliary dynamics prediction loss.
    """
    AUX_LOSS_WEIGHT = 0.1

    def _setup_model(self):
        super()._setup_model()
        self.rollout_buffer = RolloutBufferWithNextObs(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
        assert self._last_obs is not None
        self.policy.set_training_mode(False)
        n_steps = 0
        rollout_buffer.reset()
        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            with torch.no_grad():
                obs_tensor = torch.as_tensor(
                    self._last_obs, device=self.device
                ).float()
                actions, values, log_probs = self.policy(obs_tensor)

            actions_np = actions.cpu().numpy()
            new_obs, rewards, dones, infos = env.step(actions_np)

            self.num_timesteps += env.num_envs
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.reshape(-1, 1)

            # fix 3: use terminal_observation for true last state
            # fix 4: mark terminal transitions so aux loss skips them
            next_obs_stored = new_obs.copy()
            non_terminal    = 1.0 - dones.astype(np.float32)

            for idx, done in enumerate(dones):
                if done:
                    terminal_obs = infos[idx].get('terminal_observation')
                    if terminal_obs is not None:
                        next_obs_stored[idx] = terminal_obs
                    else:
                        # no terminal obs available — mark as terminal
                        non_terminal[idx] = 0.0

                    # bootstrap value for truncated episodes
                    if infos[idx].get('TimeLimit.truncated', False):
                        terminal_obs_tensor = self.policy.obs_to_tensor(
                            infos[idx]['terminal_observation']
                        )[0]
                        with torch.no_grad():
                            terminal_value = self.policy.predict_values(
                                terminal_obs_tensor
                            )[0]
                        rewards[idx] += self.gamma * terminal_value.item()

            rollout_buffer.add(
                self._last_obs,
                actions_np,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
                next_obs=next_obs_stored,
                non_terminal=non_terminal,
            )

            self._last_obs           = new_obs
            self._last_episode_starts = dones

        with torch.no_grad():
            values = self.policy.predict_values(
                torch.as_tensor(new_obs, device=self.device).float()
            )
        rollout_buffer.compute_returns_and_advantage(
            last_values=values, dones=dones
        )
        callback.on_rollout_end()
        return True

    def train(self):
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        all_pg_losses, all_value_losses = [], []
        all_entropy_losses, all_aux_losses, all_losses = [], [], []
        clip_fractions, approx_kl_divs = [], []
        continue_training = True
        grad_norm_logged  = False  # fix 7: log encoder grad norm once per train()

        for epoch in range(self.n_epochs):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions
                )
                values = values.flatten()

                advantages = rollout_data.advantages
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )

                ratio     = torch.exp(log_prob - rollout_data.old_log_prob)
                pg_loss_1 = advantages * ratio
                pg_loss_2 = advantages * torch.clamp(
                    ratio, 1 - self.clip_range, 1 + self.clip_range
                )
                pg_loss = -torch.min(pg_loss_1, pg_loss_2).mean()

                clip_fractions.append(
                    torch.mean(
                        (torch.abs(ratio - 1) > self.clip_range).float()
                    ).item()
                )

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + torch.clamp(
                        values - rollout_data.old_values,
                        -self.clip_range_vf, self.clip_range_vf
                    )
                value_loss   = F.mse_loss(rollout_data.returns, values_pred)
                entropy_loss = -torch.mean(entropy)

                # ── auxiliary loss ──────────────────────────────────────────
                extractor = self.policy.features_extractor
                predicted = extractor.predict_dynamics(
                    rollout_data.observations, actions
                )
                next_obs     = rollout_data.next_observations
                target_vy    = next_obs[:, 109:110]
                target_wz    = next_obs[:, 110:111]
                target       = torch.cat([target_vy, target_wz], dim=-1)

                # fix 4: mask out terminal transitions
                mask     = rollout_data.non_terminal.unsqueeze(-1)  # (batch,1)
                aux_loss = (F.mse_loss(predicted, target, reduction='none')
                            * mask).mean()
                # ───────────────────────────────────────────────────────────

                loss = (pg_loss
                        + self.ent_coef   * entropy_loss
                        + self.vf_coef    * value_loss
                        + self.AUX_LOSS_WEIGHT * aux_loss)

                with torch.no_grad():
                    log_ratio     = log_prob - rollout_data.old_log_prob
                    approx_kl_div = torch.mean(
                        (torch.exp(log_ratio) - 1) - log_ratio
                    ).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if (self.target_kl is not None
                        and approx_kl_div > 1.5 * self.target_kl):
                    continue_training = False
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()

                # fix 7: verify encoder is receiving gradients
                if not grad_norm_logged:
                    enc_weight = extractor.physics_encoder.net[0].weight
                    if enc_weight.grad is not None:
                        grad_norm = enc_weight.grad.norm().item()
                        self.logger.record(
                            "train/encoder_grad_norm", grad_norm
                        )
                    grad_norm_logged = True

                nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.policy.optimizer.step()

                all_pg_losses.append(pg_loss.item())
                all_value_losses.append(value_loss.item())
                all_entropy_losses.append(entropy_loss.item())
                all_aux_losses.append(aux_loss.item())
                all_losses.append(loss.item())

            if not continue_training:
                break

        self._n_updates += self.n_epochs

        # fix 5: log mean loss over all minibatches, not last batch
        self.logger.record("train/entropy_loss",        np.mean(all_entropy_losses))
        self.logger.record("train/policy_gradient_loss",np.mean(all_pg_losses))
        self.logger.record("train/value_loss",          np.mean(all_value_losses))
        self.logger.record("train/aux_loss",            np.mean(all_aux_losses))
        self.logger.record("train/loss",                np.mean(all_losses))
        self.logger.record("train/approx_kl",          np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction",       np.mean(clip_fractions))
        self.logger.record("train/n_updates",           self._n_updates,
                           exclude="tensorboard")