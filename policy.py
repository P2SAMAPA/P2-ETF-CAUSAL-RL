"""policy.py — PPO Actor-Critic for Causal RL.

Pure PyTorch implementation — no stable-baselines or RLlib dependency.

Architecture
------------
  Shared trunk: MLP(obs_dim → hidden[-1])
  Actor head:   Linear(hidden[-1] → action_dim)  → action logits
  Critic head:  Linear(hidden[-1] → 1)            → state value V(s)

PPO update
----------
  1. Collect rollout_steps of (s, a, r, s', done) from env
  2. Compute GAE advantages and returns
  3. For PPO_EPOCHS_PER_UPDATE epochs over minibatches:
     a. ratio = exp(log_pi(a|s) - log_pi_old(a|s))
     b. actor_loss  = -min(ratio * A, clip(ratio, 1±eps) * A)
     c. critic_loss = MSE(V(s), returns)
     d. entropy_loss = -H[pi]
     e. total = actor + VALUE_COEF * critic - ENTROPY_COEF * entropy
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

import config


# ── Shared MLP trunk ──────────────────────────────────────────────────────────

def _make_mlp(in_dim: int, hidden: list[int], out_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.Tanh()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


# ── Actor-Critic network ──────────────────────────────────────────────────────

class ActorCritic(nn.Module):
    """Shared-trunk PPO actor-critic.

    Outputs Gaussian action distribution (continuous action space).
    """

    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        hidden = config.PPO_HIDDEN

        # Shared feature extractor
        self.trunk = _make_mlp(obs_dim, hidden[:-1], hidden[-1])

        # Actor: mean of Gaussian
        self.actor_mean = nn.Linear(hidden[-1], action_dim)
        # Log std: learnable parameter (not input-dependent)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic: scalar value
        self.critic = nn.Linear(hidden[-1], 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        # Smaller init for actor output
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)

    def forward(
        self,
        obs: torch.Tensor,                          # (B, obs_dim)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (action_mean, log_std, value)."""
        feat   = self.trunk(obs)
        mean   = self.actor_mean(feat)
        value  = self.critic(feat).squeeze(-1)
        return mean, self.log_std.expand_as(mean), value

    def get_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample action and compute log probability.

        Returns (action, log_prob).
        """
        mean, log_std, _ = self.forward(obs)
        std = log_std.exp()
        dist = Normal(mean, std)
        if deterministic:
            action = mean
        else:
            action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)    # sum over action dims
        return action, log_prob

    def evaluate(
        self,
        obs:    torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate stored actions → (log_prob, entropy, value)."""
        mean, log_std, value = self.forward(obs)
        std  = log_std.exp()
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy  = dist.entropy().sum(dim=-1)
        return log_prob, entropy, value


# ── Rollout buffer ────────────────────────────────────────────────────────────

class RolloutBuffer:
    """Fixed-size buffer for PPO rollout storage."""

    def __init__(
        self,
        n_steps:   int,
        obs_dim:   int,
        action_dim: int,
        device:    torch.device,
    ) -> None:
        self.n      = n_steps
        self.device = device
        self.obs         = torch.zeros(n_steps, obs_dim,    device=device)
        self.actions     = torch.zeros(n_steps, action_dim, device=device)
        self.log_probs   = torch.zeros(n_steps,             device=device)
        self.rewards     = torch.zeros(n_steps,             device=device)
        self.values      = torch.zeros(n_steps,             device=device)
        self.dones       = torch.zeros(n_steps,             device=device)
        self.advantages  = torch.zeros(n_steps,             device=device)
        self.returns     = torch.zeros(n_steps,             device=device)
        self.ptr         = 0

    def add(
        self,
        obs:      np.ndarray,
        action:   np.ndarray,
        log_prob: float,
        reward:   float,
        value:    float,
        done:     bool,
    ) -> None:
        i = self.ptr % self.n
        self.obs[i]       = torch.tensor(obs,      dtype=torch.float32, device=self.device)
        self.actions[i]   = torch.tensor(action,   dtype=torch.float32, device=self.device)
        self.log_probs[i] = log_prob
        self.rewards[i]   = reward
        self.values[i]    = value
        self.dones[i]     = float(done)
        self.ptr         += 1

    def full(self) -> bool:
        return self.ptr >= self.n

    def compute_gae(
        self,
        last_value: float,
        gamma: float = config.PPO_GAMMA,
        lam:   float = config.PPO_GAE_LAMBDA,
    ) -> None:
        """Compute GAE advantages and discounted returns."""
        adv = 0.0
        for t in reversed(range(self.n)):
            next_val  = last_value if t == self.n - 1 else self.values[t + 1].item()
            next_done = self.dones[t].item()
            delta     = (self.rewards[t].item()
                         + gamma * next_val * (1 - next_done)
                         - self.values[t].item())
            adv       = delta + gamma * lam * (1 - next_done) * adv
            self.advantages[t] = adv
        self.returns = self.advantages + self.values
        # Normalise advantages
        self.advantages = (self.advantages - self.advantages.mean()) / (
            self.advantages.std() + 1e-8
        )

    def get_batches(self, batch_size: int):
        """Yield random minibatches."""
        idx = torch.randperm(self.n, device=self.device)
        for start in range(0, self.n, batch_size):
            b = idx[start: start + batch_size]
            yield (
                self.obs[b],
                self.actions[b],
                self.log_probs[b],
                self.advantages[b],
                self.returns[b],
            )

    def reset(self) -> None:
        self.ptr = 0


# ── PPO trainer ───────────────────────────────────────────────────────────────

class PPOTrainer:
    """PPO training loop."""

    def __init__(
        self,
        obs_dim:    int,
        action_dim: int,
        device:     torch.device,
    ) -> None:
        self.device = device
        self.policy = ActorCritic(obs_dim, action_dim).to(device)
        self.optimizer = optim.Adam(
            self.policy.parameters(), lr=config.PPO_LR, eps=1e-5
        )
        self.buffer = RolloutBuffer(
            n_steps=config.PPO_ROLLOUT_STEPS,
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=device,
        )

    @torch.no_grad()
    def collect_rollout(self, env) -> dict:
        """Collect PPO_ROLLOUT_STEPS transitions from env."""
        self.policy.eval()
        self.buffer.reset()

        obs, _ = env.reset()
        episode_rewards: list[float] = []
        ep_ret = 0.0
        ep_count = 0

        for _ in range(config.PPO_ROLLOUT_STEPS):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            action, log_prob = self.policy.get_action(obs_t)
            _, _, value      = self.policy.forward(obs_t)

            act_np = action.squeeze(0).cpu().numpy()
            obs_next, reward, terminated, truncated, _ = env.step(act_np)

            self.buffer.add(
                obs=obs,
                action=act_np,
                log_prob=log_prob.item(),
                reward=float(reward),
                value=value.item(),
                done=bool(terminated or truncated),
            )
            ep_ret += reward
            obs     = obs_next

            if terminated or truncated:
                episode_rewards.append(ep_ret)
                ep_ret = 0.0
                ep_count += 1
                obs, _ = env.reset()

        # Bootstrap last value
        obs_t   = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        _, _, last_val = self.policy.forward(obs_t)
        self.buffer.compute_gae(last_value=last_val.item())
        # Normalise returns for stable critic learning
        ret_std = self.buffer.returns.std() + 1e-8
        self.buffer.returns = (self.buffer.returns - self.buffer.returns.mean()) / ret_std

        return {
            "mean_ep_ret": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
            "n_episodes":  ep_count,
        }

    def update(self) -> dict:
        """Run PPO_EPOCHS_PER_UPDATE gradient steps over the buffer."""
        self.policy.train()
        losses_actor, losses_critic, losses_entropy = [], [], []

        for _ in range(config.PPO_EPOCHS_PER_UPDATE):
            for obs_b, act_b, lp_old_b, adv_b, ret_b in self.buffer.get_batches(
                config.PPO_BATCH_SIZE
            ):
                log_prob, entropy, value = self.policy.evaluate(obs_b, act_b)

                # PPO clipped objective
                ratio      = torch.exp(log_prob - lp_old_b)
                surr1      = ratio * adv_b
                surr2      = torch.clamp(ratio, 1 - config.PPO_CLIP_EPS,
                                          1 + config.PPO_CLIP_EPS) * adv_b
                actor_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                critic_loss = nn.functional.mse_loss(value, ret_b)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                loss = (actor_loss
                        + config.PPO_VALUE_COEF  * critic_loss
                        + config.PPO_ENTROPY_COEF * entropy_loss)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(), config.PPO_GRAD_CLIP
                )
                self.optimizer.step()

                losses_actor.append(actor_loss.item())
                losses_critic.append(critic_loss.item())
                losses_entropy.append(entropy_loss.item())

        return {
            "actor_loss":   float(np.mean(losses_actor)),
            "critic_loss":  float(np.mean(losses_critic)),
            "entropy_loss": float(np.mean(losses_entropy)),
        }
