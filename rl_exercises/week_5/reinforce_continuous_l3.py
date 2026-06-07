from typing import Any, List, Tuple

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# =========================
# Config
# =========================


@dataclass
class Config:
    env_name: str = "Pendulum-v1"
    seed: int = 0

    episodes: int = 200
    max_steps: int = 200

    gamma: float = 0.99
    lr: float = 1e-3
    hidden_size: int = 128

    eval_interval: int = 10
    eval_episodes: int = 5

    save_csv: str = "reinforce_continuous_results_l3.csv"
    save_plot: str = "reinforce_continuous_l3.png"


# =========================
# Seed
# =========================


def set_seed(env: gym.Env, seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)


# =========================
# Continuous Policy
# =========================


class ContinuousPolicy(nn.Module):
    """
    Policy for continuous action spaces.

    It outputs:
    - mean action: mu(s)
    - log_std: learnable parameter for exploration
    """

    def __init__(
        self, state_dim: int, action_dim: int, action_high, hidden_size: int = 128
    ):
        super().__init__()

        self.action_high = torch.tensor(action_high, dtype=torch.float32)

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Tanh(),
        )

        # One learnable log standard deviation per action dimension
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if state.dim() == 1:
            state = state.unsqueeze(0)

        # mean in valid action range
        mean = self.net(state) * self.action_high.to(state.device)

        # std must be positive
        std = torch.exp(self.log_std).expand_as(mean)

        return mean, std


# =========================
# REINFORCE Agent
# =========================


class ContinuousREINFORCEAgent:
    def __init__(
        self,
        env: gym.Env,
        lr: float = 1e-3,
        gamma: float = 0.99,
        hidden_size: int = 128,
        seed: int = 0,
    ):
        set_seed(env, seed)

        self.env = env
        self.gamma = gamma

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_high = env.action_space.high

        self.action_low = env.action_space.low
        self.action_high = env.action_space.high

        self.policy = ContinuousPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            action_high=action_high,
            hidden_size=hidden_size,
        )

        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def select_action(self, state: np.ndarray, evaluate: bool = False):
        state_t = torch.tensor(state, dtype=torch.float32)

        mean, std = self.policy(state_t)

        if evaluate:
            action_t = mean
            action = action_t.squeeze(0).detach().numpy()
            action = np.clip(action, self.action_low, self.action_high)
            return action, {}

        dist = torch.distributions.Normal(mean, std)
        raw_action = dist.rsample()

        # log_prob shape: [batch, action_dim]
        log_prob = dist.log_prob(raw_action).sum(dim=-1)

        action = raw_action.squeeze(0).detach().numpy()
        action = np.clip(action, self.action_low, self.action_high)

        return action, {"log_prob": log_prob.squeeze()}

    def compute_returns(self, rewards: List[float]) -> torch.Tensor:
        returns = []
        G = 0.0

        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        return torch.tensor(returns, dtype=torch.float32)

    def update_agent(self, trajectory: List[Tuple[Any, ...]]) -> float:
        rewards = [t[2] for t in trajectory]
        log_probs = [t[5]["log_prob"] for t in trajectory]

        returns = self.compute_returns(rewards)

        # normalize returns to reduce variance
        advantages = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)

        log_probs_t = torch.stack(log_probs)

        loss = -(log_probs_t * advantages).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())

    def evaluate(self, eval_env: gym.Env, num_episodes: int = 5, max_steps: int = 200):
        self.policy.eval()

        returns = []

        for _ in range(num_episodes):
            state, _ = eval_env.reset()
            total_return = 0.0

            for _ in range(max_steps):
                action, _ = self.select_action(state, evaluate=True)
                next_state, reward, terminated, truncated, _ = eval_env.step(action)

                total_return += float(reward)
                state = next_state

                if terminated or truncated:
                    break

            returns.append(total_return)

        self.policy.train()

        return float(np.mean(returns)), float(np.std(returns))

    def train(self, cfg: Config):
        eval_env = gym.make(cfg.env_name)

        results = []

        for ep in range(1, cfg.episodes + 1):
            state, _ = self.env.reset(seed=cfg.seed + ep)
            trajectory = []
            train_return = 0.0

            for _ in range(cfg.max_steps):
                action, info = self.select_action(state, evaluate=False)

                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                trajectory.append(
                    (state, action, float(reward), next_state, done, info)
                )

                train_return += float(reward)
                state = next_state

                if done:
                    break

            loss = self.update_agent(trajectory)

            eval_mean = np.nan
            eval_std = np.nan

            if ep % cfg.eval_interval == 0:
                eval_mean, eval_std = self.evaluate(
                    eval_env,
                    num_episodes=cfg.eval_episodes,
                    max_steps=cfg.max_steps,
                )

                print(
                    f"Ep {ep:4d} | "
                    f"TrainReturn {train_return:8.2f} | "
                    f"EvalReturn {eval_mean:8.2f} ± {eval_std:6.2f} | "
                    f"Loss {loss:8.3f}"
                )

            results.append(
                {
                    "episode": ep,
                    "train_return": train_return,
                    "loss": loss,
                    "eval_mean_return": eval_mean,
                    "eval_std_return": eval_std,
                }
            )

        df = pd.DataFrame(results)
        df.to_csv(cfg.save_csv, index=False)

        eval_env.close()

        print(f"Saved results to {cfg.save_csv}")

        return df


# =========================
# Main
# =========================

if __name__ == "__main__":
    cfg = Config()

    env = gym.make(cfg.env_name)

    # make sure max trajectory length is controlled
    env = gym.wrappers.TimeLimit(
        env.unwrapped,
        max_episode_steps=cfg.max_steps,
    )

    set_seed(env, cfg.seed)

    agent = ContinuousREINFORCEAgent(
        env=env,
        lr=cfg.lr,
        gamma=cfg.gamma,
        hidden_size=cfg.hidden_size,
        seed=cfg.seed,
    )

    df = agent.train(cfg)

    env.close()
