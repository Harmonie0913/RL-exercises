import random
from collections import deque
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    tau: float = 0.005

    actor_lr: float = 1e-4
    critic_lr: float = 1e-3

    buffer_size: int = 100_000
    batch_size: int = 64

    start_steps: int = 1000
    noise_std: float = 0.2

    hidden_dim: int = 256


# =========================
# Replay Buffer
# =========================


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(np.array(actions), dtype=torch.float32),
            torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(1),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(np.array(dones), dtype=torch.float32).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)


# =========================
# Actor Network
# =========================


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_high, hidden_dim=256):
        super().__init__()

        self.action_high = torch.tensor(action_high, dtype=torch.float32)

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, state):
        # tanh gives action in [-1, 1]
        # multiply by action_high to match env action range
        return self.net(state) * self.action_high.to(state.device)


# =========================
# Critic Network
# =========================


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net(x)


# =========================
# Soft Update
# =========================


def soft_update(target_net, online_net, tau):
    for target_param, online_param in zip(
        target_net.parameters(), online_net.parameters()
    ):
        target_param.data.copy_(
            tau * online_param.data + (1.0 - tau) * target_param.data
        )


# =========================
# DDPG Agent
# =========================


class DDPGAgent:
    def __init__(self, env, cfg: Config):
        self.env = env
        self.cfg = cfg

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        self.action_low = env.action_space.low
        self.action_high = env.action_space.high

        self.actor = Actor(
            state_dim,
            action_dim,
            self.action_high,
            cfg.hidden_dim,
        )

        self.critic = Critic(
            state_dim,
            action_dim,
            cfg.hidden_dim,
        )

        self.target_actor = Actor(
            state_dim,
            action_dim,
            self.action_high,
            cfg.hidden_dim,
        )

        self.target_critic = Critic(
            state_dim,
            action_dim,
            cfg.hidden_dim,
        )

        # copy online network weights to target networks
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=cfg.actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=cfg.critic_lr
        )

        self.replay_buffer = ReplayBuffer(cfg.buffer_size)

        self.total_steps = 0

    def select_action(self, state, add_noise=True):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            action = self.actor(state_tensor).squeeze(0).numpy()

        if add_noise:
            noise = np.random.normal(0, self.cfg.noise_std, size=action.shape)
            action = action + noise

        # keep action inside valid range
        action = np.clip(action, self.action_low, self.action_high)

        return action

    def update(self):
        if len(self.replay_buffer) < self.cfg.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.cfg.batch_size
        )

        # =========================
        # Critic update
        # =========================
        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            target_q = self.target_critic(next_states, next_actions)

            y = rewards + self.cfg.gamma * (1.0 - dones) * target_q

        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # =========================
        # Actor update
        # =========================
        predicted_actions = self.actor(states)

        # maximize Q(s, actor(s))
        # PyTorch minimizes, so use negative sign
        actor_loss = -self.critic(states, predicted_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # =========================
        # Target network update
        # =========================
        soft_update(self.target_actor, self.actor, self.cfg.tau)
        soft_update(self.target_critic, self.critic, self.cfg.tau)


# =========================
# Training
# =========================


def train_ddpg(cfg: Config):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    env = gym.make(cfg.env_name)
    env.action_space.seed(cfg.seed)
    env.observation_space.seed(cfg.seed)

    agent = DDPGAgent(env, cfg)

    episode_returns = []

    for episode in range(cfg.episodes):
        state, _ = env.reset(seed=cfg.seed + episode)
        episode_return = 0.0

        for step in range(cfg.max_steps):
            agent.total_steps += 1

            # use random actions at the beginning for better exploration
            if agent.total_steps < cfg.start_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state, add_noise=True)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            episode_return += reward

            agent.update()

            if done:
                break

        episode_returns.append(episode_return)

        if (episode + 1) % 10 == 0:
            print(
                f"Episode {episode + 1:4d} | "
                f"Return: {episode_return:8.2f} | "
                f"Buffer: {len(agent.replay_buffer)}"
            )

    env.close()
    return episode_returns


if __name__ == "__main__":
    cfg = Config()
    returns = train_ddpg(cfg)

    np.savetxt("ddpg_returns_l3.csv", np.array(returns), delimiter=",")
