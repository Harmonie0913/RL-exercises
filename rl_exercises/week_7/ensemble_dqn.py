"""
Use several Q-networks instead of one Q-network.
During training, action selection uses:

    mean_Q(s, a) + beta * std_Q(s, a)

The std term measures disagreement between ensemble members and acts as an
uncertainty-based exploration bonus.
"""

from typing import List, Tuple

from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done, info=None) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)

        states, actions, rewards, next_states, dones = zip(
            *[self.buffer[i] for i in indices]
        )

        states = torch.tensor(np.asarray(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(np.asarray(next_states), dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)


class EnsembleDQNAgent:
    """
    DQN ensemble with uncertainty-based exploration.

    Training action:
        argmax_a mean_Q(s, a) + beta * std_Q(s, a)

    Evaluation action:
        argmax_a mean_Q(s, a)
    """

    def __init__(
        self,
        env,
        num_ensemble: int = 5,
        buffer_capacity: int = 50_000,
        batch_size: int = 64,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_final: float = 0.05,
        epsilon_decay: int = 10_000,
        target_update_freq: int = 1_000,
        hidden_size: int = 128,
        ucb_beta: float = 1.0,
        seed: int = 0,
    ) -> None:
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        self.num_ensemble = num_ensemble
        self.batch_size = batch_size
        self.gamma = gamma

        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.ucb_beta = ucb_beta

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        np.random.seed(seed)
        torch.manual_seed(seed)

        self.q_nets: List[QNetwork] = []
        self.target_q_nets: List[QNetwork] = []
        self.optimizers: List[optim.Optimizer] = []

        for _ in range(num_ensemble):
            q_net = QNetwork(self.obs_dim, self.action_dim, hidden_size).to(self.device)
            target_q_net = QNetwork(self.obs_dim, self.action_dim, hidden_size).to(
                self.device
            )

            target_q_net.load_state_dict(q_net.state_dict())
            target_q_net.eval()

            self.q_nets.append(q_net)
            self.target_q_nets.append(target_q_net)
            self.optimizers.append(optim.Adam(q_net.parameters(), lr=lr))

        self.buffer = ReplayBuffer(buffer_capacity)

        self.frame_idx = 0
        self.update_idx = 0

    def epsilon_by_frame(self, frame_idx: int) -> float:
        epsilon = self.epsilon_final + (
            self.epsilon_start - self.epsilon_final
        ) * np.exp(-frame_idx / self.epsilon_decay)
        return float(epsilon)

    def _ensemble_q_values(
        self, state: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        state_t = torch.tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = torch.stack([net(state_t).squeeze(0) for net in self.q_nets])

        q_mean = q_values.mean(dim=0)
        q_std = q_values.std(dim=0, unbiased=False)

        return q_mean, q_std

    def predict_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        """
        During training:
            epsilon-greedy over UCB Q-values.

        During evaluation:
            greedy over mean Q-values only.
        """
        if not eval_mode:
            self.frame_idx += 1
            epsilon = self.epsilon_by_frame(self.frame_idx)

            if np.random.rand() < epsilon:
                return int(self.env.action_space.sample())

        q_mean, q_std = self._ensemble_q_values(state)

        if eval_mode:
            q_select = q_mean
        else:
            q_select = q_mean + self.ucb_beta * q_std

        return int(torch.argmax(q_select).item())

    def update_agent(self, batch) -> float:
        states, actions, rewards, next_states, dones = batch

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        losses = []

        for i in range(self.num_ensemble):
            q_net = self.q_nets[i]
            target_q_net = self.target_q_nets[i]
            optimizer = self.optimizers[i]

            q_values = q_net(states)
            q_action = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                next_q_values = target_q_net(next_states)
                next_q_max = next_q_values.max(dim=1)[0]
                target = rewards + self.gamma * next_q_max * (1.0 - dones)

            loss = F.mse_loss(q_action, target)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=0.5)
            optimizer.step()

            losses.append(loss.item())

        self.update_idx += 1

        if self.update_idx % self.target_update_freq == 0:
            self.update_target_networks()

        return float(np.mean(losses))

    def update_target_networks(self) -> None:
        for q_net, target_q_net in zip(self.q_nets, self.target_q_nets):
            target_q_net.load_state_dict(q_net.state_dict())
