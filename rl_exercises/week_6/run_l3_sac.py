from typing import Dict, List, Tuple

import argparse
import os
import random

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rliable import library as rly
from torch.distributions import Normal


def parse_args():
    parser = argparse.ArgumentParser(description="Run SAC experiments for L3.")

    parser.add_argument("--env-name", type=str, default="Pendulum-v1")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])

    parser.add_argument("--total-steps", type=int, default=100_000)
    parser.add_argument("--eval-interval", type=int, default=5_000)
    parser.add_argument("--eval-episodes", type=int, default=5)

    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--replay-size", type=int, default=1_000_000)

    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)

    parser.add_argument("--lr-actor", type=float, default=3e-4)
    parser.add_argument("--lr-critic", type=float, default=3e-4)

    parser.add_argument("--start-steps", type=int, default=5_000)
    parser.add_argument("--updates-per-step", type=int, default=1)

    parser.add_argument("--result-dir", type=str, default="results_l3")
    parser.add_argument("--plot-dir", type=str, default="plots_l3")

    parser.add_argument("--ppo-result-dir", type=str, default="results_l3")
    parser.add_argument("--ci-reps", type=int, default=2000)

    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-plot", action="store_true")

    return parser.parse_args()


def set_seed(env: gym.Env, seed: int) -> None:
    env.reset(seed=seed)

    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)

    if hasattr(env.observation_space, "seed"):
        env.observation_space.seed(seed)

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        capacity: int,
    ):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: float,
    ) -> None:
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        idxs = np.random.randint(0, self.size, size=batch_size)

        states = torch.tensor(self.states[idxs], dtype=torch.float32)
        actions = torch.tensor(self.actions[idxs], dtype=torch.float32)
        rewards = torch.tensor(self.rewards[idxs], dtype=torch.float32)
        next_states = torch.tensor(self.next_states[idxs], dtype=torch.float32)
        dones = torch.tensor(self.dones[idxs], dtype=torch.float32)

        return states, actions, rewards, next_states, dones


class GaussianPolicy(nn.Module):
    """
    Tanh-squashed Gaussian policy for continuous SAC.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        hidden_size: int = 256,
    ):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        self.mean = nn.Linear(hidden_size, action_dim)
        self.log_std = nn.Linear(hidden_size, action_dim)

        action_low_t = torch.tensor(action_low, dtype=torch.float32)
        action_high_t = torch.tensor(action_high, dtype=torch.float32)

        self.register_buffer("action_scale", (action_high_t - action_low_t) / 2.0)
        self.register_buffer("action_bias", (action_high_t + action_low_t) / 2.0)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)

        return mean, log_std

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(state)
        std = log_std.exp()

        dist = Normal(mean, std)
        raw_action = dist.rsample()

        tanh_action = torch.tanh(raw_action)
        action = tanh_action * self.action_scale + self.action_bias

        log_prob = dist.log_prob(raw_action)

        # Tanh correction.
        log_prob -= torch.log(self.action_scale * (1.0 - tanh_action.pow(2)) + 1e-6)

        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob

    def deterministic(self, state: torch.Tensor) -> torch.Tensor:
        mean, _ = self.forward(state)
        tanh_action = torch.tanh(mean)
        action = tanh_action * self.action_scale + self.action_bias
        return action


class QNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_size: int = 256,
    ):
        super().__init__()

        self.fc1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return self.out(x)


class SACAgent:
    def __init__(
        self,
        env: gym.Env,
        hidden_size: int,
        replay_size: int,
        batch_size: int,
        gamma: float,
        tau: float,
        alpha: float,
        lr_actor: float,
        lr_critic: float,
    ):
        self.env = env

        self.state_dim = int(np.prod(env.observation_space.shape))
        self.action_dim = int(np.prod(env.action_space.shape))

        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        self.policy = GaussianPolicy(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            action_low=env.action_space.low,
            action_high=env.action_space.high,
            hidden_size=hidden_size,
        )

        self.q1 = QNetwork(self.state_dim, self.action_dim, hidden_size)
        self.q2 = QNetwork(self.state_dim, self.action_dim, hidden_size)

        self.q1_target = QNetwork(self.state_dim, self.action_dim, hidden_size)
        self.q2_target = QNetwork(self.state_dim, self.action_dim, hidden_size)

        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr_actor)
        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=lr_critic)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=lr_critic)

        self.replay_buffer = ReplayBuffer(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            capacity=replay_size,
        )

    def select_action(
        self,
        state: np.ndarray,
        evaluate: bool = False,
    ) -> np.ndarray:
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            if evaluate:
                action = self.policy.deterministic(state_t)
            else:
                action, _ = self.policy.sample(state_t)

        return action.squeeze(0).numpy()

    def update(self) -> Tuple[float, float, float]:
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        with torch.no_grad():
            next_actions, next_log_probs = self.policy.sample(next_states)

            q1_next = self.q1_target(next_states, next_actions)
            q2_next = self.q2_target(next_states, next_actions)

            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs

            target_q = rewards + self.gamma * (1.0 - dones) * q_next

        q1_pred = self.q1(states, actions)
        q2_pred = self.q2(states, actions)

        q1_loss = F.mse_loss(q1_pred, target_q)
        q2_loss = F.mse_loss(q2_pred, target_q)

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        new_actions, log_probs = self.policy.sample(states)

        q1_new = self.q1(states, new_actions)
        q2_new = self.q2(states, new_actions)

        q_new = torch.min(q1_new, q2_new)

        policy_loss = (self.alpha * log_probs - q_new).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.soft_update(self.q1, self.q1_target)
        self.soft_update(self.q2, self.q2_target)

        return (
            float(policy_loss.item()),
            float(q1_loss.item()),
            float(q2_loss.item()),
        )

    def soft_update(self, source: nn.Module, target: nn.Module) -> None:
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1.0 - self.tau) * target_param.data
            )

    def evaluate(self, eval_env: gym.Env, eval_episodes: int) -> Tuple[float, float]:
        returns = []

        for _ in range(eval_episodes):
            state, _ = eval_env.reset()
            done = False
            total_return = 0.0

            while not done:
                action = self.select_action(state, evaluate=True)
                next_state, reward, term, trunc, _ = eval_env.step(action)

                done = term or trunc
                state = next_state
                total_return += reward

            returns.append(total_return)

        return float(np.mean(returns)), float(np.std(returns))


def train_one_seed(args, seed: int) -> pd.DataFrame:
    env = gym.make(args.env_name)
    eval_env = gym.make(args.env_name)

    set_seed(env, seed)
    set_seed(eval_env, seed + 10_000)

    agent = SACAgent(
        env=env,
        hidden_size=args.hidden_size,
        replay_size=args.replay_size,
        batch_size=args.batch_size,
        gamma=args.gamma,
        tau=args.tau,
        alpha=args.alpha,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
    )

    state, _ = env.reset()
    results: List[Dict] = []

    for step in range(1, args.total_steps + 1):
        if step <= args.start_steps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state, evaluate=False)

        next_state, reward, term, trunc, _ = env.step(action)

        # For time-limit truncation, do not mark as true terminal.
        done_for_buffer = float(term)

        agent.replay_buffer.add(
            state=state,
            action=action,
            reward=float(reward),
            next_state=next_state,
            done=done_for_buffer,
        )

        done = term or trunc
        state = next_state

        if done:
            state, _ = env.reset()

        if agent.replay_buffer.size >= args.batch_size and step > args.start_steps:
            for _ in range(args.updates_per_step):
                policy_loss, q1_loss, q2_loss = agent.update()

        if step % args.eval_interval == 0:
            mean_return, std_return = agent.evaluate(
                eval_env,
                eval_episodes=args.eval_episodes,
            )

            results.append(
                {
                    "env_name": args.env_name,
                    "algorithm": "sac",
                    "seed": seed,
                    "step": step,
                    "mean_return": mean_return,
                    "std_return": std_return,
                }
            )

            print(
                f"[Eval] env={args.env_name} "
                f"algo=sac "
                f"seed={seed} "
                f"step={step:7d} "
                f"return={mean_return:8.2f} ± {std_return:7.2f}"
            )

    env.close()
    eval_env.close()

    return pd.DataFrame(results)


def run_all(args) -> None:
    os.makedirs(args.result_dir, exist_ok=True)

    all_results = []

    for seed in args.seeds:
        print("=" * 80)
        print(f"Running SAC on {args.env_name}, seed={seed}")
        print("=" * 80)

        df = train_one_seed(args, seed)

        csv_path = os.path.join(
            args.result_dir,
            f"{args.env_name}_sac_seed_{seed}.csv",
        )

        df.to_csv(csv_path, index=False)
        all_results.append(df)

        print(f"Saved: {csv_path}")

    merged = pd.concat(all_results, ignore_index=True)

    merged_path = os.path.join(
        args.result_dir,
        f"{args.env_name}_sac_all_results.csv",
    )

    merged.to_csv(merged_path, index=False)

    print(f"Saved merged results: {merged_path}")


def load_curves(
    result_dir: str,
    env_name: str,
    algorithm: str,
    seeds: List[int],
) -> Tuple[np.ndarray, np.ndarray]:
    curves = []
    steps = None

    for seed in seeds:
        path = os.path.join(
            result_dir,
            f"{env_name}_{algorithm}_seed_{seed}.csv",
        )

        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")

        df = pd.read_csv(path)
        current_steps = df["step"].to_numpy()

        if steps is None:
            steps = current_steps
        elif not np.array_equal(steps, current_steps):
            raise ValueError(f"Step mismatch in {path}")

        curves.append(df["mean_return"].to_numpy())

    return np.array(curves), steps


def plot_rliable(
    score_dict: Dict[str, np.ndarray],
    steps: np.ndarray,
    title: str,
    output_path: str,
    ci_reps: int,
) -> None:
    mean_scores, mean_cis = rly.get_interval_estimates(
        score_dict,
        lambda scores: np.mean(scores, axis=0),
        reps=ci_reps,
    )

    plt.figure(figsize=(7, 4))

    for name in score_dict.keys():
        mean = mean_scores[name]
        lower, upper = mean_cis[name]

        plt.plot(steps, mean, label=name)
        plt.fill_between(steps, lower, upper, alpha=0.2)

    plt.xlabel("Environment steps")
    plt.ylabel("Average return")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()

    print(f"Saved plot: {output_path}")


def plot_results(args) -> None:
    os.makedirs(args.plot_dir, exist_ok=True)

    sac_curves, sac_steps = load_curves(
        result_dir=args.result_dir,
        env_name=args.env_name,
        algorithm="sac",
        seeds=args.seeds,
    )

    sac_plot_path = os.path.join(
        args.plot_dir,
        f"{args.env_name}_sac.png",
    )

    plot_rliable(
        score_dict={"sac": sac_curves},
        steps=sac_steps,
        title=f"SAC on {args.env_name}",
        output_path=sac_plot_path,
        ci_reps=args.ci_reps,
    )

    # Optional: compare with continuous PPO if CSV files exist.
    try:
        ppo_curves, ppo_steps = load_curves(
            result_dir=args.ppo_result_dir,
            env_name=args.env_name,
            algorithm="continuous_ppo",
            seeds=args.seeds,
        )

        if not np.array_equal(sac_steps, ppo_steps):
            print("Skip SAC vs PPO plot because steps do not match.")
            return

        compare_plot_path = os.path.join(
            args.plot_dir,
            f"{args.env_name}_sac_vs_continuous_ppo.png",
        )

        plot_rliable(
            score_dict={
                "sac": sac_curves,
                "continuous_ppo": ppo_curves,
            },
            steps=sac_steps,
            title=f"SAC vs Continuous PPO on {args.env_name}",
            output_path=compare_plot_path,
            ci_reps=args.ci_reps,
        )

    except FileNotFoundError as e:
        print("Skip SAC vs PPO plot.")
        print(e)


def main() -> None:
    args = parse_args()

    if not args.skip_train:
        run_all(args)

    if not args.skip_plot:
        plot_results(args)


if __name__ == "__main__":
    main()
