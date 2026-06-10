from typing import Any, Dict, List, Tuple

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
    parser = argparse.ArgumentParser(description="Run continuous PPO experiments.")

    parser.add_argument("--env-name", type=str, default="Pendulum-v1")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])

    parser.add_argument("--total-steps", type=int, default=100_000)
    parser.add_argument("--eval-interval", type=int, default=5_000)
    parser.add_argument("--eval-episodes", type=int, default=5)

    parser.add_argument("--lr-actor", type=float, default=3e-4)
    parser.add_argument("--lr-critic", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--grad-clip-norm", type=float, default=0.5)

    parser.add_argument("--result-dir", type=str, default="results_l3")
    parser.add_argument("--plot-dir", type=str, default="plots_l3")
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


class ContinuousPolicy(nn.Module):
    """
    Gaussian policy for continuous action spaces.
    It outputs the mean action and learns a global log_std.
    """

    def __init__(
        self,
        state_space: gym.spaces.Box,
        action_space: gym.spaces.Box,
        hidden_size: int = 128,
    ):
        super().__init__()

        self.state_dim = int(np.prod(state_space.shape))
        self.action_dim = int(np.prod(action_space.shape))

        self.action_low = torch.tensor(action_space.low, dtype=torch.float32)
        self.action_high = torch.tensor(action_space.high, dtype=torch.float32)

        self.fc1 = nn.Linear(self.state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.mean = nn.Linear(hidden_size, self.action_dim)

        self.log_std = nn.Parameter(torch.zeros(self.action_dim))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 1:
            x = x.unsqueeze(0)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        mean = self.mean(x)

        log_std = torch.clamp(self.log_std, -20, 2)
        std = torch.exp(log_std).expand_as(mean)

        return mean, std

    def get_dist(self, x: torch.Tensor) -> Normal:
        mean, std = self.forward(x)
        return Normal(mean, std)

    # The method sample_action samples a continuous action from this Gaussian distribution and computes its log probability for the PPO ratio.
    def sample_action(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self.get_dist(x)

        raw_action = dist.rsample()
        log_prob = dist.log_prob(raw_action).sum(dim=-1)

        action_low = self.action_low.to(raw_action.device)
        action_high = self.action_high.to(raw_action.device)

        clipped_action = torch.clamp(raw_action, action_low, action_high)

        return clipped_action, log_prob

    def deterministic_action(self, x: torch.Tensor) -> torch.Tensor:
        mean, _ = self.forward(x)

        action_low = self.action_low.to(mean.device)
        action_high = self.action_high.to(mean.device)

        return torch.clamp(mean, action_low, action_high)


class ValueNetwork(nn.Module):
    def __init__(self, state_space: gym.spaces.Box, hidden_size: int = 128):
        super().__init__()

        self.state_dim = int(np.prod(state_space.shape))

        self.fc1 = nn.Linear(self.state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return self.out(x).squeeze(-1)


class ContinuousPPO:
    def __init__(
        self,
        env: gym.Env,
        lr_actor: float,
        lr_critic: float,
        gamma: float,
        gae_lambda: float,
        clip_eps: float,
        epochs: int,
        batch_size: int,
        ent_coef: float,
        vf_coef: float,
        hidden_size: int,
        grad_clip_norm: float,
    ):
        self.env = env

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.grad_clip_norm = grad_clip_norm

        self.policy = ContinuousPolicy(
            env.observation_space,
            env.action_space,
            hidden_size,
        )

        self.value_fn = ValueNetwork(
            env.observation_space,
            hidden_size,
        )

        self.optimizer = torch.optim.Adam(
            [
                {"params": self.policy.parameters(), "lr": lr_actor},
                {"params": self.value_fn.parameters(), "lr": lr_critic},
            ]
        )

    def predict(self, state: np.ndarray) -> Tuple[np.ndarray, torch.Tensor]:
        state_t = torch.from_numpy(state).float()

        action_t, logp = self.policy.sample_action(state_t)

        action = action_t.squeeze(0).detach().numpy()

        return action, logp.squeeze(0)

    def predict_eval(self, state: np.ndarray) -> np.ndarray:
        state_t = torch.from_numpy(state).float()

        with torch.no_grad():
            action_t = self.policy.deterministic_action(state_t)

        return action_t.squeeze(0).numpy()

    def compute_gae(
        self,
        rewards: List[float],
        values: torch.Tensor,
        next_values: torch.Tensor,
        dones: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rewards_t = torch.tensor(rewards, dtype=torch.float32)

        deltas = rewards_t + self.gamma * next_values * (1.0 - dones) - values

        advantages = torch.zeros_like(rewards_t)
        gae = 0.0

        for t in reversed(range(len(rewards_t))):
            gae = deltas[t] + self.gamma * self.gae_lambda * (1.0 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values

        advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1e-8
        )

        return advantages.detach(), returns.detach()

    def update(self, trajectory: List[Any]) -> Tuple[float, float, float]:
        states = torch.stack([torch.from_numpy(t[0]).float() for t in trajectory])
        actions = torch.tensor(
            np.array([t[1] for t in trajectory]),
            dtype=torch.float32,
        )
        old_logps = torch.stack([t[2] for t in trajectory]).detach()
        rewards = [t[3] for t in trajectory]
        dones = torch.tensor([t[4] for t in trajectory], dtype=torch.float32)
        next_states = torch.stack([torch.from_numpy(t[5]).float() for t in trajectory])

        with torch.no_grad():
            values = self.value_fn(states)
            next_values = self.value_fn(next_states)

        advantages, returns = self.compute_gae(
            rewards,
            values,
            next_values,
            dones,
        )

        dataset = torch.utils.data.TensorDataset(
            states,
            actions,
            old_logps,
            advantages,
            returns,
        )

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        last_policy_loss = 0.0
        last_value_loss = 0.0
        last_entropy_loss = 0.0

        for _ in range(self.epochs):
            for b_states, b_actions, b_oldlogp, b_adv, b_ret in loader:
                dist = self.policy.get_dist(b_states)

                new_logp = dist.log_prob(b_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()

                ratio = torch.exp(new_logp - b_oldlogp)

                unclipped = ratio * b_adv
                clipped = (
                    torch.clamp(
                        ratio,
                        1.0 - self.clip_eps,
                        1.0 + self.clip_eps,
                    )
                    * b_adv
                )

                policy_loss = -torch.min(unclipped, clipped).mean()

                values_pred = self.value_fn(b_states)
                value_loss = F.mse_loss(values_pred, b_ret)

                entropy_loss = -entropy

                loss = (
                    policy_loss
                    + self.vf_coef * value_loss
                    + self.ent_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.value_fn.parameters()),
                    max_norm=self.grad_clip_norm,
                )

                self.optimizer.step()

                last_policy_loss = float(policy_loss.item())
                last_value_loss = float(value_loss.item())
                last_entropy_loss = float(entropy_loss.item())

        return last_policy_loss, last_value_loss, last_entropy_loss

    def evaluate(self, eval_env: gym.Env, eval_episodes: int) -> Tuple[float, float]:
        returns = []

        for _ in range(eval_episodes):
            state, _ = eval_env.reset()
            done = False
            total_return = 0.0

            while not done:
                action = self.predict_eval(state)
                state, reward, term, trunc, _ = eval_env.step(action)
                done = term or trunc
                total_return += reward

            returns.append(total_return)

        return float(np.mean(returns)), float(np.std(returns))


def train_one_seed(args, seed: int) -> pd.DataFrame:
    env = gym.make(args.env_name)
    eval_env = gym.make(args.env_name)

    set_seed(env, seed)
    set_seed(eval_env, seed + 10_000)

    agent = ContinuousPPO(
        env=env,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_eps=args.clip_eps,
        epochs=args.epochs,
        batch_size=args.batch_size,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        hidden_size=args.hidden_size,
        grad_clip_norm=args.grad_clip_norm,
    )

    step_count = 0
    results: List[Dict[str, Any]] = []

    while step_count < args.total_steps:
        state, _ = env.reset()
        done = False
        trajectory = []

        while not done and step_count < args.total_steps:
            action, logp = agent.predict(state)

            next_state, reward, term, trunc, _ = env.step(action)
            done = term or trunc

            trajectory.append(
                (
                    state,
                    action,
                    logp,
                    float(reward),
                    float(done),
                    next_state,
                )
            )

            state = next_state
            step_count += 1

            if step_count % args.eval_interval == 0:
                mean_return, std_return = agent.evaluate(
                    eval_env,
                    args.eval_episodes,
                )

                results.append(
                    {
                        "env_name": args.env_name,
                        "algorithm": "continuous_ppo",
                        "seed": seed,
                        "step": step_count,
                        "mean_return": mean_return,
                        "std_return": std_return,
                    }
                )

                print(
                    f"[Eval] env={args.env_name} "
                    f"algo=continuous_ppo "
                    f"seed={seed} "
                    f"step={step_count:7d} "
                    f"return={mean_return:8.2f} ± {std_return:7.2f}"
                )

        policy_loss, value_loss, entropy_loss = agent.update(trajectory)

        print(
            f"[Train] seed={seed} "
            f"step={step_count:7d} "
            f"policy_loss={policy_loss:8.4f} "
            f"value_loss={value_loss:8.4f} "
            f"entropy_loss={entropy_loss:8.4f}"
        )

    env.close()
    eval_env.close()

    return pd.DataFrame(results)


def run_all(args) -> None:
    os.makedirs(args.result_dir, exist_ok=True)

    all_results = []

    for seed in args.seeds:
        print("=" * 80)
        print(f"Running continuous PPO on {args.env_name}, seed={seed}")
        print("=" * 80)

        df = train_one_seed(args, seed)

        csv_path = os.path.join(
            args.result_dir,
            f"{args.env_name}_continuous_ppo_seed_{seed}.csv",
        )

        df.to_csv(csv_path, index=False)
        all_results.append(df)

        print(f"Saved: {csv_path}")

    merged = pd.concat(all_results, ignore_index=True)

    merged_path = os.path.join(
        args.result_dir,
        f"{args.env_name}_continuous_ppo_all_results.csv",
    )

    merged.to_csv(merged_path, index=False)

    print(f"Saved merged results: {merged_path}")


def plot_results(args) -> None:
    os.makedirs(args.plot_dir, exist_ok=True)

    score_dict = {}
    seed_curves = []
    steps = None

    for seed in args.seeds:
        path = os.path.join(
            args.result_dir,
            f"{args.env_name}_continuous_ppo_seed_{seed}.csv",
        )

        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing result file: {path}")

        df = pd.read_csv(path)

        current_steps = df["step"].to_numpy()

        if steps is None:
            steps = current_steps
        elif not np.array_equal(steps, current_steps):
            raise ValueError(f"Step mismatch in {path}")

        seed_curves.append(df["mean_return"].to_numpy())

    score_dict["continuous_ppo"] = np.array(seed_curves)

    mean_scores, mean_cis = rly.get_interval_estimates(
        score_dict,
        lambda scores: np.mean(scores, axis=0),
        reps=args.ci_reps,
    )

    plt.figure(figsize=(7, 4))

    mean = mean_scores["continuous_ppo"]
    lower, upper = mean_cis["continuous_ppo"]

    plt.plot(steps, mean, label="continuous_ppo")
    plt.fill_between(steps, lower, upper, alpha=0.2)

    plt.xlabel("Environment steps")
    plt.ylabel("Average return")
    plt.title(f"Continuous PPO on {args.env_name}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plot_path = os.path.join(
        args.plot_dir,
        f"{args.env_name}_continuous_ppo.png",
    )

    plt.savefig(plot_path, dpi=300)
    plt.show()

    print(f"Saved plot: {plot_path}")


def main() -> None:
    args = parse_args()

    if not args.skip_train:
        run_all(args)

    if not args.skip_plot:
        plot_results(args)


if __name__ == "__main__":
    main()
