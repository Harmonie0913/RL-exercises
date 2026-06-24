from typing import Dict, List

import math
import os
import random

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# ============================================================
# Paths
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(BASE_DIR, "results", "level1")
FIGURES_DIR = os.path.join(BASE_DIR, "figures", "level1")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ============================================================
# Experiment config
# ============================================================

ENV_ID = "CartPole-v1"

# Important:
# We use fixed environment steps, not fixed episodes.
# This makes all seeds aligned on the x-axis.
TOTAL_STEPS = 20_000
EVAL_EVERY_STEPS = 1_000
EVAL_EPISODES = 20

GAMMA = 0.99
LR = 1e-3

BATCH_SIZE = 64
BUFFER_SIZE = 50_000
MIN_BUFFER_SIZE = 1_000
TARGET_UPDATE_INTERVAL = 500

EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_STEPS = 8_000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# L1: low / medium / large seed amounts
LOW_SEED_SETS = [
    [0, 1, 2],
    [10, 11, 12],
    [100, 101, 102],
]

MEDIUM_SEEDS = list(range(10))
LARGE_SEEDS = list(range(30))

# If True, rerun training even if csv already exists.
# If False, reuse old csv when available.
FORCE_RERUN = True


# ============================================================
# Utilities
# ============================================================


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_env(seed: int):
    env = gym.make(ENV_ID)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def iqm(values: np.ndarray) -> float:
    """
    Interquartile mean:
    sort values, remove lowest 25% and highest 25%,
    then average the middle 50%.
    """
    values = np.sort(values)
    n = len(values)

    if n == 0:
        return np.nan

    lower = int(math.floor(0.25 * n))
    upper = int(math.ceil(0.75 * n))

    trimmed = values[lower:upper]

    if len(trimmed) == 0:
        return float(np.mean(values))

    return float(np.mean(trimmed))


def aggregate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Correct aggregation:
    all runs are evaluated at the same fixed environment steps:
    1000, 2000, ..., TOTAL_STEPS.

    Therefore grouping by step is valid here.
    """
    rows = []

    for step, group in df.groupby("step"):
        rewards = group["eval_return"].to_numpy()
        n = len(rewards)

        mean = float(np.mean(rewards))
        median = float(np.median(rewards))
        iqm_value = iqm(rewards)

        if n > 1:
            std = float(np.std(rewards, ddof=1))
            stderr = float(std / np.sqrt(n))
            ci95 = float(1.96 * stderr)
        else:
            std = 0.0
            stderr = 0.0
            ci95 = 0.0

        rows.append(
            {
                "step": int(step),
                "n_seeds": int(n),
                "mean": mean,
                "median": median,
                "iqm": iqm_value,
                "std": std,
                "stderr": stderr,
                "ci95_low": mean - ci95,
                "ci95_high": mean + ci95,
                "ci95_width": 2.0 * ci95,
            }
        )

    return pd.DataFrame(rows).sort_values("step")


def smooth(values: np.ndarray, window: int = 3) -> np.ndarray:
    values = np.asarray(values)

    if len(values) < window:
        return values

    return (
        pd.Series(values)
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )


# ============================================================
# DQN components
# ============================================================


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, obs, action, reward, next_obs, done):
        transition = (obs, action, reward, next_obs, done)

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)

        obs, actions, rewards, next_obs, dones = zip(*batch)

        obs = torch.tensor(np.array(obs), dtype=torch.float32, device=DEVICE)
        actions = torch.tensor(actions, dtype=torch.long, device=DEVICE)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)
        next_obs = torch.tensor(np.array(next_obs), dtype=torch.float32, device=DEVICE)
        dones = torch.tensor(dones, dtype=torch.float32, device=DEVICE)

        return obs, actions, rewards, next_obs, dones

    def __len__(self):
        return len(self.buffer)


# ============================================================
# DQN training and evaluation
# ============================================================


def evaluate_dqn(q_net: QNetwork, seed: int) -> float:
    """
    Evaluate greedy policy for EVAL_EPISODES episodes.
    """
    eval_env = make_env(seed + 10_000)
    returns = []

    for _ in range(EVAL_EPISODES):
        obs, _ = eval_env.reset()
        done = False
        episode_return = 0.0

        while not done:
            with torch.no_grad():
                obs_t = torch.tensor(
                    obs,
                    dtype=torch.float32,
                    device=DEVICE,
                ).unsqueeze(0)

                action = q_net(obs_t).argmax(dim=1).item()

            obs, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            episode_return += reward

        returns.append(episode_return)

    eval_env.close()

    return float(np.mean(returns))


def train_dqn(seed: int) -> pd.DataFrame:
    """
    Train DQN for a fixed number of environment steps.
    Evaluation is also done at fixed environment steps.

    This guarantees that all seeds are aligned:
    step = 1000, 2000, ..., TOTAL_STEPS
    """
    set_seed(seed)

    env = make_env(seed)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    q_net = QNetwork(obs_dim, action_dim).to(DEVICE)
    target_net = QNetwork(obs_dim, action_dim).to(DEVICE)
    target_net.load_state_dict(q_net.state_dict())

    optimizer = optim.Adam(q_net.parameters(), lr=LR)
    replay_buffer = ReplayBuffer(BUFFER_SIZE)

    global_step = 0
    episode = 0
    eval_idx = 0
    next_eval_step = EVAL_EVERY_STEPS

    logs = []

    obs, _ = env.reset()

    while global_step < TOTAL_STEPS:
        epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * max(
            0.0,
            1.0 - global_step / EPSILON_DECAY_STEPS,
        )

        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                obs_t = torch.tensor(
                    obs,
                    dtype=torch.float32,
                    device=DEVICE,
                ).unsqueeze(0)

                action = q_net(obs_t).argmax(dim=1).item()

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        replay_buffer.push(obs, action, reward, next_obs, done)

        obs = next_obs
        global_step += 1

        if done:
            episode += 1
            obs, _ = env.reset()

        if len(replay_buffer) >= MIN_BUFFER_SIZE:
            batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones = (
                replay_buffer.sample(BATCH_SIZE)
            )

            q_values = q_net(batch_obs)
            q_values = q_values.gather(
                1,
                batch_actions.unsqueeze(1),
            ).squeeze(1)

            with torch.no_grad():
                next_q_values = target_net(batch_next_obs).max(dim=1)[0]
                targets = batch_rewards + GAMMA * (1.0 - batch_dones) * next_q_values

            loss = nn.functional.mse_loss(q_values, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if global_step % TARGET_UPDATE_INTERVAL == 0:
            target_net.load_state_dict(q_net.state_dict())

        if global_step >= next_eval_step:
            eval_idx += 1

            eval_return = evaluate_dqn(q_net, seed)

            logs.append(
                {
                    "algorithm": "DQN",
                    "seed": seed,
                    "eval_idx": eval_idx,
                    "step": next_eval_step,
                    "episode": episode,
                    "eval_return": eval_return,
                }
            )

            print(
                f"seed={seed:3d} | "
                f"eval_idx={eval_idx:2d} | "
                f"step={next_eval_step:6d} | "
                f"episode={episode:4d} | "
                f"eval_return={eval_return:8.2f}"
            )

            next_eval_step += EVAL_EVERY_STEPS

    env.close()

    return pd.DataFrame(logs)


# ============================================================
# Plotting
# ============================================================


def plot_mean_ci(metrics_dict: Dict[str, pd.DataFrame], save_path: str, title: str):
    plt.figure(figsize=(9, 5))

    for label, metrics in metrics_dict.items():
        x = metrics["step"].to_numpy()
        mean = metrics["mean"].to_numpy()
        ci_low = metrics["ci95_low"].to_numpy()
        ci_high = metrics["ci95_high"].to_numpy()

        plt.plot(x, mean, label=label)
        plt.fill_between(x, ci_low, ci_high, alpha=0.15)

    plt.xlabel("Environment steps")
    plt.ylabel("Evaluation return")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_smoothed_mean_ci(
    metrics_dict: Dict[str, pd.DataFrame], save_path: str, title: str
):
    plt.figure(figsize=(9, 5))

    for label, metrics in metrics_dict.items():
        x = metrics["step"].to_numpy()
        mean = smooth(metrics["mean"].to_numpy(), window=3)
        ci_low = smooth(metrics["ci95_low"].to_numpy(), window=3)
        ci_high = smooth(metrics["ci95_high"].to_numpy(), window=3)

        plt.plot(x, mean, label=label)
        plt.fill_between(x, ci_low, ci_high, alpha=0.15)

    plt.xlabel("Environment steps")
    plt.ylabel("Evaluation return")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_metric(
    metrics_dict: Dict[str, pd.DataFrame],
    metric: str,
    save_path: str,
    title: str,
    ylabel: str,
):
    plt.figure(figsize=(9, 5))

    for label, metrics in metrics_dict.items():
        x = metrics["step"].to_numpy()
        y = metrics[metric].to_numpy()

        plt.plot(x, y, label=label)

    plt.xlabel("Environment steps")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_mean_median_iqm(metrics: pd.DataFrame, label: str, save_path: str):
    plt.figure(figsize=(9, 5))

    x = metrics["step"].to_numpy()

    plt.plot(x, metrics["mean"].to_numpy(), label="mean")
    plt.plot(x, metrics["median"].to_numpy(), label="median")
    plt.plot(x, metrics["iqm"].to_numpy(), label="IQM")

    plt.xlabel("Environment steps")
    plt.ylabel("Evaluation return")
    plt.title(f"{label}: mean vs median vs IQM")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_raw_seed_curves(
    df: pd.DataFrame, seeds: List[int], save_path: str, title: str
):
    plt.figure(figsize=(9, 5))

    for seed in seeds:
        seed_df = df[df["seed"] == seed].sort_values("step")
        plt.plot(
            seed_df["step"].to_numpy(),
            seed_df["eval_return"].to_numpy(),
            alpha=0.5,
            label=f"seed {seed}",
        )

    plt.xlabel("Environment steps")
    plt.ylabel("Evaluation return")
    plt.title(title)
    plt.legend(ncol=2, fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


# ============================================================
# Main
# ============================================================


def main():
    print("=" * 80)
    print("Week 8 Level 1: Seeding experiment")
    print(f"Environment: {ENV_ID}")
    print("Algorithm: DQN")
    print(f"Device: {DEVICE}")
    print(f"Total steps per seed: {TOTAL_STEPS}")
    print(f"Evaluation every steps: {EVAL_EVERY_STEPS}")
    print(f"Evaluation episodes: {EVAL_EPISODES}")
    print("=" * 80)

    raw_csv_path = os.path.join(RESULTS_DIR, "dqn_all_seed_runs_fixed_steps.csv")

    if os.path.exists(raw_csv_path) and not FORCE_RERUN:
        print(f"Loading existing results from: {raw_csv_path}")
        all_runs_df = pd.read_csv(raw_csv_path)
    else:
        all_needed_seeds = sorted(
            set(seed for seed_set in LOW_SEED_SETS for seed in seed_set)
            .union(MEDIUM_SEEDS)
            .union(LARGE_SEEDS)
        )

        print(f"Total unique seeds to run: {len(all_needed_seeds)}")
        print(f"Seeds: {all_needed_seeds}")

        all_runs = []

        for seed in all_needed_seeds:
            print("\n" + "-" * 80)
            print(f"Running DQN with seed {seed}")
            print("-" * 80)

            run_df = train_dqn(seed)
            all_runs.append(run_df)

        all_runs_df = pd.concat(all_runs, ignore_index=True)
        all_runs_df.to_csv(raw_csv_path, index=False)

        print(f"\nSaved raw runs to: {raw_csv_path}")

    # ------------------------------------------------------------
    # Low seed sets
    # ------------------------------------------------------------

    low_metrics_dict = {}

    for idx, seed_set in enumerate(LOW_SEED_SETS):
        subset = all_runs_df[all_runs_df["seed"].isin(seed_set)]
        metrics = aggregate_metrics(subset)

        label = f"low set {idx}: {seed_set}"
        low_metrics_dict[label] = metrics

        metrics_path = os.path.join(
            RESULTS_DIR,
            f"l1_low_set_{idx}_metrics_fixed_steps.csv",
        )
        metrics.to_csv(metrics_path, index=False)

        print(f"Saved low seed set {idx} metrics to: {metrics_path}")

    # ------------------------------------------------------------
    # Low / medium / large seed amounts
    # ------------------------------------------------------------

    low_main_seeds = LOW_SEED_SETS[0]

    low_df = all_runs_df[all_runs_df["seed"].isin(low_main_seeds)]
    medium_df = all_runs_df[all_runs_df["seed"].isin(MEDIUM_SEEDS)]
    large_df = all_runs_df[all_runs_df["seed"].isin(LARGE_SEEDS)]

    low_metrics = aggregate_metrics(low_df)
    medium_metrics = aggregate_metrics(medium_df)
    large_metrics = aggregate_metrics(large_df)

    low_metrics.to_csv(
        os.path.join(RESULTS_DIR, "l1_low_metrics_fixed_steps.csv"),
        index=False,
    )
    medium_metrics.to_csv(
        os.path.join(RESULTS_DIR, "l1_medium_metrics_fixed_steps.csv"),
        index=False,
    )
    large_metrics.to_csv(
        os.path.join(RESULTS_DIR, "l1_large_metrics_fixed_steps.csv"),
        index=False,
    )

    seed_amount_metrics = {
        f"low n={len(low_main_seeds)}": low_metrics,
        f"medium n={len(MEDIUM_SEEDS)}": medium_metrics,
        f"large n={len(LARGE_SEEDS)}": large_metrics,
    }

    # ------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------

    plot_mean_ci(
        seed_amount_metrics,
        os.path.join(FIGURES_DIR, "l1_01_seed_amount_mean_ci_fixed_steps.png"),
        "DQN on CartPole-v1: mean return with 95% CI",
    )

    plot_smoothed_mean_ci(
        seed_amount_metrics,
        os.path.join(FIGURES_DIR, "l1_02_seed_amount_mean_ci_smoothed_fixed_steps.png"),
        "DQN on CartPole-v1: smoothed mean return with 95% CI",
    )

    plot_mean_ci(
        low_metrics_dict,
        os.path.join(FIGURES_DIR, "l1_03_low_seed_sets_mean_ci_fixed_steps.png"),
        "DQN on CartPole-v1: different low seed sets",
    )

    plot_smoothed_mean_ci(
        low_metrics_dict,
        os.path.join(
            FIGURES_DIR, "l1_04_low_seed_sets_mean_ci_smoothed_fixed_steps.png"
        ),
        "DQN on CartPole-v1: different low seed sets, smoothed",
    )

    plot_metric(
        seed_amount_metrics,
        "std",
        os.path.join(FIGURES_DIR, "l1_05_std_comparison_fixed_steps.png"),
        "Standard deviation over different seed amounts",
        "Standard deviation",
    )

    plot_metric(
        seed_amount_metrics,
        "stderr",
        os.path.join(FIGURES_DIR, "l1_06_stderr_comparison_fixed_steps.png"),
        "Standard error over different seed amounts",
        "Standard error",
    )

    plot_metric(
        seed_amount_metrics,
        "ci95_width",
        os.path.join(FIGURES_DIR, "l1_07_ci_width_comparison_fixed_steps.png"),
        "95% CI width over different seed amounts",
        "95% CI width",
    )

    plot_mean_median_iqm(
        low_metrics,
        "low n=3",
        os.path.join(FIGURES_DIR, "l1_08_low_mean_median_iqm_fixed_steps.png"),
    )

    plot_mean_median_iqm(
        medium_metrics,
        "medium n=10",
        os.path.join(FIGURES_DIR, "l1_09_medium_mean_median_iqm_fixed_steps.png"),
    )

    plot_mean_median_iqm(
        large_metrics,
        "large n=30",
        os.path.join(FIGURES_DIR, "l1_10_large_mean_median_iqm_fixed_steps.png"),
    )

    plot_raw_seed_curves(
        all_runs_df,
        LOW_SEED_SETS[0],
        os.path.join(FIGURES_DIR, "l1_11_raw_low_set_0_seed_curves.png"),
        "Raw curves: low seed set 0",
    )

    plot_raw_seed_curves(
        all_runs_df,
        MEDIUM_SEEDS,
        os.path.join(FIGURES_DIR, "l1_12_raw_medium_seed_curves.png"),
        "Raw curves: medium seed set",
    )

    print("\n" + "=" * 80)
    print("Level 1 finished.")
    print(f"Raw results saved to: {raw_csv_path}")
    print(f"Aggregated results saved in: {RESULTS_DIR}")
    print(f"Figures saved in: {FIGURES_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
