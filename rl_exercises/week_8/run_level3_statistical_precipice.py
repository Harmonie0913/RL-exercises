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
from torch.distributions import Categorical

# ============================================================
# Paths
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(BASE_DIR, "results", "level3")
FIGURES_DIR = os.path.join(BASE_DIR, "figures", "level3")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ============================================================
# Config
# ============================================================

ENV_IDS = [
    "CartPole-v1",
    "Acrobot-v1",
    "MountainCar-v0",
]

ALGORITHMS = ["DQN", "REINFORCE"]


SEEDS = list(range(5))

TOTAL_STEPS = 20_000
EVAL_EVERY_STEPS = 1_000
EVAL_EPISODES = 10

GAMMA = 0.99

ALPHA = 0.05
N_PERMUTATIONS = 10_000
N_BOOTSTRAPS = 2_000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FORCE_RERUN = True


# Normalization ranges.
# Higher normalized_return is always better.
RETURN_RANGES = {
    "CartPole-v1": {
        "min": 0.0,
        "max": 500.0,
    },
    "Acrobot-v1": {
        "min": -500.0,
        "max": 0.0,
    },
    "MountainCar-v0": {
        "min": -200.0,
        "max": 0.0,
    },
}


# ============================================================
# Utilities
# ============================================================


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_env(env_id: str, seed: int):
    env = gym.make(env_id)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def normalize_return(env_id: str, value: float) -> float:
    low = RETURN_RANGES[env_id]["min"]
    high = RETURN_RANGES[env_id]["max"]

    norm = (value - low) / (high - low)
    return float(np.clip(norm, 0.0, 1.0))


def iqm(values: np.ndarray) -> float:
    values = np.sort(np.asarray(values))
    n = len(values)

    if n == 0:
        return np.nan

    lower = int(math.floor(0.25 * n))
    upper = int(math.ceil(0.75 * n))

    trimmed = values[lower:upper]

    if len(trimmed) == 0:
        return float(np.mean(values))

    return float(np.mean(trimmed))


def bootstrap_ci(values, stat_fn, n_bootstraps=2_000, seed=0):
    rng = np.random.default_rng(seed)
    values = np.asarray(values)

    if len(values) == 0:
        return np.nan, np.nan

    samples = []

    for _ in range(n_bootstraps):
        sample = rng.choice(values, size=len(values), replace=True)
        samples.append(stat_fn(sample))

    low, high = np.percentile(samples, [2.5, 97.5])
    return float(low), float(high)


def integrate_curve(y, x):

    return np.trapezoid(y, x)


def paired_permutation_test(x, y, n_permutations=10_000, seed=0):
    rng = np.random.default_rng(seed)

    x = np.asarray(x)
    y = np.asarray(y)

    diffs = x - y
    observed = float(np.mean(diffs))

    count = 0

    for _ in range(n_permutations):
        signs = rng.choice([-1, 1], size=len(diffs))
        stat = np.mean(diffs * signs)

        if abs(stat) >= abs(observed):
            count += 1

    p_value = (count + 1) / (n_permutations + 1)

    return observed, p_value


# ============================================================
# Evaluation
# ============================================================


def evaluate_policy(env_id: str, policy_fn, seed: int) -> float:
    env = make_env(env_id, seed + 10_000)
    returns = []

    for _ in range(EVAL_EPISODES):
        obs, _ = env.reset()
        done = False
        episode_return = 0.0

        while not done:
            action = policy_fn(obs)

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_return += reward

        returns.append(episode_return)

    env.close()
    return float(np.mean(returns))


# ============================================================
# DQN
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


def train_dqn(env_id: str, seed: int) -> pd.DataFrame:
    set_seed(seed)

    env = make_env(env_id, seed)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    q_net = QNetwork(obs_dim, action_dim).to(DEVICE)
    target_net = QNetwork(obs_dim, action_dim).to(DEVICE)
    target_net.load_state_dict(q_net.state_dict())

    optimizer = optim.Adam(q_net.parameters(), lr=1e-3)
    replay_buffer = ReplayBuffer(capacity=50_000)

    batch_size = 64
    min_buffer_size = 1_000
    target_update_interval = 500

    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay_steps = 8_000

    global_step = 0
    eval_idx = 0
    next_eval_step = EVAL_EVERY_STEPS

    obs, _ = env.reset()
    logs = []

    while global_step < TOTAL_STEPS:
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * max(
            0.0,
            1.0 - global_step / epsilon_decay_steps,
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
            obs, _ = env.reset()

        if len(replay_buffer) >= min_buffer_size:
            batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones = (
                replay_buffer.sample(batch_size)
            )

            q_values = q_net(batch_obs)
            q_values = q_values.gather(1, batch_actions.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                next_q_values = target_net(batch_next_obs).max(dim=1)[0]
                targets = batch_rewards + GAMMA * (1.0 - batch_dones) * next_q_values

            loss = nn.functional.mse_loss(q_values, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if global_step % target_update_interval == 0:
            target_net.load_state_dict(q_net.state_dict())

        if global_step >= next_eval_step:
            eval_idx += 1

            def policy_fn(obs_eval):
                with torch.no_grad():
                    obs_t = torch.tensor(
                        obs_eval,
                        dtype=torch.float32,
                        device=DEVICE,
                    ).unsqueeze(0)

                    return q_net(obs_t).argmax(dim=1).item()

            eval_return = evaluate_policy(env_id, policy_fn, seed)

            logs.append(
                {
                    "env": env_id,
                    "algorithm": "DQN",
                    "seed": seed,
                    "eval_idx": eval_idx,
                    "step": next_eval_step,
                    "eval_return": eval_return,
                    "normalized_return": normalize_return(env_id, eval_return),
                }
            )

            print(
                f"DQN       | env={env_id:15s} | "
                f"seed={seed:2d} | step={next_eval_step:6d} | "
                f"eval_return={eval_return:8.2f}"
            )

            next_eval_step += EVAL_EVERY_STEPS

    env.close()
    return pd.DataFrame(logs)


# ============================================================
# REINFORCE
# ============================================================


class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.net(x)


def train_reinforce(env_id: str, seed: int) -> pd.DataFrame:
    set_seed(seed)

    env = make_env(env_id, seed)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy = PolicyNetwork(obs_dim, action_dim).to(DEVICE)
    optimizer = optim.Adam(policy.parameters(), lr=1e-2)

    global_step = 0
    eval_idx = 0
    next_eval_step = EVAL_EVERY_STEPS

    logs = []

    while global_step < TOTAL_STEPS:
        obs, _ = env.reset()

        log_probs = []
        rewards = []
        done = False

        while not done and global_step < TOTAL_STEPS:
            obs_t = torch.tensor(
                obs,
                dtype=torch.float32,
                device=DEVICE,
            ).unsqueeze(0)

            logits = policy(obs_t)
            dist = Categorical(logits=logits)

            action = dist.sample()
            log_prob = dist.log_prob(action)

            next_obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            log_probs.append(log_prob)
            rewards.append(reward)

            obs = next_obs
            global_step += 1

            if global_step >= next_eval_step:
                eval_idx += 1

                def policy_fn(obs_eval):
                    with torch.no_grad():
                        obs_eval_t = torch.tensor(
                            obs_eval,
                            dtype=torch.float32,
                            device=DEVICE,
                        ).unsqueeze(0)

                        logits_eval = policy(obs_eval_t)
                        return logits_eval.argmax(dim=1).item()

                eval_return = evaluate_policy(env_id, policy_fn, seed)

                logs.append(
                    {
                        "env": env_id,
                        "algorithm": "REINFORCE",
                        "seed": seed,
                        "eval_idx": eval_idx,
                        "step": next_eval_step,
                        "eval_return": eval_return,
                        "normalized_return": normalize_return(env_id, eval_return),
                    }
                )

                print(
                    f"REINFORCE | env={env_id:15s} | "
                    f"seed={seed:2d} | step={next_eval_step:6d} | "
                    f"eval_return={eval_return:8.2f}"
                )

                next_eval_step += EVAL_EVERY_STEPS

        if len(rewards) == 0:
            continue

        returns = []
        G = 0.0

        for r in reversed(rewards):
            G = r + GAMMA * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32, device=DEVICE)

        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = 0.0

        for log_prob, G in zip(log_probs, returns):
            loss = loss - log_prob * G

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    env.close()
    return pd.DataFrame(logs)


# ============================================================
# Analysis
# ============================================================


def compute_auc(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (env_id, algo, seed), group in df.groupby(["env", "algorithm", "seed"]):
        group = group.sort_values("step")

        steps = group["step"].to_numpy()
        raw_returns = group["eval_return"].to_numpy()
        norm_returns = group["normalized_return"].to_numpy()

        raw_auc = integrate_curve(raw_returns, steps)
        norm_auc = integrate_curve(norm_returns, steps) / (steps[-1] - steps[0])

        rows.append(
            {
                "env": env_id,
                "algorithm": algo,
                "seed": seed,
                "raw_auc": float(raw_auc),
                "normalized_auc": float(norm_auc),
                "final_return": float(raw_returns[-1]),
                "final_normalized_return": float(norm_returns[-1]),
            }
        )

    return pd.DataFrame(rows)


def compute_aggregate_metrics(auc_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for algo, group in auc_df.groupby("algorithm"):
        values = group["normalized_auc"].to_numpy()
        n = len(values)

        mean = float(np.mean(values))
        median = float(np.median(values))
        iqm_value = iqm(values)

        std = float(np.std(values, ddof=1)) if n > 1 else 0.0
        stderr = float(std / np.sqrt(n)) if n > 1 else 0.0
        ci95 = float(1.96 * stderr) if n > 1 else 0.0

        mean_ci_low, mean_ci_high = bootstrap_ci(
            values,
            np.mean,
            n_bootstraps=N_BOOTSTRAPS,
            seed=0,
        )

        median_ci_low, median_ci_high = bootstrap_ci(
            values,
            np.median,
            n_bootstraps=N_BOOTSTRAPS,
            seed=1,
        )

        iqm_ci_low, iqm_ci_high = bootstrap_ci(
            values,
            iqm,
            n_bootstraps=N_BOOTSTRAPS,
            seed=2,
        )

        rows.append(
            {
                "algorithm": algo,
                "n": n,
                "mean": mean,
                "median": median,
                "iqm": iqm_value,
                "std": std,
                "stderr": stderr,
                "ci95_low_normal_approx": mean - ci95,
                "ci95_high_normal_approx": mean + ci95,
                "mean_bootstrap_ci_low": mean_ci_low,
                "mean_bootstrap_ci_high": mean_ci_high,
                "median_bootstrap_ci_low": median_ci_low,
                "median_bootstrap_ci_high": median_ci_high,
                "iqm_bootstrap_ci_low": iqm_ci_low,
                "iqm_bootstrap_ci_high": iqm_ci_high,
            }
        )

    return pd.DataFrame(rows)


def compute_env_tests(auc_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for env_id in ENV_IDS:
        env_df = auc_df[auc_df["env"] == env_id]

        pivot = env_df.pivot(
            index="seed",
            columns="algorithm",
            values="normalized_auc",
        ).dropna()

        dqn = pivot["DQN"].to_numpy()
        reinforce = pivot["REINFORCE"].to_numpy()

        diff, p_value = paired_permutation_test(
            dqn,
            reinforce,
            n_permutations=N_PERMUTATIONS,
            seed=0,
        )

        rows.append(
            {
                "env": env_id,
                "test": "paired permutation test",
                "aggregation": "normalized AUC per seed",
                "alpha": ALPHA,
                "n_seeds": len(pivot),
                "mean_auc_dqn": float(np.mean(dqn)),
                "mean_auc_reinforce": float(np.mean(reinforce)),
                "mean_difference_dqn_minus_reinforce": float(diff),
                "p_value": float(p_value),
                "significant": bool(p_value < ALPHA),
            }
        )

    return pd.DataFrame(rows)


def compute_overall_test(auc_df: pd.DataFrame) -> dict:
    pivot = auc_df.pivot_table(
        index=["env", "seed"],
        columns="algorithm",
        values="normalized_auc",
    ).dropna()

    dqn = pivot["DQN"].to_numpy()
    reinforce = pivot["REINFORCE"].to_numpy()

    diff, p_value = paired_permutation_test(
        dqn,
        reinforce,
        n_permutations=N_PERMUTATIONS,
        seed=0,
    )

    return {
        "test": "paired permutation test",
        "aggregation": "normalized AUC over env-seed pairs",
        "alpha": ALPHA,
        "n_pairs": len(pivot),
        "mean_auc_dqn": float(np.mean(dqn)),
        "mean_auc_reinforce": float(np.mean(reinforce)),
        "mean_difference_dqn_minus_reinforce": float(diff),
        "p_value": float(p_value),
        "significant": bool(p_value < ALPHA),
    }


def compute_probability_of_improvement(auc_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for env_id in ENV_IDS + ["ALL"]:
        if env_id == "ALL":
            sub = auc_df
        else:
            sub = auc_df[auc_df["env"] == env_id]

        pivot = sub.pivot_table(
            index=["env", "seed"],
            columns="algorithm",
            values="normalized_auc",
        ).dropna()

        diff = pivot["REINFORCE"].to_numpy() - pivot["DQN"].to_numpy()

        prob = float(np.mean(diff > 0) + 0.5 * np.mean(diff == 0))

        rows.append(
            {
                "env": env_id,
                "comparison": "P(REINFORCE > DQN)",
                "n_pairs": len(diff),
                "probability_of_improvement": prob,
                "mean_difference_reinforce_minus_dqn": float(np.mean(diff)),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# Plotting
# ============================================================


def plot_learning_curve(df: pd.DataFrame, env_id: str, save_path: str):
    plt.figure(figsize=(9, 5))

    env_df = df[df["env"] == env_id]

    for algo, group in env_df.groupby("algorithm"):
        summary = (
            group.groupby("step")["eval_return"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )

        summary["stderr"] = summary["std"] / np.sqrt(summary["count"])
        summary["ci95"] = 1.96 * summary["stderr"]

        x = summary["step"].to_numpy()
        mean = summary["mean"].to_numpy()
        low = mean - summary["ci95"].to_numpy()
        high = mean + summary["ci95"].to_numpy()

        plt.plot(x, mean, label=algo)
        plt.fill_between(x, low, high, alpha=0.15)

    plt.xlabel("Environment steps")
    plt.ylabel("Evaluation return")
    plt.title(f"{env_id}: DQN vs REINFORCE")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_normalized_learning_curve(df: pd.DataFrame, env_id: str, save_path: str):
    plt.figure(figsize=(9, 5))

    env_df = df[df["env"] == env_id]

    for algo, group in env_df.groupby("algorithm"):
        summary = (
            group.groupby("step")["normalized_return"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )

        summary["stderr"] = summary["std"] / np.sqrt(summary["count"])
        summary["ci95"] = 1.96 * summary["stderr"]

        x = summary["step"].to_numpy()
        mean = summary["mean"].to_numpy()
        low = mean - summary["ci95"].to_numpy()
        high = mean + summary["ci95"].to_numpy()

        plt.plot(x, mean, label=algo)
        plt.fill_between(x, low, high, alpha=0.15)

    plt.xlabel("Environment steps")
    plt.ylabel("Normalized evaluation return")
    plt.title(f"{env_id}: normalized learning curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_auc_boxplot(auc_df: pd.DataFrame, save_path: str):
    data = []
    labels = []

    for env_id in ENV_IDS:
        for algo in ALGORITHMS:
            values = auc_df[(auc_df["env"] == env_id) & (auc_df["algorithm"] == algo)][
                "normalized_auc"
            ].to_numpy()

            data.append(values)
            labels.append(f"{env_id}\n{algo}")

    plt.figure(figsize=(12, 5))
    plt.boxplot(data, labels=labels)
    plt.ylabel("Normalized AUC")
    plt.title("Normalized AUC by environment and algorithm")
    plt.xticks(rotation=25, ha="right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_overall_auc_boxplot(auc_df: pd.DataFrame, save_path: str):
    data = []
    labels = []

    for algo in ALGORITHMS:
        values = auc_df[auc_df["algorithm"] == algo]["normalized_auc"].to_numpy()
        data.append(values)
        labels.append(algo)

    plt.figure(figsize=(7, 5))
    plt.boxplot(data, labels=labels)
    plt.ylabel("Normalized AUC")
    plt.title("Overall normalized AUC across environments")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_performance_profile(auc_df: pd.DataFrame, save_path: str):
    thresholds = np.linspace(0.0, 1.0, 101)

    plt.figure(figsize=(8, 5))

    for algo in ALGORITHMS:
        values = auc_df[auc_df["algorithm"] == algo]["normalized_auc"].to_numpy()

        fractions = []

        for threshold in thresholds:
            fractions.append(np.mean(values >= threshold))

        plt.plot(thresholds, fractions, label=algo)

    plt.xlabel("Normalized AUC threshold")
    plt.ylabel("Fraction of runs above threshold")
    plt.title("Performance profile")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_aggregate_metrics(metrics_df: pd.DataFrame, save_path: str):
    x = np.arange(len(metrics_df))
    width = 0.25

    plt.figure(figsize=(8, 5))

    plt.bar(x - width, metrics_df["mean"], width, label="mean")
    plt.bar(x, metrics_df["median"], width, label="median")
    plt.bar(x + width, metrics_df["iqm"], width, label="IQM")

    plt.xticks(x, metrics_df["algorithm"])
    plt.ylabel("Normalized AUC")
    plt.title("Aggregate metrics across environments")
    plt.legend()
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


# ============================================================
# Main
# ============================================================


def main():
    print("=" * 80)
    print("Week 8 Level 3: Statistical Precipice")
    print(f"Environments: {ENV_IDS}")
    print(f"Algorithms: {ALGORITHMS}")
    print(f"Seeds: {SEEDS}")
    print(f"Device: {DEVICE}")
    print(f"Total steps: {TOTAL_STEPS}")
    print(f"Evaluation every: {EVAL_EVERY_STEPS}")
    print(f"Evaluation episodes: {EVAL_EPISODES}")
    print("=" * 80)

    raw_path = os.path.join(RESULTS_DIR, "l3_all_runs.csv")

    if os.path.exists(raw_path) and not FORCE_RERUN:
        print(f"Loading existing raw results from: {raw_path}")
        all_runs_df = pd.read_csv(raw_path)
    else:
        all_runs = []

        for env_id in ENV_IDS:
            for seed in SEEDS:
                print("\n" + "-" * 80)
                print(f"Running env={env_id}, seed={seed}")
                print("-" * 80)

                dqn_df = train_dqn(env_id, seed)
                reinforce_df = train_reinforce(env_id, seed)

                all_runs.append(dqn_df)
                all_runs.append(reinforce_df)

        all_runs_df = pd.concat(all_runs, ignore_index=True)
        all_runs_df.to_csv(raw_path, index=False)

    normalized_path = os.path.join(RESULTS_DIR, "l3_normalized_runs.csv")
    all_runs_df.to_csv(normalized_path, index=False)

    # ------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------

    auc_df = compute_auc(all_runs_df)
    auc_path = os.path.join(RESULTS_DIR, "l3_normalized_auc.csv")
    auc_df.to_csv(auc_path, index=False)

    aggregate_metrics_df = compute_aggregate_metrics(auc_df)
    aggregate_path = os.path.join(RESULTS_DIR, "l3_aggregate_metrics.csv")
    aggregate_metrics_df.to_csv(aggregate_path, index=False)

    env_tests_df = compute_env_tests(auc_df)
    env_tests_path = os.path.join(RESULTS_DIR, "l3_env_tests.csv")
    env_tests_df.to_csv(env_tests_path, index=False)

    overall_test = compute_overall_test(auc_df)
    overall_test_path = os.path.join(RESULTS_DIR, "l3_overall_test.txt")

    with open(overall_test_path, "w", encoding="utf-8") as f:
        for key, value in overall_test.items():
            f.write(f"{key}: {value}\n")

    prob_improvement_df = compute_probability_of_improvement(auc_df)
    prob_path = os.path.join(RESULTS_DIR, "l3_probability_of_improvement.csv")
    prob_improvement_df.to_csv(prob_path, index=False)

    # ------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------

    for env_id in ENV_IDS:
        safe_env_name = env_id.replace("/", "_")

        plot_learning_curve(
            all_runs_df,
            env_id,
            os.path.join(FIGURES_DIR, f"l3_{safe_env_name}_learning_curve.png"),
        )

        plot_normalized_learning_curve(
            all_runs_df,
            env_id,
            os.path.join(
                FIGURES_DIR, f"l3_{safe_env_name}_normalized_learning_curve.png"
            ),
        )

    plot_auc_boxplot(
        auc_df,
        os.path.join(FIGURES_DIR, "l3_normalized_auc_by_env_boxplot.png"),
    )

    plot_overall_auc_boxplot(
        auc_df,
        os.path.join(FIGURES_DIR, "l3_overall_normalized_auc_boxplot.png"),
    )

    plot_performance_profile(
        auc_df,
        os.path.join(FIGURES_DIR, "l3_performance_profile.png"),
    )

    plot_aggregate_metrics(
        aggregate_metrics_df,
        os.path.join(FIGURES_DIR, "l3_aggregate_metrics.png"),
    )

    # ------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------

    print("\n" + "=" * 80)
    print("Level 3 finished.")
    print("=" * 80)

    print("\nSaved results:")
    print(f"Raw runs: {raw_path}")
    print(f"Normalized runs: {normalized_path}")
    print(f"AUC: {auc_path}")
    print(f"Aggregate metrics: {aggregate_path}")
    print(f"Per-environment tests: {env_tests_path}")
    print(f"Overall test: {overall_test_path}")
    print(f"Probability of improvement: {prob_path}")

    print("\nSaved figures:")
    print(FIGURES_DIR)

    print("\nPer-environment tests:")
    print(env_tests_df)

    print("\nAggregate metrics:")
    print(aggregate_metrics_df)

    print("\nOverall test:")
    for key, value in overall_test.items():
        print(f"{key}: {value}")

    print("\nProbability of improvement:")
    print(prob_improvement_df)


if __name__ == "__main__":
    main()
