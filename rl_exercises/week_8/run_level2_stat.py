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

RESULTS_DIR = os.path.join(BASE_DIR, "results", "level2")
FIGURES_DIR = os.path.join(BASE_DIR, "figures", "level2")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ============================================================
# Config
# ============================================================

ENV_ID = "CartPole-v1"

TOTAL_STEPS = 20_000
EVAL_EVERY_STEPS = 1_000
EVAL_EPISODES = 20

SEEDS = list(range(10))

GAMMA = 0.99
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ALPHA = 0.05
N_PERMUTATIONS = 10_000


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


def evaluate_policy(policy_fn, seed: int) -> float:
    env = make_env(seed + 10_000)
    returns = []

    for _ in range(EVAL_EPISODES):
        obs, _ = env.reset()
        done = False
        ep_return = 0.0

        while not done:
            action = policy_fn(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_return += reward

        returns.append(ep_return)

    env.close()
    return float(np.mean(returns))


def compute_auc(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (algo, seed), group in df.groupby(["algorithm", "seed"]):
        group = group.sort_values("step")
        auc = np.trapezoid(group["eval_return"], group["step"])

        rows.append(
            {
                "algorithm": algo,
                "seed": seed,
                "auc": auc,
                "final_return": group["eval_return"].iloc[-1],
            }
        )

    return pd.DataFrame(rows)


def paired_permutation_test(x, y, n_permutations=10_000, seed=0):
    """
    Paired two-sided permutation test.
    x and y must have the same seed order.
    """
    rng = np.random.default_rng(seed)

    diffs = np.asarray(x) - np.asarray(y)
    observed = float(np.mean(diffs))

    count = 0

    for _ in range(n_permutations):
        signs = rng.choice([-1, 1], size=len(diffs))
        permuted_stat = np.mean(diffs * signs)

        if abs(permuted_stat) >= abs(observed):
            count += 1

    p_value = (count + 1) / (n_permutations + 1)

    return observed, p_value


# ============================================================
# DQN
# ============================================================


class QNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim):
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
    def __init__(self, capacity):
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

    def sample(self, batch_size):
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


def train_dqn(seed: int) -> pd.DataFrame:
    set_seed(seed)

    env = make_env(seed)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    q_net = QNetwork(obs_dim, action_dim).to(DEVICE)
    target_net = QNetwork(obs_dim, action_dim).to(DEVICE)
    target_net.load_state_dict(q_net.state_dict())

    optimizer = optim.Adam(q_net.parameters(), lr=1e-3)
    buffer = ReplayBuffer(capacity=50_000)

    batch_size = 64
    min_buffer_size = 1_000
    target_update_interval = 500

    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay_steps = 8_000

    global_step = 0
    next_eval_step = EVAL_EVERY_STEPS
    eval_idx = 0

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
                obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(
                    0
                )
                action = q_net(obs_t).argmax(dim=1).item()

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        buffer.push(obs, action, reward, next_obs, done)

        obs = next_obs
        global_step += 1

        if done:
            obs, _ = env.reset()

        if len(buffer) >= min_buffer_size:
            batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones = (
                buffer.sample(batch_size)
            )

            q_values = q_net(batch_obs)
            q_values = q_values.gather(1, batch_actions.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                next_q = target_net(batch_next_obs).max(dim=1)[0]
                target = batch_rewards + GAMMA * (1.0 - batch_dones) * next_q

            loss = nn.functional.mse_loss(q_values, target)

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
                        obs_eval, dtype=torch.float32, device=DEVICE
                    ).unsqueeze(0)
                    return q_net(obs_t).argmax(dim=1).item()

            eval_return = evaluate_policy(policy_fn, seed)

            logs.append(
                {
                    "algorithm": "DQN",
                    "seed": seed,
                    "eval_idx": eval_idx,
                    "step": next_eval_step,
                    "eval_return": eval_return,
                }
            )

            print(
                f"DQN       | seed={seed:2d} | "
                f"step={next_eval_step:6d} | "
                f"eval_return={eval_return:8.2f}"
            )

            next_eval_step += EVAL_EVERY_STEPS

    env.close()
    return pd.DataFrame(logs)


# ============================================================
# REINFORCE
# ============================================================


class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.net(x)


def train_reinforce(seed: int) -> pd.DataFrame:
    set_seed(seed)

    env = make_env(seed)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy = PolicyNetwork(obs_dim, action_dim).to(DEVICE)
    optimizer = optim.Adam(policy.parameters(), lr=1e-2)

    global_step = 0
    next_eval_step = EVAL_EVERY_STEPS
    eval_idx = 0

    logs = []

    while global_step < TOTAL_STEPS:
        obs, _ = env.reset()

        log_probs = []
        rewards = []
        done = False

        while not done and global_step < TOTAL_STEPS:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)

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
                            obs_eval, dtype=torch.float32, device=DEVICE
                        ).unsqueeze(0)
                        logits_eval = policy(obs_eval_t)
                        return logits_eval.argmax(dim=1).item()

                eval_return = evaluate_policy(policy_fn, seed)

                logs.append(
                    {
                        "algorithm": "REINFORCE",
                        "seed": seed,
                        "eval_idx": eval_idx,
                        "step": next_eval_step,
                        "eval_return": eval_return,
                    }
                )

                print(
                    f"REINFORCE | seed={seed:2d} | "
                    f"step={next_eval_step:6d} | "
                    f"eval_return={eval_return:8.2f}"
                )

                next_eval_step += EVAL_EVERY_STEPS

        # policy update after one episode
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
# Plot
# ============================================================


def plot_learning_curves(df: pd.DataFrame, save_path: str):
    plt.figure(figsize=(9, 5))

    for algo, group in df.groupby("algorithm"):
        summary = (
            group.groupby("step")["eval_return"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )

        summary["stderr"] = summary["std"] / np.sqrt(summary["count"])
        summary["ci95"] = 1.96 * summary["stderr"]

        x = summary["step"].to_numpy()
        mean = summary["mean"].to_numpy()
        ci_low = mean - summary["ci95"].to_numpy()
        ci_high = mean + summary["ci95"].to_numpy()

        plt.plot(x, mean, label=algo)
        plt.fill_between(x, ci_low, ci_high, alpha=0.15)

    plt.xlabel("Environment steps")
    plt.ylabel("Evaluation return")
    plt.title("Level 2: DQN vs REINFORCE")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_auc_boxplot(auc_df: pd.DataFrame, save_path: str):
    dqn_auc = auc_df[auc_df["algorithm"] == "DQN"]["auc"].to_numpy()
    reinforce_auc = auc_df[auc_df["algorithm"] == "REINFORCE"]["auc"].to_numpy()

    plt.figure(figsize=(7, 5))
    plt.boxplot([dqn_auc, reinforce_auc], labels=["DQN", "REINFORCE"])
    plt.ylabel("AUC over learning curve")
    plt.title("Level 2: AUC distribution over seeds")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


# ============================================================
# Main
# ============================================================


def main():
    print("=" * 80)
    print("Week 8 Level 2: Statistical Testing")
    print(f"Environment: {ENV_ID}")
    print("Algorithms: DQN vs REINFORCE")
    print(f"Seeds: {SEEDS}")
    print(f"Device: {DEVICE}")
    print("=" * 80)

    all_runs = []

    for seed in SEEDS:
        print("\n" + "-" * 80)
        print(f"Running seed {seed}")
        print("-" * 80)

        dqn_df = train_dqn(seed)
        reinforce_df = train_reinforce(seed)

        all_runs.append(dqn_df)
        all_runs.append(reinforce_df)

    results_df = pd.concat(all_runs, ignore_index=True)

    raw_path = os.path.join(RESULTS_DIR, "l2_dqn_vs_reinforce_runs.csv")
    results_df.to_csv(raw_path, index=False)

    auc_df = compute_auc(results_df)

    auc_path = os.path.join(RESULTS_DIR, "l2_auc_per_seed.csv")
    auc_df.to_csv(auc_path, index=False)

    plot_learning_curves(
        results_df,
        os.path.join(FIGURES_DIR, "l2_learning_curves.png"),
    )

    plot_auc_boxplot(
        auc_df,
        os.path.join(FIGURES_DIR, "l2_auc_boxplot.png"),
    )

    # Paired test: same seeds for both algorithms
    pivot = auc_df.pivot(index="seed", columns="algorithm", values="auc").sort_index()

    dqn_auc = pivot["DQN"].to_numpy()
    reinforce_auc = pivot["REINFORCE"].to_numpy()

    observed_diff, p_value = paired_permutation_test(
        dqn_auc,
        reinforce_auc,
        n_permutations=N_PERMUTATIONS,
        seed=0,
    )

    test_result = {
        "test": "paired permutation test",
        "aggregation": "AUC over evaluation curve per seed",
        "alpha": ALPHA,
        "n_seeds": len(SEEDS),
        "mean_auc_dqn": float(np.mean(dqn_auc)),
        "mean_auc_reinforce": float(np.mean(reinforce_auc)),
        "mean_auc_difference_dqn_minus_reinforce": float(observed_diff),
        "p_value": float(p_value),
        "significant": bool(p_value < ALPHA),
    }

    test_path = os.path.join(RESULTS_DIR, "l2_test_result.txt")

    with open(test_path, "w", encoding="utf-8") as f:
        for key, value in test_result.items():
            f.write(f"{key}: {value}\n")

    print("\n" + "=" * 80)
    print("Level 2 finished.")
    print(f"Raw runs saved to: {raw_path}")
    print(f"AUC results saved to: {auc_path}")
    print(f"Test result saved to: {test_path}")
    print("=" * 80)

    print("\nTest result:")
    for key, value in test_result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
