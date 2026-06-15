"""
Compare:
1. DQN
2. RND-DQN
3. Ensemble-DQN

CSV results are saved to:
rl_exercises/week_7/results/level3/

Figures are saved to:
rl_exercises/week_7/figures/level3/
"""

from typing import Any, Dict, List, Tuple

from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rl_exercises.week_4.dqn import DQNAgent
from rl_exercises.week_4.dqn import set_seed as set_seed_dqn
from rl_exercises.week_7.ensemble_dqn import EnsembleDQNAgent
from rl_exercises.week_7.rnd_dqn import RNDDQNAgent
from rliable import library as rly
from rliable import plot_utils

# ============================================================
# Config
# ============================================================

ENV_NAME = "LunarLander-v3"

SEEDS = [0, 1, 2]

NUM_FRAMES = 50_000
EVAL_INTERVAL = 5_000
EVAL_EPISODES = 5

BUFFER_CAPACITY = 50_000
BATCH_SIZE = 64
LR = 1e-3
GAMMA = 0.99

EPSILON_START = 1.0
EPSILON_FINAL = 0.05
EPSILON_DECAY = 10_000
TARGET_UPDATE_FREQ = 1_000

HIDDEN_SIZE = 128

# RND settings
RND_HIDDEN_SIZE = 128
RND_LR = 1e-3
RND_UPDATE_FREQ = 1_000
RND_N_LAYERS = 2
RND_REWARD_WEIGHT = 0.1

# Ensemble settings
NUM_ENSEMBLE = 5
UCB_BETA = 1.0

RLIABLE_REPS = 2_000

BASE_DIR = Path("rl_exercises/week_7")
RESULT_DIR = BASE_DIR / "results" / "level3"
FIGURE_DIR = BASE_DIR / "figures" / "level3"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Evaluation
# ============================================================


def evaluate_dqn_like_agent(
    agent: Any,
    env_name: str,
    seed: int,
    num_episodes: int = 5,
    ensemble_eval: bool = False,
) -> Tuple[float, float]:
    eval_env = gym.make(env_name)
    returns = []

    old_epsilon_start = getattr(agent, "epsilon_start", None)
    old_epsilon_final = getattr(agent, "epsilon_final", None)

    if hasattr(agent, "epsilon_start") and hasattr(agent, "epsilon_final"):
        agent.epsilon_start = 0.0
        agent.epsilon_final = 0.0

    for ep in range(num_episodes):
        state, _ = eval_env.reset(seed=seed + 10_000 + ep)
        done = False
        total_return = 0.0

        while not done:
            if ensemble_eval:
                action = agent.predict_action(state, eval_mode=True)
            else:
                action = agent.predict_action(state)

            state, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            total_return += reward

        returns.append(total_return)

    if old_epsilon_start is not None:
        agent.epsilon_start = old_epsilon_start
    if old_epsilon_final is not None:
        agent.epsilon_final = old_epsilon_final

    eval_env.close()
    return float(np.mean(returns)), float(np.std(returns))


def make_eval_row(
    agent_name: str,
    seed: int,
    step: int,
    mean_return: float,
    std_return: float,
) -> Dict[str, Any]:
    return {
        "agent": agent_name,
        "seed": seed,
        "step": step,
        "eval_mean_return": mean_return,
        "eval_std_return": std_return,
    }


# ============================================================
# Training loops
# ============================================================


def train_dqn_with_logging(
    agent: DQNAgent,
    env: gym.Env,
    num_frames: int,
    agent_name: str,
    seed: int,
) -> pd.DataFrame:
    state, _ = env.reset(seed=seed)
    eval_logs: List[Dict[str, Any]] = []

    for frame in range(1, num_frames + 1):
        action = agent.predict_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.buffer.add(state, action, reward, next_state, done, {})
        state = next_state

        if len(agent.buffer) >= agent.batch_size:
            batch = agent.buffer.sample(agent.batch_size)
            agent.update_agent(batch)

        if done:
            state, _ = env.reset()

        if frame % EVAL_INTERVAL == 0:
            mean_r, std_r = evaluate_dqn_like_agent(
                agent=agent,
                env_name=ENV_NAME,
                seed=seed,
                num_episodes=EVAL_EPISODES,
            )

            eval_logs.append(make_eval_row(agent_name, seed, frame, mean_r, std_r))

            print(
                f"[Eval] {agent_name:12s} | Seed {seed} | "
                f"Step {frame:6d} | Return {mean_r:8.2f} ± {std_r:6.2f}"
            )

    return pd.DataFrame(eval_logs)


def train_rnd_dqn_with_logging(
    agent: RNDDQNAgent,
    env: gym.Env,
    num_frames: int,
    agent_name: str,
    seed: int,
) -> pd.DataFrame:
    state, _ = env.reset(seed=seed)
    eval_logs: List[Dict[str, Any]] = []

    for frame in range(1, num_frames + 1):
        action = agent.predict_action(state)
        next_state, ext_reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        rnd_bonus = agent.get_rnd_bonus(next_state)
        total_reward = ext_reward + rnd_bonus

        agent.buffer.add(state, action, total_reward, next_state, done, {})
        state = next_state

        if len(agent.buffer) >= agent.batch_size:
            batch = agent.buffer.sample(agent.batch_size)
            agent.update_agent(batch)

            if frame % agent.rnd_update_freq == 0:
                agent.update_rnd(batch)

        if done:
            state, _ = env.reset()

        if frame % EVAL_INTERVAL == 0:
            mean_r, std_r = evaluate_dqn_like_agent(
                agent=agent,
                env_name=ENV_NAME,
                seed=seed,
                num_episodes=EVAL_EPISODES,
            )

            eval_logs.append(make_eval_row(agent_name, seed, frame, mean_r, std_r))

            print(
                f"[Eval] {agent_name:12s} | Seed {seed} | "
                f"Step {frame:6d} | Return {mean_r:8.2f} ± {std_r:6.2f}"
            )

    return pd.DataFrame(eval_logs)


def train_ensemble_dqn_with_logging(
    agent: EnsembleDQNAgent,
    env: gym.Env,
    num_frames: int,
    agent_name: str,
    seed: int,
) -> pd.DataFrame:
    state, _ = env.reset(seed=seed)
    eval_logs: List[Dict[str, Any]] = []

    for frame in range(1, num_frames + 1):
        action = agent.predict_action(state, eval_mode=False)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.buffer.add(state, action, reward, next_state, done, {})
        state = next_state

        if len(agent.buffer) >= agent.batch_size:
            batch = agent.buffer.sample(agent.batch_size)
            agent.update_agent(batch)

        if done:
            state, _ = env.reset()

        if frame % EVAL_INTERVAL == 0:
            mean_r, std_r = evaluate_dqn_like_agent(
                agent=agent,
                env_name=ENV_NAME,
                seed=seed,
                num_episodes=EVAL_EPISODES,
                ensemble_eval=True,
            )

            eval_logs.append(make_eval_row(agent_name, seed, frame, mean_r, std_r))

            print(
                f"[Eval] {agent_name:12s} | Seed {seed} | "
                f"Step {frame:6d} | Return {mean_r:8.2f} ± {std_r:6.2f}"
            )

    return pd.DataFrame(eval_logs)


# ============================================================
# Runners
# ============================================================


def run_dqn(seed: int) -> pd.DataFrame:
    env = gym.make(ENV_NAME)
    set_seed_dqn(env, seed)

    agent = DQNAgent(
        env=env,
        buffer_capacity=BUFFER_CAPACITY,
        batch_size=BATCH_SIZE,
        lr=LR,
        gamma=GAMMA,
        epsilon_start=EPSILON_START,
        epsilon_final=EPSILON_FINAL,
        epsilon_decay=EPSILON_DECAY,
        target_update_freq=TARGET_UPDATE_FREQ,
        seed=seed,
    )

    df = train_dqn_with_logging(agent, env, NUM_FRAMES, "DQN", seed)
    env.close()
    return df


def run_rnd_dqn(seed: int) -> pd.DataFrame:
    env = gym.make(ENV_NAME)
    set_seed_dqn(env, seed)

    agent = RNDDQNAgent(
        env=env,
        buffer_capacity=BUFFER_CAPACITY,
        batch_size=BATCH_SIZE,
        lr=LR,
        gamma=GAMMA,
        epsilon_start=EPSILON_START,
        epsilon_final=EPSILON_FINAL,
        epsilon_decay=EPSILON_DECAY,
        target_update_freq=TARGET_UPDATE_FREQ,
        seed=seed,
        rnd_hidden_size=RND_HIDDEN_SIZE,
        rnd_lr=RND_LR,
        rnd_update_freq=RND_UPDATE_FREQ,
        rnd_n_layers=RND_N_LAYERS,
        rnd_reward_weight=RND_REWARD_WEIGHT,
    )

    df = train_rnd_dqn_with_logging(agent, env, NUM_FRAMES, "RND-DQN", seed)
    env.close()
    return df


def run_ensemble_dqn(seed: int) -> pd.DataFrame:
    env = gym.make(ENV_NAME)
    set_seed_dqn(env, seed)

    agent = EnsembleDQNAgent(
        env=env,
        num_ensemble=NUM_ENSEMBLE,
        buffer_capacity=BUFFER_CAPACITY,
        batch_size=BATCH_SIZE,
        lr=LR,
        gamma=GAMMA,
        epsilon_start=EPSILON_START,
        epsilon_final=EPSILON_FINAL,
        epsilon_decay=EPSILON_DECAY,
        target_update_freq=TARGET_UPDATE_FREQ,
        hidden_size=HIDDEN_SIZE,
        ucb_beta=UCB_BETA,
        seed=seed,
    )

    df = train_ensemble_dqn_with_logging(agent, env, NUM_FRAMES, "Ensemble-DQN", seed)

    env.close()
    return df


# ============================================================
# Plotting
# ============================================================


def build_score_dict(
    df: pd.DataFrame,
    agents: List[str],
) -> Dict[str, np.ndarray]:
    eval_steps = np.arange(EVAL_INTERVAL, NUM_FRAMES + 1, EVAL_INTERVAL)
    score_dict: Dict[str, List[np.ndarray]] = {}

    for agent_name in agents:
        score_dict[agent_name] = []

        for seed in SEEDS:
            sub = df[(df["agent"] == agent_name) & (df["seed"] == seed)]
            sub = sub.sort_values("step")

            if len(sub) == 0:
                raise ValueError(f"No data found for agent={agent_name}, seed={seed}")

            x = sub["step"].to_numpy()
            y = sub["eval_mean_return"].to_numpy()

            aligned_y = np.interp(eval_steps, x, y, left=y[0], right=y[-1])
            score_dict[agent_name].append(aligned_y)

    return {
        agent_name: np.stack(curves, axis=0)
        for agent_name, curves in score_dict.items()
    }


def save_aligned_rliable_scores(
    score_dict: Dict[str, np.ndarray],
    filename: str,
) -> None:
    eval_steps = np.arange(EVAL_INTERVAL, NUM_FRAMES + 1, EVAL_INTERVAL)
    rows = []

    for agent_name, scores in score_dict.items():
        for seed_idx, seed in enumerate(SEEDS):
            for step_idx, step in enumerate(eval_steps):
                rows.append(
                    {
                        "agent": agent_name,
                        "seed": seed,
                        "step": step,
                        "aligned_eval_mean_return": scores[seed_idx, step_idx],
                    }
                )

    path = RESULT_DIR / filename
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Saved aligned RLiable data: {path}")


def plot_rliable_curve(
    df: pd.DataFrame,
    agents: List[str],
    title: str,
    filename: str,
    aligned_filename: str,
) -> None:
    eval_steps = np.arange(EVAL_INTERVAL, NUM_FRAMES + 1, EVAL_INTERVAL)
    score_dict = build_score_dict(df, agents)

    save_aligned_rliable_scores(score_dict, aligned_filename)

    def aggregate_mean_curve(scores: np.ndarray) -> np.ndarray:
        return np.mean(scores, axis=0)

    point_estimates, interval_estimates = rly.get_interval_estimates(
        score_dict,
        aggregate_mean_curve,
        reps=RLIABLE_REPS,
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    plot_utils.plot_sample_efficiency_curve(
        eval_steps,
        point_estimates,
        interval_estimates,
        algorithms=agents,
        xlabel="Environment steps",
        ylabel="Evaluation return",
        ax=ax,
    )

    ax.set_title(title)

    # Add legend manually.
    handles, labels = ax.get_legend_handles_labels()

    if len(labels) == 0:
        handles = ax.get_lines()[: len(agents)]
        labels = agents

    ax.legend(
        handles,
        labels,
        loc="best",
        frameon=True,
        fontsize=10,
    )

    path = FIGURE_DIR / filename
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved RLiable figure: {path}")


def plot_all_figures(results: pd.DataFrame) -> None:
    plot_rliable_curve(
        df=results,
        agents=["DQN", "RND-DQN", "Ensemble-DQN"],
        title="Week 7 Level 3: DQN vs RND-DQN vs Ensemble-DQN",
        filename="level3_dqn_vs_rnd_vs_ensemble_rliable.png",
        aligned_filename="level3_dqn_vs_rnd_vs_ensemble_aligned.csv",
    )

    plot_rliable_curve(
        df=results,
        agents=["RND-DQN", "Ensemble-DQN"],
        title="Week 7 Level 3: RND-DQN vs Ensemble-DQN",
        filename="level3_rnd_vs_ensemble_rliable.png",
        aligned_filename="level3_rnd_vs_ensemble_aligned.csv",
    )


# ============================================================
# Config saving
# ============================================================


def save_config() -> None:
    config: Dict[str, Any] = {
        "env_name": ENV_NAME,
        "seeds": SEEDS,
        "num_frames": NUM_FRAMES,
        "eval_interval": EVAL_INTERVAL,
        "eval_episodes": EVAL_EPISODES,
        "buffer_capacity": BUFFER_CAPACITY,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "gamma": GAMMA,
        "epsilon_start": EPSILON_START,
        "epsilon_final": EPSILON_FINAL,
        "epsilon_decay": EPSILON_DECAY,
        "target_update_freq": TARGET_UPDATE_FREQ,
        "hidden_size": HIDDEN_SIZE,
        "rnd_hidden_size": RND_HIDDEN_SIZE,
        "rnd_lr": RND_LR,
        "rnd_update_freq": RND_UPDATE_FREQ,
        "rnd_n_layers": RND_N_LAYERS,
        "rnd_reward_weight": RND_REWARD_WEIGHT,
        "num_ensemble": NUM_ENSEMBLE,
        "ucb_beta": UCB_BETA,
        "result_dir": str(RESULT_DIR),
        "figure_dir": str(FIGURE_DIR),
    }

    path = RESULT_DIR / "level3_config.txt"

    with open(path, "w", encoding="utf-8") as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")

    print(f"Saved config: {path}")


# ============================================================
# Main
# ============================================================


def main() -> None:
    all_results: List[pd.DataFrame] = []

    runners = [
        ("DQN", run_dqn, "dqn"),
        ("RND-DQN", run_rnd_dqn, "rnd_dqn"),
        ("Ensemble-DQN", run_ensemble_dqn, "ensemble_dqn"),
    ]

    for seed in SEEDS:
        for agent_name, runner, filename_prefix in runners:
            print("=" * 80)
            print(f"Running {agent_name} | seed={seed}")

            df = runner(seed)
            all_results.append(df)

            seed_path = RESULT_DIR / f"{filename_prefix}_seed_{seed}.csv"
            df.to_csv(seed_path, index=False)

            print(f"Saved CSV: {seed_path}")

    results = pd.concat(all_results, ignore_index=True)

    all_csv_path = RESULT_DIR / "level3_all_results.csv"
    results.to_csv(all_csv_path, index=False)

    print(f"Saved all results: {all_csv_path}")

    save_config()
    plot_all_figures(results)

    print("=" * 80)
    print(f"All CSV files saved in: {RESULT_DIR}")
    print(f"All figures saved in: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
