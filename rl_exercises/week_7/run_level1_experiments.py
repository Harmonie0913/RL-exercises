"""
Run Week 7 Level 1 experiments.

Compare:
1. DQN epsilon-greedy baseline
2. RND-DQN
3. PPO baseline
4. RND-PPO

CSV results are saved to:
rl_exercises/week_7/results/level1/

RLiable figures are saved to:
rl_exercises/week_7/figures/level1/
"""

from typing import Any, Dict, List, Tuple

from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from rl_exercises.week_4.dqn import DQNAgent
from rl_exercises.week_4.dqn import set_seed as set_seed_dqn
from rl_exercises.week_6.ppo import PPOAgent
from rl_exercises.week_6.ppo import set_seed as set_seed_ppo
from rl_exercises.week_7.rnd_dqn import RNDDQNAgent
from rl_exercises.week_7.rnd_ppo import RNDPPOAgent
from rliable import library as rly
from rliable import plot_utils

# ============================================================
# Config
# ============================================================

ENV_NAME = "LunarLander-v3"

SEEDS = [0, 1, 2]

# DQN settings
DQN_NUM_FRAMES = 50_000
DQN_BUFFER_CAPACITY = 50_000
DQN_BATCH_SIZE = 64
DQN_LR = 1e-3
DQN_GAMMA = 0.99
DQN_EPSILON_START = 1.0
DQN_EPSILON_FINAL = 0.05
DQN_EPSILON_DECAY = 10_000
DQN_TARGET_UPDATE_FREQ = 1_000

# PPO settings
PPO_TOTAL_STEPS = 50_000
PPO_LR_ACTOR = 5e-4
PPO_LR_CRITIC = 1e-3
PPO_GAMMA = 0.99
PPO_GAE_LAMBDA = 0.95
PPO_CLIP_EPS = 0.2
PPO_EPOCHS = 4
PPO_BATCH_SIZE = 64
PPO_ENT_COEF = 0.01
PPO_VF_COEF = 0.5
PPO_HIDDEN_SIZE = 128

# RND settings
RND_HIDDEN_SIZE = 128
RND_LR = 1e-3
RND_COMBINED_LR = 1e-4
RND_UPDATE_FREQ = 1_000
RND_PPO_UPDATE_FREQ = 4
RND_N_LAYERS = 2
RND_REWARD_WEIGHT = 0.1
RND_UPDATE_PROPORTION = 0.25
RND_INT_COEF = 1.0
RND_EXT_COEF = 2.0
RND_INT_GAMMA = 0.99
RND_OBS_NORM_INIT_EPISODES = 10

# Evaluation / plotting settings
EVAL_INTERVAL = 5_000
EVAL_EPISODES = 5
RLIABLE_REPS = 2_000

RESULT_DIR = Path("rl_exercises/week_7/results/level1")
FIGURE_DIR = Path("rl_exercises/week_7/figures/level1")

RESULT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# General helper functions
# ============================================================


def moving_average(values: List[float], window: int = 10) -> np.ndarray:
    return pd.Series(values).rolling(window=window, min_periods=1).mean().to_numpy()


def evaluate_dqn_like_agent(
    agent: Any,
    env_name: str,
    seed: int,
    num_episodes: int = 5,
) -> Tuple[float, float]:
    """
    Evaluate DQN-like agents greedily.

    DQNAgent.predict_action normally still uses epsilon-greedy.
    Therefore, for evaluation we temporarily force epsilon to its final value
    if the agent exposes epsilon_final.
    """
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


def evaluate_ppo_like_agent(
    agent: Any,
    env_name: str,
    seed: int,
    num_episodes: int = 5,
) -> Tuple[float, float]:
    """
    Evaluate PPO-like agents using only extrinsic environment reward.
    """
    eval_env = gym.make(env_name)

    if hasattr(agent, "evaluate"):
        try:
            mean_r, std_r = agent.evaluate(eval_env, num_episodes=num_episodes)
            eval_env.close()
            return float(mean_r), float(std_r)
        except Exception:
            pass

    returns = []

    for ep in range(num_episodes):
        state, _ = eval_env.reset(seed=seed + 20_000 + ep)
        done = False
        total_return = 0.0

        while not done:
            pred = agent.predict(state)
            action = pred[0]
            state, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            total_return += reward

        returns.append(total_return)

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
# DQN training
# ============================================================


def train_dqn_with_logging(
    agent: DQNAgent,
    env: gym.Env,
    num_frames: int,
    agent_name: str,
    seed: int,
) -> pd.DataFrame:
    state, _ = env.reset(seed=seed)

    ep_reward = 0.0
    eval_logs: List[Dict[str, Any]] = []

    for frame in range(1, num_frames + 1):
        action = agent.predict_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.buffer.add(state, action, reward, next_state, done, {})

        state = next_state
        ep_reward += reward

        if len(agent.buffer) >= agent.batch_size:
            batch = agent.buffer.sample(agent.batch_size)
            agent.update_agent(batch)

        if done:
            state, _ = env.reset()
            ep_reward = 0.0

        if frame % EVAL_INTERVAL == 0:
            mean_r, std_r = evaluate_dqn_like_agent(
                agent=agent,
                env_name=ENV_NAME,
                seed=seed,
                num_episodes=EVAL_EPISODES,
            )
            eval_logs.append(make_eval_row(agent_name, seed, frame, mean_r, std_r))

            print(
                f"[Eval] {agent_name:8s} | Seed {seed} | "
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

    ep_ext_reward = 0.0
    ep_int_reward = 0.0

    for frame in range(1, num_frames + 1):
        action = agent.predict_action(state)
        next_state, ext_reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        rnd_bonus = agent.get_rnd_bonus(next_state)
        total_reward = ext_reward + rnd_bonus

        agent.buffer.add(state, action, total_reward, next_state, done, {})

        state = next_state
        ep_ext_reward += ext_reward
        ep_int_reward += rnd_bonus

        if len(agent.buffer) >= agent.batch_size:
            batch = agent.buffer.sample(agent.batch_size)
            agent.update_agent(batch)

            if frame % agent.rnd_update_freq == 0:
                agent.update_rnd(batch)

        if done:
            state, _ = env.reset()
            ep_ext_reward = 0.0
            ep_int_reward = 0.0

        if frame % EVAL_INTERVAL == 0:
            mean_r, std_r = evaluate_dqn_like_agent(
                agent=agent,
                env_name=ENV_NAME,
                seed=seed,
                num_episodes=EVAL_EPISODES,
            )
            eval_logs.append(make_eval_row(agent_name, seed, frame, mean_r, std_r))

            print(
                f"[Eval] {agent_name:8s} | Seed {seed} | "
                f"Step {frame:6d} | Return {mean_r:8.2f} ± {std_r:6.2f}"
            )

    return pd.DataFrame(eval_logs)


def run_dqn(seed: int) -> pd.DataFrame:
    env = gym.make(ENV_NAME)
    set_seed_dqn(env, seed)

    agent = DQNAgent(
        env=env,
        buffer_capacity=DQN_BUFFER_CAPACITY,
        batch_size=DQN_BATCH_SIZE,
        lr=DQN_LR,
        gamma=DQN_GAMMA,
        epsilon_start=DQN_EPSILON_START,
        epsilon_final=DQN_EPSILON_FINAL,
        epsilon_decay=DQN_EPSILON_DECAY,
        target_update_freq=DQN_TARGET_UPDATE_FREQ,
        seed=seed,
    )

    df = train_dqn_with_logging(agent, env, DQN_NUM_FRAMES, "DQN", seed)
    env.close()
    return df


def run_rnd_dqn(seed: int) -> pd.DataFrame:
    env = gym.make(ENV_NAME)
    set_seed_dqn(env, seed)

    agent = RNDDQNAgent(
        env=env,
        buffer_capacity=DQN_BUFFER_CAPACITY,
        batch_size=DQN_BATCH_SIZE,
        lr=DQN_LR,
        gamma=DQN_GAMMA,
        epsilon_start=DQN_EPSILON_START,
        epsilon_final=DQN_EPSILON_FINAL,
        epsilon_decay=DQN_EPSILON_DECAY,
        target_update_freq=DQN_TARGET_UPDATE_FREQ,
        seed=seed,
        rnd_hidden_size=RND_HIDDEN_SIZE,
        rnd_lr=RND_LR,
        rnd_update_freq=RND_UPDATE_FREQ,
        rnd_n_layers=RND_N_LAYERS,
        rnd_reward_weight=RND_REWARD_WEIGHT,
    )

    df = train_rnd_dqn_with_logging(agent, env, DQN_NUM_FRAMES, "RND-DQN", seed)
    env.close()
    return df


# ============================================================
# PPO training
# ============================================================


def train_ppo_with_logging(
    agent: PPOAgent,
    env: gym.Env,
    total_steps: int,
    agent_name: str,
    seed: int,
) -> pd.DataFrame:
    """
    Custom PPO training loop.

    Expected trajectory format for the week 6 PPO implementation:
        (state, action, logp, entropy, reward, done, next_state)

    This mirrors the RND-PPO trajectory format, but without intrinsic reward.
    """
    step_count = 0
    eval_logs: List[Dict[str, Any]] = []

    while step_count < total_steps:
        state, _ = env.reset(seed=seed)
        done = False
        trajectory: List[Any] = []

        while not done and step_count < total_steps:
            pred = agent.predict(state)

            action = pred[0]
            logp = pred[1]
            entropy = pred[2] if len(pred) > 2 else torch.tensor(0.0)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            trajectory.append(
                (
                    state,
                    action,
                    logp,
                    entropy,
                    float(reward),
                    float(done),
                    next_state,
                )
            )

            state = next_state
            step_count += 1

            if step_count % EVAL_INTERVAL == 0:
                mean_r, std_r = evaluate_ppo_like_agent(
                    agent=agent,
                    env_name=ENV_NAME,
                    seed=seed,
                    num_episodes=EVAL_EPISODES,
                )
                eval_logs.append(
                    make_eval_row(agent_name, seed, step_count, mean_r, std_r)
                )

                print(
                    f"[Eval] {agent_name:8s} | Seed {seed} | "
                    f"Step {step_count:6d} | Return {mean_r:8.2f} ± {std_r:6.2f}"
                )

        agent.update(trajectory)

    return pd.DataFrame(eval_logs)


def train_rnd_ppo_with_logging(
    agent: RNDPPOAgent,
    env: gym.Env,
    total_steps: int,
    agent_name: str,
    seed: int,
) -> pd.DataFrame:
    """
    Custom RND-PPO training loop.

    The agent learns from:
        extrinsic reward + normalized intrinsic reward

    Evaluation uses only extrinsic environment reward.
    """
    step_count = 0
    eval_logs: List[Dict[str, Any]] = []

    agent.rnd_update_counter = 0
    agent._init_obs_normalization()

    while step_count < total_steps:
        state, _ = env.reset(seed=seed)
        done = False
        trajectory: List[Any] = []

        while not done and step_count < total_steps:
            action, logp, entropy, _, _ = agent.predict(state)

            next_state, ext_reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.obs_rms.update(next_state[np.newaxis])
            obs_norm = (next_state - agent.obs_rms.mean) / np.sqrt(
                agent.obs_rms.var + 1e-8
            )

            int_reward_raw = agent.get_rnd_bonus(obs_norm.astype(np.float32))
            discounted = agent.discounted_reward.update(np.array([int_reward_raw]))
            agent.reward_rms.update(discounted)
            int_reward = int_reward_raw / np.sqrt(agent.reward_rms.var + 1e-8)

            trajectory.append(
                (
                    state,
                    action,
                    logp,
                    entropy,
                    float(ext_reward),
                    float(int_reward),
                    float(done),
                    next_state,
                )
            )

            state = next_state
            step_count += 1
            agent.rnd_update_counter += 1

            if step_count % EVAL_INTERVAL == 0:
                mean_r, std_r = evaluate_ppo_like_agent(
                    agent=agent,
                    env_name=ENV_NAME,
                    seed=seed,
                    num_episodes=EVAL_EPISODES,
                )
                eval_logs.append(
                    make_eval_row(agent_name, seed, step_count, mean_r, std_r)
                )

                print(
                    f"[Eval] {agent_name:8s} | Seed {seed} | "
                    f"Step {step_count:6d} | Return {mean_r:8.2f} ± {std_r:6.2f}"
                )

        agent.update(trajectory)

    return pd.DataFrame(eval_logs)


def run_ppo(seed: int) -> pd.DataFrame:
    env = gym.make(ENV_NAME)
    set_seed_ppo(env, seed)

    agent = PPOAgent(
        env=env,
        lr_actor=PPO_LR_ACTOR,
        lr_critic=PPO_LR_CRITIC,
        gamma=PPO_GAMMA,
        gae_lambda=PPO_GAE_LAMBDA,
        clip_eps=PPO_CLIP_EPS,
        epochs=PPO_EPOCHS,
        batch_size=PPO_BATCH_SIZE,
        ent_coef=PPO_ENT_COEF,
        vf_coef=PPO_VF_COEF,
        seed=seed,
        hidden_size=PPO_HIDDEN_SIZE,
    )

    df = train_ppo_with_logging(agent, env, PPO_TOTAL_STEPS, "PPO", seed)
    env.close()
    return df


def run_rnd_ppo(seed: int) -> pd.DataFrame:
    env = gym.make(ENV_NAME)
    set_seed_ppo(env, seed)

    agent = RNDPPOAgent(
        env=env,
        lr_actor=PPO_LR_ACTOR,
        lr_critic=PPO_LR_CRITIC,
        gamma=PPO_GAMMA,
        gae_lambda=PPO_GAE_LAMBDA,
        clip_eps=PPO_CLIP_EPS,
        epochs=PPO_EPOCHS,
        batch_size=PPO_BATCH_SIZE,
        ent_coef=PPO_ENT_COEF,
        vf_coef=PPO_VF_COEF,
        seed=seed,
        hidden_size=PPO_HIDDEN_SIZE,
        rnd_hidden_size=RND_HIDDEN_SIZE,
        combined_lr=RND_COMBINED_LR,
        rnd_update_freq=RND_PPO_UPDATE_FREQ,
        rnd_n_layers=RND_N_LAYERS,
        rnd_reward_weight=RND_REWARD_WEIGHT,
        update_proportion=RND_UPDATE_PROPORTION,
        int_coef=RND_INT_COEF,
        ext_coef=RND_EXT_COEF,
        int_gamma=RND_INT_GAMMA,
        num_iterations_obs_norm_init=RND_OBS_NORM_INIT_EPISODES,
    )

    df = train_rnd_ppo_with_logging(agent, env, PPO_TOTAL_STEPS, "RND-PPO", seed)
    env.close()
    return df


# ============================================================
# RLiable plotting
# ============================================================


def build_score_dict(
    df: pd.DataFrame,
    agents: List[str],
) -> Dict[str, np.ndarray]:
    """
    RLiable expects:
        score_dict[algorithm].shape == (num_runs, num_eval_points)
    """
    eval_steps = np.arange(EVAL_INTERVAL, PPO_TOTAL_STEPS + 1, EVAL_INTERVAL)
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
    eval_steps = np.arange(EVAL_INTERVAL, PPO_TOTAL_STEPS + 1, EVAL_INTERVAL)
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
    eval_steps = np.arange(EVAL_INTERVAL, PPO_TOTAL_STEPS + 1, EVAL_INTERVAL)
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

    # Add legend manually because rliable sometimes does not show it automatically.
    handles, labels = ax.get_legend_handles_labels()

    if len(labels) == 0:
        # Fallback: use the plotted mean curves as legend handles.
        line_handles = ax.get_lines()
        handles = line_handles[: len(agents)]
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
        agents=["DQN", "RND-DQN"],
        title="Week 7 Level 1: DQN vs RND-DQN",
        filename="level1_dqn_vs_rnd_dqn_rliable.png",
        aligned_filename="level1_dqn_vs_rnd_dqn_aligned.csv",
    )

    plot_rliable_curve(
        df=results,
        agents=["PPO", "RND-PPO"],
        title="Week 7 Level 1: PPO vs RND-PPO",
        filename="level1_ppo_vs_rnd_ppo_rliable.png",
        aligned_filename="level1_ppo_vs_rnd_ppo_aligned.csv",
    )

    plot_rliable_curve(
        df=results,
        agents=["DQN", "RND-DQN", "PPO", "RND-PPO"],
        title="Week 7 Level 1: All Agents",
        filename="level1_all_agents_rliable.png",
        aligned_filename="level1_all_agents_aligned.csv",
    )


# ============================================================
# Config saving
# ============================================================


def save_config() -> None:
    config: Dict[str, Any] = {
        "env_name": ENV_NAME,
        "seeds": SEEDS,
        "dqn_num_frames": DQN_NUM_FRAMES,
        "ppo_total_steps": PPO_TOTAL_STEPS,
        "eval_interval": EVAL_INTERVAL,
        "eval_episodes": EVAL_EPISODES,
        "dqn_buffer_capacity": DQN_BUFFER_CAPACITY,
        "dqn_batch_size": DQN_BATCH_SIZE,
        "dqn_lr": DQN_LR,
        "dqn_gamma": DQN_GAMMA,
        "dqn_epsilon_start": DQN_EPSILON_START,
        "dqn_epsilon_final": DQN_EPSILON_FINAL,
        "dqn_epsilon_decay": DQN_EPSILON_DECAY,
        "dqn_target_update_freq": DQN_TARGET_UPDATE_FREQ,
        "ppo_lr_actor": PPO_LR_ACTOR,
        "ppo_lr_critic": PPO_LR_CRITIC,
        "ppo_gamma": PPO_GAMMA,
        "ppo_gae_lambda": PPO_GAE_LAMBDA,
        "ppo_clip_eps": PPO_CLIP_EPS,
        "ppo_epochs": PPO_EPOCHS,
        "ppo_batch_size": PPO_BATCH_SIZE,
        "ppo_ent_coef": PPO_ENT_COEF,
        "ppo_vf_coef": PPO_VF_COEF,
        "ppo_hidden_size": PPO_HIDDEN_SIZE,
        "rnd_hidden_size": RND_HIDDEN_SIZE,
        "rnd_lr": RND_LR,
        "rnd_combined_lr": RND_COMBINED_LR,
        "rnd_update_freq_dqn": RND_UPDATE_FREQ,
        "rnd_update_freq_ppo": RND_PPO_UPDATE_FREQ,
        "rnd_n_layers": RND_N_LAYERS,
        "rnd_reward_weight": RND_REWARD_WEIGHT,
        "rnd_update_proportion": RND_UPDATE_PROPORTION,
        "rnd_int_coef": RND_INT_COEF,
        "rnd_ext_coef": RND_EXT_COEF,
        "rnd_int_gamma": RND_INT_GAMMA,
        "rnd_obs_norm_init_episodes": RND_OBS_NORM_INIT_EPISODES,
        "result_dir": str(RESULT_DIR),
        "figure_dir": str(FIGURE_DIR),
    }

    path = RESULT_DIR / "level1_config.txt"

    with open(path, "w", encoding="utf-8") as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")

    print(f"Saved config: {path}")


# ============================================================
# Main
# ============================================================

# def main() -> None:
#     all_results: List[pd.DataFrame] = []

#     runners = [
#         ("DQN", run_dqn, "dqn"),
#         ("RND-DQN", run_rnd_dqn, "rnd_dqn"),
#         ("PPO", run_ppo, "ppo"),
#         ("RND-PPO", run_rnd_ppo, "rnd_ppo"),
#     ]

#     for seed in SEEDS:
#         for agent_name, runner, filename_prefix in runners:
#             print("=" * 80)
#             print(f"Running {agent_name} | seed={seed}")

#             df = runner(seed)
#             all_results.append(df)

#             seed_path = RESULT_DIR / f"{filename_prefix}_seed_{seed}.csv"
#             df.to_csv(seed_path, index=False)

#             print(f"Saved CSV: {seed_path}")

#     results = pd.concat(all_results, ignore_index=True)

#     all_csv_path = RESULT_DIR / "level1_all_results.csv"
#     results.to_csv(all_csv_path, index=False)
#     print(f"Saved all results: {all_csv_path}")

#     save_config()
#     plot_all_figures(results)

#     print("=" * 80)
#     print(f"All CSV files saved in: {RESULT_DIR}")
#     print(f"All figures saved in: {FIGURE_DIR}")


def main() -> None:
    results = pd.read_csv(RESULT_DIR / "level1_all_results.csv")
    plot_all_figures(results)

    print("=" * 80)
    print(f"All figures saved in: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
