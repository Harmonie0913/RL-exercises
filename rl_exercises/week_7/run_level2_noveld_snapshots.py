"""
Level 2: NovelD-PPO behavior snapshots.

CSV results are saved to:
rl_exercises/week_7/results/level2/

Behavior snapshot figures are saved to:
rl_exercises/week_7/figures/level2/
"""

from typing import Any, List

from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rl_exercises.week_6.ppo import set_seed
from rl_exercises.week_7.noveid_ppo import NovelDPPOAgent

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
BASE_DIR = Path("rl_exercises/week_7")
RESULT_DIR = BASE_DIR / "results" / "level2"
FIGURE_DIR = BASE_DIR / "figures" / "level2"


# -----------------------------------------------------------------------------
# Experiment settings
# -----------------------------------------------------------------------------
ENV_NAME = "LunarLander-v3"
SEED = 0

TOTAL_STEPS = 100_000
EVAL_INTERVAL = 10_000
EVAL_EPISODES = 5

# Minimum three snapshots are required by Level 2.
SNAPSHOT_STEPS = [10_000, 50_000, 100_000]


# -----------------------------------------------------------------------------
# Plot helper
# -----------------------------------------------------------------------------
def save_behavior_snapshot(
    states: List[np.ndarray],
    bonuses: List[float],
    step: int,
    figure_dir: Path,
) -> None:
    """
    Save a behavior snapshot.

    For LunarLander, state[0] and state[1] roughly correspond to the lander's
    x and y position. The color represents the raw NovelD intrinsic reward.
    """
    figure_dir.mkdir(parents=True, exist_ok=True)

    states_arr = np.asarray(states)
    bonuses_arr = np.asarray(bonuses)

    if len(states_arr) == 0:
        print(f"[Snapshot] No states collected for step {step}.")
        return

    plt.figure(figsize=(7, 5))
    scatter = plt.scatter(
        states_arr[:, 0],
        states_arr[:, 1],
        c=bonuses_arr,
        s=8,
        alpha=0.7,
    )
    plt.colorbar(scatter, label="NovelD intrinsic reward")
    plt.xlabel("state[0] = x position")
    plt.ylabel("state[1] = y position")
    plt.title(f"NovelD-PPO behavior snapshot at step {step}")
    plt.tight_layout()

    out_path = figure_dir / f"noveld_snapshot_step_{step}.png"
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"[Snapshot] Saved figure: {out_path}")


def save_snapshot_csv(
    states: List[np.ndarray],
    bonuses: List[float],
    step: int,
    result_dir: Path,
) -> None:
    """Save the raw data used for the behavior snapshot."""
    result_dir.mkdir(parents=True, exist_ok=True)

    states_arr = np.asarray(states)
    bonuses_arr = np.asarray(bonuses)

    if len(states_arr) == 0:
        return

    df = pd.DataFrame(
        {
            "step_snapshot": step,
            "state_0_x": states_arr[:, 0],
            "state_1_y": states_arr[:, 1],
            "noveld_bonus_raw": bonuses_arr,
        }
    )

    out_path = result_dir / f"noveld_snapshot_step_{step}.csv"
    df.to_csv(out_path, index=False)
    print(f"[Snapshot] Saved CSV: {out_path}")


# -----------------------------------------------------------------------------
# Main experiment
# -----------------------------------------------------------------------------
def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    print("CSV results are saved to:")
    print(f"{RESULT_DIR}/")
    print()
    print("Behavior snapshot figures are saved to:")
    print(f"{FIGURE_DIR}/")
    print()

    env = gym.make(ENV_NAME)
    eval_env = gym.make(ENV_NAME)
    set_seed(env, SEED)

    agent = NovelDPPOAgent(
        env=env,
        seed=SEED,
        hidden_size=128,
        rnd_hidden_size=128,
        rnd_n_layers=2,
        noveld_alpha=1.0,
        combined_lr=1e-4,
        update_proportion=0.25,
        int_coef=1.0,
        ext_coef=2.0,
        int_gamma=0.99,
        num_iterations_obs_norm_init=50,
    )

    eval_rows = []
    step_count = 0
    next_snapshot_idx = 0

    # These buffers store states and NovelD bonuses since the previous snapshot.
    snapshot_states: List[np.ndarray] = []
    snapshot_bonuses: List[float] = []

    print("[Level 2] Start NovelD-PPO training with behavior snapshots.")

    # Same warm-up idea as in NovelDPPOAgent.train().
    agent._init_obs_normalization()

    while step_count < TOTAL_STEPS:
        state, _ = env.reset(seed=SEED)
        done = False
        trajectory: List[Any] = []

        # NovelD first-visit memory is reset at the beginning of each episode.
        agent._episode_visited = set()

        # Normalize initial state.
        agent.obs_rms.update(state[np.newaxis])
        prev_obs_norm = agent._normalize_obs(state)

        while not done and step_count < TOTAL_STEPS:
            action, logp, entropy, _, _ = agent.predict(state)
            next_state, ext_reward, term, trunc, _ = env.step(action)
            done = term or trunc

            # Observation normalization.
            agent.obs_rms.update(next_state[np.newaxis])
            next_obs_norm = agent._normalize_obs(next_state)

            # Raw NovelD intrinsic reward.
            int_reward_raw = agent.get_noveld_bonus(prev_obs_norm, next_obs_norm)

            # Normalized intrinsic reward for PPO update.
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

            # Data used for the behavior snapshot.
            snapshot_states.append(next_state.copy())
            snapshot_bonuses.append(float(int_reward_raw))

            state = next_state
            prev_obs_norm = next_obs_norm
            step_count += 1

            # Save snapshots at early / middle / late training.
            if (
                next_snapshot_idx < len(SNAPSHOT_STEPS)
                and step_count >= SNAPSHOT_STEPS[next_snapshot_idx]
            ):
                snapshot_step = SNAPSHOT_STEPS[next_snapshot_idx]
                save_behavior_snapshot(
                    snapshot_states,
                    snapshot_bonuses,
                    snapshot_step,
                    FIGURE_DIR,
                )
                save_snapshot_csv(
                    snapshot_states,
                    snapshot_bonuses,
                    snapshot_step,
                    RESULT_DIR,
                )

                # Clear buffers so each snapshot only describes the latest period.
                snapshot_states = []
                snapshot_bonuses = []
                next_snapshot_idx += 1

            # Evaluation with extrinsic reward only.
            if step_count % EVAL_INTERVAL == 0:
                mean_return, std_return = agent.evaluate(
                    eval_env, num_episodes=EVAL_EPISODES
                )
                eval_rows.append(
                    {
                        "step": step_count,
                        "mean_return": mean_return,
                        "std_return": std_return,
                        "seed": SEED,
                        "algorithm": "NovelD-PPO",
                    }
                )

                print(
                    f"[Eval] Step {step_count:6d} "
                    f"AvgReturn {mean_return:7.2f} ± {std_return:6.2f}"
                )

        # PPO + NovelD/RND update after one trajectory.
        policy_loss, value_loss, entropy_loss, rnd_loss = agent.update(trajectory)

        episode_return = sum(t[4] for t in trajectory)
        print(
            f"[Train] Step {step_count:6d} "
            f"Return {episode_return:7.2f} "
            f"Policy Loss {policy_loss:.3f} "
            f"Value Loss {value_loss:.3f} "
            f"Entropy Loss {entropy_loss:.3f} "
            f"RND Loss {rnd_loss:.3f}"
        )

    # Save evaluation CSV.
    eval_df = pd.DataFrame(eval_rows)
    eval_path = RESULT_DIR / f"noveld_ppo_eval_seed_{SEED}.csv"
    eval_df.to_csv(eval_path, index=False)
    print(f"[Results] Saved evaluation CSV: {eval_path}")

    env.close()
    eval_env.close()
    print("[Level 2] Done.")


if __name__ == "__main__":
    main()
