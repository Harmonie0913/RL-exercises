from typing import Any, Dict, List

import argparse
import os

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rl_exercises.week_6.actor_critic import ActorCriticAgent
from rliable import library as rly


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run L1 Actor-Critic baseline experiments."
    )

    parser.add_argument(
        "--env-name",
        type=str,
        default="CartPole-v1",
        help="Gymnasium environment name, e.g. CartPole-v1 or LunarLander-v3.",
    )

    parser.add_argument(
        "--baselines",
        nargs="+",
        default=["none", "avg", "value", "gae"],
        choices=["none", "avg", "value", "gae"],
        help="Baseline types to run.",
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2],
        help="Random seeds to run.",
    )

    parser.add_argument(
        "--total-steps",
        type=int,
        default=100_000,
        help="Total environment steps per run.",
    )

    parser.add_argument(
        "--eval-interval",
        type=int,
        default=5_000,
        help="Evaluate every N environment steps.",
    )

    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes.",
    )

    parser.add_argument(
        "--lr-actor",
        type=float,
        default=5e-4,
        help="Learning rate for actor.",
    )

    parser.add_argument(
        "--lr-critic",
        type=float,
        default=1e-3,
        help="Learning rate for critic.",
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor.",
    )

    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95,
        help="GAE lambda.",
    )

    parser.add_argument(
        "--hidden-size",
        type=int,
        default=128,
        help="Hidden layer size.",
    )

    parser.add_argument(
        "--baseline-decay",
        type=float,
        default=0.9,
        help="Decay for running-average baseline.",
    )

    parser.add_argument(
        "--result-dir",
        type=str,
        default="results_l1",
        help="Directory for CSV results.",
    )

    parser.add_argument(
        "--plot-dir",
        type=str,
        default="plots_l1",
        help="Directory for plots.",
    )

    parser.add_argument(
        "--ci-reps",
        type=int,
        default=2000,
        help="Number of bootstrap repetitions for confidence intervals.",
    )

    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training and only plot from existing CSV files.",
    )

    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="Skip plotting and only run training.",
    )

    return parser.parse_args()


def train_one_run(
    env_name: str,
    baseline_type: str,
    seed: int,
    total_steps: int,
    eval_interval: int,
    eval_episodes: int,
    lr_actor: float,
    lr_critic: float,
    gamma: float,
    gae_lambda: float,
    hidden_size: int,
    baseline_decay: float,
) -> pd.DataFrame:
    env = gym.make(env_name)

    agent = ActorCriticAgent(
        env=env,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        gamma=gamma,
        gae_lambda=gae_lambda,
        seed=seed,
        hidden_size=hidden_size,
        baseline_type=baseline_type,
        baseline_decay=baseline_decay,
    )

    eval_env = gym.make(env_name)

    step_count = 0
    results: List[Dict[str, Any]] = []

    while step_count < total_steps:
        state, _ = env.reset(seed=seed)
        done = False
        trajectory = []

        while not done and step_count < total_steps:
            action, logp = agent.predict_action(state)
            next_state, reward, term, trunc, _ = env.step(action)

            done = term or trunc

            trajectory.append(
                (
                    state,
                    action,
                    float(reward),
                    next_state,
                    done,
                    logp,
                )
            )

            state = next_state
            step_count += 1

            if step_count % eval_interval == 0:
                mean_return, std_return = agent.evaluate(
                    eval_env,
                    num_episodes=eval_episodes,
                )

                results.append(
                    {
                        "env_name": env_name,
                        "baseline": baseline_type,
                        "seed": seed,
                        "step": step_count,
                        "mean_return": mean_return,
                        "std_return": std_return,
                    }
                )

                print(
                    f"[Eval] env={env_name} "
                    f"baseline={baseline_type:5s} "
                    f"seed={seed} "
                    f"step={step_count:7d} "
                    f"return={mean_return:8.2f} ± {std_return:7.2f}"
                )

        agent.update_agent(trajectory)

    env.close()
    eval_env.close()

    return pd.DataFrame(results)


def run_all_experiments(args) -> None:
    os.makedirs(args.result_dir, exist_ok=True)

    all_results = []

    for baseline in args.baselines:
        for seed in args.seeds:
            print("=" * 80)
            print(f"Running env={args.env_name}, baseline={baseline}, seed={seed}")
            print("=" * 80)

            df = train_one_run(
                env_name=args.env_name,
                baseline_type=baseline,
                seed=seed,
                total_steps=args.total_steps,
                eval_interval=args.eval_interval,
                eval_episodes=args.eval_episodes,
                lr_actor=args.lr_actor,
                lr_critic=args.lr_critic,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                hidden_size=args.hidden_size,
                baseline_decay=args.baseline_decay,
            )

            csv_path = os.path.join(
                args.result_dir,
                f"{args.env_name}_{baseline}_seed_{seed}.csv",
            )

            df.to_csv(csv_path, index=False)
            all_results.append(df)

            print(f"Saved: {csv_path}")

    if all_results:
        merged = pd.concat(all_results, ignore_index=True)
        merged_path = os.path.join(args.result_dir, f"{args.env_name}_all_results.csv")
        merged.to_csv(merged_path, index=False)

        print(f"Saved merged results: {merged_path}")


def plot_with_rliable(args) -> None:
    os.makedirs(args.plot_dir, exist_ok=True)

    score_dict = {}
    steps = None

    for baseline in args.baselines:
        seed_curves = []

        for seed in args.seeds:
            csv_path = os.path.join(
                args.result_dir,
                f"{args.env_name}_{baseline}_seed_{seed}.csv",
            )

            if not os.path.exists(csv_path):
                raise FileNotFoundError(
                    f"Missing result file: {csv_path}. "
                    f"Run training first or remove this baseline/seed from args."
                )

            df = pd.read_csv(csv_path)

            if steps is None:
                steps = df["step"].to_numpy()
            else:
                current_steps = df["step"].to_numpy()
                if not np.array_equal(steps, current_steps):
                    raise ValueError(
                        f"Step mismatch in {csv_path}. "
                        f"Make sure all runs use the same eval_interval and total_steps."
                    )

            seed_curves.append(df["mean_return"].to_numpy())

        score_dict[baseline] = np.array(seed_curves)

    mean_scores, mean_cis = rly.get_interval_estimates(
        score_dict,
        lambda scores: np.mean(scores, axis=0),
        reps=args.ci_reps,
    )

    plt.figure(figsize=(7, 4))

    for baseline in args.baselines:
        mean = mean_scores[baseline]
        lower, upper = mean_cis[baseline]

        plt.plot(steps, mean, label=baseline)
        plt.fill_between(steps, lower, upper, alpha=0.2)

    plt.xlabel("Environment steps")
    plt.ylabel("Average return")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plot_path = os.path.join(
        args.plot_dir,
        f"{args.env_name}_l1_baselines_rliable.png",
    )

    plt.savefig(plot_path, dpi=300)
    plt.show()

    print(f"Saved plot: {plot_path}")


def main() -> None:
    args = parse_args()

    if not args.skip_train:
        run_all_experiments(args)

    if not args.skip_plot:
        plot_with_rliable(args)


if __name__ == "__main__":
    main()
