import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rliable import library as rly
from rliable import metrics, plot_utils

RESULT_DIR = r"outputs\2026-05-11\00-02-22\l2_results"
SEEDS = [0, 1, 2, 3, 4]


def load_seed_curves():
    dfs = []

    for seed in SEEDS:
        path = os.path.join(RESULT_DIR, f"seed_{seed}.csv")
        df = pd.read_csv(path)
        df = df.rename(columns={"mean_reward": f"seed_{seed}"})
        dfs.append(df)

    # Use the first seed as the base curve.
    merged = dfs[0]

    # Add the other seed curves.
    # Since different seeds log rewards at slightly different frame numbers,
    # merge_asof matches each row with the nearest available frame.
    for df in dfs[1:]:
        merged = pd.merge_asof(
            merged.sort_values("frame"),
            df.sort_values("frame"),
            on="frame",
            direction="nearest",
        )

    frames = merged["frame"].to_numpy()
    scores = merged[[f"seed_{s}" for s in SEEDS]].to_numpy().T

    return frames, scores


def main():
    os.makedirs("l2_plots", exist_ok=True)

    frames, scores = load_seed_curves()

    BASELINE = 20.0
    TARGET = 250.0

    scores = (scores - BASELINE) / (TARGET - BASELINE)
    scores = np.clip(scores, 0.0, 1.0)

    # shape: num_seeds x num_tasks x num_frames
    scores = scores[:, None, :]

    score_dict = {"DQN": scores}

    # 1. RLiable IQM training curve
    iqm_func = lambda x: np.array(
        [metrics.aggregate_iqm(x[..., t]) for t in range(x.shape[-1])]
    )

    iqm_scores, iqm_cis = rly.get_interval_estimates(
        score_dict,
        iqm_func,
        reps=2000,
    )

    fig, ax = plt.subplots()
    plot_utils.plot_sample_efficiency_curve(
        frames,
        iqm_scores,
        iqm_cis,
        algorithms=["DQN"],
        xlabel="Number of frames",
        ylabel="IQM reward",
        ax=ax,
    )
    plt.savefig("l2_plots/rliable_iqm_curve.png", bbox_inches="tight")
    plt.close()

    # 2. Mean curve with seed variation
    mean_rewards = scores[:, 0, :].mean(axis=0)
    std_rewards = scores[:, 0, :].std(axis=0)

    plt.figure()
    plt.plot(frames, mean_rewards, label="Mean")
    plt.fill_between(
        frames,
        mean_rewards - std_rewards,
        mean_rewards + std_rewards,
        alpha=0.2,
        label="±1 std",
    )
    plt.xlabel("Number of frames")
    plt.ylabel("Mean reward")
    plt.title("DQN mean curve across 5 seeds")
    plt.grid(True)
    plt.legend()
    plt.savefig("l2_plots/plain_mean_curve.png", bbox_inches="tight")
    plt.close()

    # 3. Aggregate metrics at final frame
    final_score_dict = {"DQN": scores[:, :, -1]}

    aggregate_func = lambda x: np.array(
        [
            metrics.aggregate_median(x),
            metrics.aggregate_iqm(x),
            metrics.aggregate_mean(x),
            metrics.aggregate_optimality_gap(x),
        ]
    )

    aggregate_scores, aggregate_cis = rly.get_interval_estimates(
        final_score_dict,
        aggregate_func,
        reps=2000,
    )

    fig, axes = plot_utils.plot_interval_estimates(
        aggregate_scores,
        aggregate_cis,
        metric_names=["Median", "IQM", "Mean", "Optimality Gap"],
        algorithms=["DQN"],
    )

    # Make the figure taller
    fig.set_size_inches(12, 2.5)

    # Force remove all existing x-axis labels from every subplot
    for ax in np.ravel(axes):
        ax.set_xlabel("")

    # Give enough bottom space
    fig.subplots_adjust(bottom=0.28)

    plt.savefig(
        "l2_plots/rliable_aggregate_metrics.png",
        bbox_inches="tight",
        pad_inches=0.2,
    )
    plt.close()

    print("Saved plots to l2_plots/")


if __name__ == "__main__":
    main()
