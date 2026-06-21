import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ============================================================
# Paths
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(BASE_DIR, "results", "level2")
FIGURES_DIR = os.path.join(BASE_DIR, "figures", "level2")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

RAW_CSV = os.path.join(RESULTS_DIR, "l2_dqn_vs_reinforce_runs.csv")

ALPHA = 0.05
N_PERMUTATIONS = 10_000


# ============================================================
# AUC and test
# ============================================================


def compute_auc(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    # Compatible with both old and new NumPy
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

    for (algo, seed), group in df.groupby(["algorithm", "seed"]):
        group = group.sort_values("step")

        auc = integrate(
            group["eval_return"].to_numpy(),
            group["step"].to_numpy(),
        )

        rows.append(
            {
                "algorithm": algo,
                "seed": seed,
                "auc": float(auc),
                "final_return": float(group["eval_return"].iloc[-1]),
            }
        )

    return pd.DataFrame(rows)


def paired_permutation_test(x, y, n_permutations=10_000, seed=0):
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
# Plotting
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
    if not os.path.exists(RAW_CSV):
        raise FileNotFoundError(
            f"Cannot find raw result file:\n{RAW_CSV}\n"
            "Make sure the RL training script has already produced this file."
        )

    print("=" * 80)
    print("Analyze Level 2 only")
    print(f"Reading: {RAW_CSV}")
    print("=" * 80)

    df = pd.read_csv(RAW_CSV)

    required_columns = {"algorithm", "seed", "step", "eval_return"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # ------------------------------------------------------------
    # Compute AUC
    # ------------------------------------------------------------

    auc_df = compute_auc(df)

    auc_path = os.path.join(RESULTS_DIR, "l2_auc_per_seed.csv")
    auc_df.to_csv(auc_path, index=False)

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------

    learning_curve_path = os.path.join(FIGURES_DIR, "l2_learning_curves.png")
    auc_boxplot_path = os.path.join(FIGURES_DIR, "l2_auc_boxplot.png")

    plot_learning_curves(df, learning_curve_path)
    plot_auc_boxplot(auc_df, auc_boxplot_path)

    # ------------------------------------------------------------
    # Statistical test
    # ------------------------------------------------------------

    pivot = auc_df.pivot(index="seed", columns="algorithm", values="auc").sort_index()

    if "DQN" not in pivot.columns or "REINFORCE" not in pivot.columns:
        raise ValueError("CSV must contain both algorithms: DQN and REINFORCE")

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
        "n_seeds": len(pivot),
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

    print("\nSaved:")
    print(f"AUC CSV: {auc_path}")
    print(f"Learning curve plot: {learning_curve_path}")
    print(f"AUC boxplot: {auc_boxplot_path}")
    print(f"Test result: {test_path}")

    print("\nTest result:")
    for key, value in test_result.items():
        print(f"{key}: {value}")

    print("\nDone.")


if __name__ == "__main__":
    main()
