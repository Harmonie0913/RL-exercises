import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(__file__)

ddpg_path = os.path.join(BASE_DIR, "ddpg_returns_l3.csv")
reinforce_path = os.path.join(BASE_DIR, "reinforce_continuous_results_l3.csv")

ddpg_returns = np.loadtxt(ddpg_path, delimiter=",")
reinforce_df = pd.read_csv(reinforce_path)

reinforce_returns = reinforce_df["train_return"].values


def moving_average(values, window=10):
    values = np.array(values, dtype=np.float32)

    if len(values) < window:
        return values

    return np.convolve(values, np.ones(window) / window, mode="valid")


plt.figure(figsize=(8, 5))

plt.plot(
    moving_average(ddpg_returns, 10),
    label="DDPG",
)

plt.plot(
    moving_average(reinforce_returns, 10),
    label="REINFORCE",
)

plt.xlabel("Episode")
plt.ylabel("Return")
plt.title("Comparison of DDPG and REINFORCE on Pendulum-v1")
plt.legend()
plt.grid(True)
plt.tight_layout()

save_path = os.path.join(BASE_DIR, "ddpg_vs_reinforce_l3.png")
plt.savefig(save_path, dpi=300)
plt.show()

print(f"Saved comparison plot to: {save_path}")
