from typing import Any, Dict, List, Tuple

import argparse
import os
import random

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rl_exercises.week_6.networks import Policy, ValueNetwork
from rliable import library as rly
from torch.distributions import Categorical


def parse_args():
    parser = argparse.ArgumentParser(description="Run L2 PPO experiments.")

    parser.add_argument("--env-name", type=str, default="LunarLander-v3")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])

    parser.add_argument("--total-steps", type=int, default=200_000)
    parser.add_argument("--eval-interval", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=5)

    parser.add_argument("--lr-actor", type=float, default=5e-4)
    parser.add_argument("--lr-critic", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--hidden-size", type=int, default=128)

    parser.add_argument("--grad-clip-norm", type=float, default=0.5)

    parser.add_argument("--result-dir", type=str, default="results_l2")
    parser.add_argument("--plot-dir", type=str, default="plots_l2")

    parser.add_argument("--l1-result-dir", type=str, default="results_l1")
    parser.add_argument("--ac-baseline", type=str, default="gae")

    parser.add_argument("--ci-reps", type=int, default=2000)

    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-plot", action="store_true")

    return parser.parse_args()


def set_seed(env: gym.Env, seed: int) -> None:
    env.reset(seed=seed)

    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)

    if hasattr(env.observation_space, "seed"):
        env.observation_space.seed(seed)

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class SimplePPO:
    def __init__(
        self,
        env: gym.Env,
        lr_actor: float,
        lr_critic: float,
        gamma: float,
        gae_lambda: float,
        clip_eps: float,
        epochs: int,
        batch_size: int,
        ent_coef: float,
        vf_coef: float,
        hidden_size: int,
        use_adv_norm: bool,
        use_grad_clip: bool,
        grad_clip_norm: float,
    ):
        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef

        self.use_adv_norm = use_adv_norm
        self.use_grad_clip = use_grad_clip
        self.grad_clip_norm = grad_clip_norm

        self.policy = Policy(env.observation_space, env.action_space, hidden_size)
        self.value_fn = ValueNetwork(env.observation_space, hidden_size)

        self.optimizer = torch.optim.Adam(
            [
                {"params": self.policy.parameters(), "lr": lr_actor},
                {"params": self.value_fn.parameters(), "lr": lr_critic},
            ]
        )

    def predict(self, state: np.ndarray) -> Tuple[int, torch.Tensor]:
        state_t = torch.from_numpy(state).float()
        probs = self.policy(state_t).squeeze(0)
        dist = Categorical(probs)
        action = dist.sample()
        return int(action.item()), dist.log_prob(action)

    def predict_eval(self, state: np.ndarray) -> int:
        state_t = torch.from_numpy(state).float()

        with torch.no_grad():
            probs = self.policy(state_t).squeeze(0)

        return int(torch.argmax(probs).item())

    def compute_gae(
        self,
        rewards: List[float],
        values: torch.Tensor,
        next_values: torch.Tensor,
        dones: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rewards_t = torch.tensor(rewards, dtype=torch.float32)

        deltas = rewards_t + self.gamma * next_values * (1.0 - dones) - values

        advantages = torch.zeros_like(rewards_t)
        gae = 0.0

        for t in reversed(range(len(rewards_t))):
            gae = deltas[t] + self.gamma * self.gae_lambda * (1.0 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values

        # Enhancement 1: advantage normalization stabilizes policy updates.
        if self.use_adv_norm:
            advantages = (advantages - advantages.mean()) / (
                advantages.std(unbiased=False) + 1e-8
            )

        return advantages.detach(), returns.detach()

    def update(self, trajectory: List[Any]) -> Tuple[float, float, float]:
        states = torch.stack([torch.from_numpy(t[0]).float() for t in trajectory])
        actions = torch.tensor([t[1] for t in trajectory])
        old_logps = torch.stack([t[2] for t in trajectory]).detach()
        rewards = [t[3] for t in trajectory]
        dones = torch.tensor([t[4] for t in trajectory], dtype=torch.float32)
        next_states = torch.stack([torch.from_numpy(t[5]).float() for t in trajectory])

        with torch.no_grad():
            values = self.value_fn(states)
            next_values = self.value_fn(next_states)

        advantages, returns = self.compute_gae(rewards, values, next_values, dones)

        dataset = torch.utils.data.TensorDataset(
            states,
            actions,
            old_logps,
            advantages,
            returns,
        )

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        last_policy_loss = 0.0
        last_value_loss = 0.0
        last_entropy_loss = 0.0

        for _ in range(self.epochs):
            for b_states, b_actions, b_oldlogp, b_adv, b_ret in loader:
                probs = self.policy(b_states)
                dist = Categorical(probs)

                new_logp = dist.log_prob(b_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_logp - b_oldlogp)

                unclipped = ratio * b_adv
                clipped = (
                    torch.clamp(
                        ratio,
                        1.0 - self.clip_eps,
                        1.0 + self.clip_eps,
                    )
                    * b_adv
                )

                policy_loss = -torch.min(unclipped, clipped).mean()

                value_pred = self.value_fn(b_states)
                value_loss = F.mse_loss(value_pred, b_ret)

                entropy_loss = -entropy

                loss = (
                    policy_loss
                    + self.vf_coef * value_loss
                    + self.ent_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()

                # Enhancement 2: gradient clipping avoids overly large updates.
                if self.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(
                        list(self.policy.parameters())
                        + list(self.value_fn.parameters()),
                        max_norm=self.grad_clip_norm,
                    )

                self.optimizer.step()

                last_policy_loss = float(policy_loss.item())
                last_value_loss = float(value_loss.item())
                last_entropy_loss = float(entropy_loss.item())

        return last_policy_loss, last_value_loss, last_entropy_loss

    def evaluate(self, eval_env: gym.Env, eval_episodes: int) -> Tuple[float, float]:
        returns = []

        for _ in range(eval_episodes):
            state, _ = eval_env.reset()
            done = False
            total_return = 0.0

            while not done:
                action = self.predict_eval(state)
                state, reward, term, trunc, _ = eval_env.step(action)
                done = term or trunc
                total_return += reward

            returns.append(total_return)

        return float(np.mean(returns)), float(np.std(returns))


def train_one_run(
    args,
    variant: str,
    seed: int,
    use_adv_norm: bool,
    use_grad_clip: bool,
) -> pd.DataFrame:
    env = gym.make(args.env_name)
    eval_env = gym.make(args.env_name)

    set_seed(env, seed)
    set_seed(eval_env, seed + 10_000)

    agent = SimplePPO(
        env=env,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_eps=args.clip_eps,
        epochs=args.epochs,
        batch_size=args.batch_size,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        hidden_size=args.hidden_size,
        use_adv_norm=use_adv_norm,
        use_grad_clip=use_grad_clip,
        grad_clip_norm=args.grad_clip_norm,
    )

    step_count = 0
    results: List[Dict[str, Any]] = []

    while step_count < args.total_steps:
        state, _ = env.reset()
        done = False
        trajectory = []

        while not done and step_count < args.total_steps:
            action, logp = agent.predict(state)
            next_state, reward, term, trunc, _ = env.step(action)

            done = term or trunc

            trajectory.append(
                (
                    state,
                    action,
                    logp,
                    float(reward),
                    float(done),
                    next_state,
                )
            )

            state = next_state
            step_count += 1

            if step_count % args.eval_interval == 0:
                mean_return, std_return = agent.evaluate(
                    eval_env,
                    eval_episodes=args.eval_episodes,
                )

                results.append(
                    {
                        "env_name": args.env_name,
                        "algorithm": variant,
                        "seed": seed,
                        "step": step_count,
                        "mean_return": mean_return,
                        "std_return": std_return,
                    }
                )

                print(
                    f"[Eval] env={args.env_name} "
                    f"algo={variant:12s} "
                    f"seed={seed} "
                    f"step={step_count:7d} "
                    f"return={mean_return:8.2f} ± {std_return:7.2f}"
                )

        policy_loss, value_loss, entropy_loss = agent.update(trajectory)

        print(
            f"[Train] algo={variant:12s} "
            f"seed={seed} "
            f"step={step_count:7d} "
            f"policy_loss={policy_loss:8.4f} "
            f"value_loss={value_loss:8.4f} "
            f"entropy_loss={entropy_loss:8.4f}"
        )

    env.close()
    eval_env.close()

    return pd.DataFrame(results)


def run_all_experiments(args) -> None:
    os.makedirs(args.result_dir, exist_ok=True)

    variants = {
        "ppo_vanilla": {
            "use_adv_norm": False,
            "use_grad_clip": False,
        },
        "ppo_enhanced": {
            "use_adv_norm": True,
            "use_grad_clip": True,
        },
    }

    all_results = []

    for variant, settings in variants.items():
        for seed in args.seeds:
            print("=" * 80)
            print(f"Running {variant}, seed={seed}")
            print("=" * 80)

            df = train_one_run(
                args=args,
                variant=variant,
                seed=seed,
                use_adv_norm=settings["use_adv_norm"],
                use_grad_clip=settings["use_grad_clip"],
            )

            csv_path = os.path.join(
                args.result_dir,
                f"{args.env_name}_{variant}_seed_{seed}.csv",
            )

            df.to_csv(csv_path, index=False)
            all_results.append(df)

            print(f"Saved: {csv_path}")

    merged = pd.concat(all_results, ignore_index=True)
    merged_path = os.path.join(args.result_dir, f"{args.env_name}_ppo_all_results.csv")
    merged.to_csv(merged_path, index=False)

    print(f"Saved merged results: {merged_path}")


def load_curves_from_l2(
    args, algorithms: List[str]
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    score_dict = {}
    steps = None

    for algo in algorithms:
        seed_curves = []

        for seed in args.seeds:
            path = os.path.join(
                args.result_dir,
                f"{args.env_name}_{algo}_seed_{seed}.csv",
            )

            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing file: {path}")

            df = pd.read_csv(path)

            current_steps = df["step"].to_numpy()

            if steps is None:
                steps = current_steps
            elif not np.array_equal(steps, current_steps):
                raise ValueError(f"Step mismatch in {path}")

            seed_curves.append(df["mean_return"].to_numpy())

        score_dict[algo] = np.array(seed_curves)

    return score_dict, steps


def plot_rliable(
    score_dict: Dict[str, np.ndarray],
    steps: np.ndarray,
    title: str,
    ylabel: str,
    output_path: str,
    ci_reps: int,
) -> None:
    mean_scores, mean_cis = rly.get_interval_estimates(
        score_dict,
        lambda scores: np.mean(scores, axis=0),
        reps=ci_reps,
    )

    plt.figure(figsize=(7, 4))

    for name in score_dict.keys():
        mean = mean_scores[name]
        lower, upper = mean_cis[name]

        plt.plot(steps, mean, label=name)
        plt.fill_between(steps, lower, upper, alpha=0.2)

    plt.xlabel("Environment steps")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()

    print(f"Saved plot: {output_path}")


def plot_ppo_variants(args) -> None:
    os.makedirs(args.plot_dir, exist_ok=True)

    score_dict, steps = load_curves_from_l2(
        args,
        algorithms=["ppo_vanilla", "ppo_enhanced"],
    )

    output_path = os.path.join(
        args.plot_dir,
        f"{args.env_name}_ppo_vanilla_vs_enhanced.png",
    )

    plot_rliable(
        score_dict=score_dict,
        steps=steps,
        title=f"PPO variants on {args.env_name}",
        ylabel="Average return",
        output_path=output_path,
        ci_reps=args.ci_reps,
    )


def plot_ppo_vs_actor_critic(args) -> None:
    os.makedirs(args.plot_dir, exist_ok=True)

    score_dict = {}

    ppo_score_dict, ppo_steps = load_curves_from_l2(
        args,
        algorithms=["ppo_enhanced"],
    )

    score_dict["ppo_enhanced"] = ppo_score_dict["ppo_enhanced"]

    ac_curves = []

    for seed in args.seeds:
        ac_path = os.path.join(
            args.l1_result_dir,
            f"{args.env_name}_{args.ac_baseline}_seed_{seed}.csv",
        )

        if not os.path.exists(ac_path):
            print(f"Skip PPO vs Actor-Critic plot because missing L1 file: {ac_path}")
            return

        ac_df = pd.read_csv(ac_path)

        # Keep only steps that also exist in PPO results.
        ac_df = ac_df[ac_df["step"].isin(ppo_steps)]

        if len(ac_df) != len(ppo_steps):
            print("Skip PPO vs Actor-Critic plot because L1 and L2 steps do not match.")
            print("Use the same eval_interval and total_steps for both experiments.")
            return

        ac_curves.append(ac_df["mean_return"].to_numpy())

    score_dict[f"actor_critic_{args.ac_baseline}"] = np.array(ac_curves)

    output_path = os.path.join(
        args.plot_dir,
        f"{args.env_name}_ppo_vs_actor_critic.png",
    )

    plot_rliable(
        score_dict=score_dict,
        steps=ppo_steps,
        title=f"PPO vs Actor-Critic on {args.env_name}",
        ylabel="Average return",
        output_path=output_path,
        ci_reps=args.ci_reps,
    )


def main() -> None:
    args = parse_args()

    if not args.skip_train:
        run_all_experiments(args)

    if not args.skip_plot:
        plot_ppo_variants(args)
        plot_ppo_vs_actor_critic(args)


if __name__ == "__main__":
    main()
