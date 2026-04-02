import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_PACKAGE_ROOT = PROJECT_ROOT / "coverage-gridworld"
if str(LOCAL_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCAL_PACKAGE_ROOT))

import coverage_gridworld  # noqa: F401 - registers the env IDs with Gymnasium
from coverage_gridworld import custom


def str2bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a PPO agent on the Coverage Gridworld environment with Stable Baselines3."
    )
    parser.add_argument("--env-id", default="standard", help="Gymnasium environment ID to train on.")
    parser.add_argument(
        "--observation-version",
        type=int,
        default=custom.ACTIVE_OBSERVATION,
        help="Observation version to activate from custom.py. This selects both observation_spaceX and observationX.",
    )
    parser.add_argument(
        "--reward-version",
        type=int,
        default=custom.ACTIVE_REWARD,
        help="Reward version to activate from custom.py.",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=300_000,
        help="Total number of environment steps used for training.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=4,
        help="Number of parallel environments used during training.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="PPO learning rate.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=1024,
        help="Rollout length collected by each environment before an update.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Mini-batch size for PPO updates.",
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda.")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy bonus coefficient.")
    parser.add_argument("--clip-range", type=float, default=0.2, help="PPO clip range.")
    parser.add_argument("--n-epochs", type=int, default=10, help="Number of PPO epochs per update.")
    parser.add_argument(
        "--frame-stack",
        type=int,
        default=4,
        help="Number of observations to stack together. Helps local observations retain short-term memory.",
    )
    parser.add_argument(
        "--normalize-observations",
        type=str2bool,
        default=True,
        help="Whether to normalize observations with VecNormalize (true/false).",
    )
    parser.add_argument(
        "--normalize-rewards",
        type=str2bool,
        default=True,
        help="Whether to normalize rewards with VecNormalize during training (true/false).",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=50_000,
        help="Run evaluation every N timesteps. Set to 0 to disable periodic evaluation.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="Number of episodes to run during each evaluation pass.",
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=50_000,
        help="Save a checkpoint every N timesteps.",
    )
    parser.add_argument(
        "--save-dir",
        default="trained_agents",
        help="Directory where checkpoints, logs, and the final model are stored.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional folder name for this run. Defaults to a timestamped PPO name.",
    )
    parser.add_argument(
        "--progress-bar",
        action="store_true",
        help="Show the SB3 progress bar. This may require extra optional dependencies.",
    )
    parser.add_argument(
        "--list-versions",
        action="store_true",
        help="Print the available observation/reward versions from custom.py and exit.",
    )
    return parser.parse_args()


def configure_custom_versions(args):
    if (
        not hasattr(custom, f"observation_space{args.observation_version}")
        or not hasattr(custom, f"observation{args.observation_version}")
    ):
        raise ValueError(
            f"Observation version {args.observation_version} is not implemented in custom.py."
        )

    if args.reward_version != 0 and not hasattr(custom, f"reward{args.reward_version}"):
        raise ValueError(f"Reward version {args.reward_version} is not implemented in custom.py.")

    custom.ACTIVE_OBSERVATION_SPACE = args.observation_version
    custom.ACTIVE_OBSERVATION = args.observation_version
    custom.ACTIVE_REWARD = args.reward_version


def available_versions():
    observations = [
        version
        for version in range(10)
        if version == 0
        or (
            hasattr(custom, f"observation_space{version}")
            and hasattr(custom, f"observation{version}")
        )
    ]
    rewards = [version for version in range(10) if version == 0 or hasattr(custom, f"reward{version}")]
    return observations, rewards


def make_run_dir(save_dir: str, env_id: str, run_name: str | None) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    resolved_name = run_name or f"ppo_{env_id}_{timestamp}"
    run_dir = Path(save_dir) / resolved_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_run_metadata(run_dir: Path, args):
    metadata = vars(args).copy()
    metadata["active_observation_space"] = custom.ACTIVE_OBSERVATION_SPACE
    metadata["active_observation"] = custom.ACTIVE_OBSERVATION
    metadata["active_reward"] = custom.ACTIVE_REWARD

    with (run_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def build_vec_env(make_vec_env, Monitor, args, training: bool):
    from stable_baselines3.common.vec_env import VecFrameStack, VecNormalize

    env_kwargs = {"render_mode": None, "predefined_map_list": None}
    n_envs = args.num_envs if training else 1
    seed = args.seed if training else args.seed + 10_000

    vec_env = make_vec_env(
        args.env_id,
        n_envs=n_envs,
        seed=seed,
        env_kwargs=env_kwargs,
        wrapper_class=Monitor,
    )

    if args.frame_stack > 1:
        vec_env = VecFrameStack(vec_env, n_stack=args.frame_stack)

    if args.normalize_observations or args.normalize_rewards:
        vec_env = VecNormalize(
            vec_env,
            training=training,
            norm_obs=args.normalize_observations,
            norm_reward=args.normalize_rewards and training,
        )

    return vec_env


class CoverageEvalCallback:
    def __init__(self, model, train_env, eval_env, run_dir: Path, eval_freq: int, eval_episodes: int):
        from stable_baselines3.common.callbacks import BaseCallback

        class _InnerCallback(BaseCallback):
            def __init__(self, outer):
                super().__init__()
                self.outer = outer

            def _on_step(self) -> bool:
                if self.outer.eval_freq <= 0 or self.num_timesteps % self.outer.eval_freq != 0:
                    return True
                self.outer.evaluate()
                return True

        self.callback = _InnerCallback(self)
        self.model = model
        self.train_env = train_env
        self.eval_env = eval_env
        self.run_dir = run_dir
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.best_mean_coverage = float("-inf")

    def evaluate(self):
        from stable_baselines3.common.vec_env import VecNormalize, sync_envs_normalization

        if isinstance(self.train_env, VecNormalize) and isinstance(self.eval_env, VecNormalize):
            sync_envs_normalization(self.train_env, self.eval_env)

        rewards = []
        coverages = []
        completions = 0
        deaths = 0

        for episode in range(self.eval_episodes):
            obs = self.eval_env.reset()
            done = False
            episode_reward = 0.0
            final_info = {}

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, dones, infos = self.eval_env.step(action)
                episode_reward += float(reward[0])
                done = bool(dones[0])
                if infos:
                    final_info = infos[0]

            coverable_cells = int(final_info.get("coverable_cells", 0))
            total_covered_cells = int(final_info.get("total_covered_cells", 0))
            game_over = bool(final_info.get("game_over", False))
            completed = coverable_cells > 0 and total_covered_cells == coverable_cells
            coverage = 0.0
            if coverable_cells:
                coverage = 100.0 * total_covered_cells / coverable_cells

            rewards.append(episode_reward)
            coverages.append(coverage)
            completions += int(completed)
            deaths += int(game_over)

        mean_reward = sum(rewards) / len(rewards)
        mean_coverage = sum(coverages) / len(coverages)
        completion_rate = completions / len(coverages)
        death_rate = deaths / len(coverages)

        self.callback.logger.record("eval/mean_reward", mean_reward)
        self.callback.logger.record("eval/mean_coverage", mean_coverage)
        self.callback.logger.record("eval/completion_rate", completion_rate)
        self.callback.logger.record("eval/death_rate", death_rate)

        print(
            f"[eval] steps={self.callback.num_timesteps} "
            f"mean_reward={mean_reward:.2f} "
            f"mean_coverage={mean_coverage:.1f}% "
            f"completion_rate={completion_rate:.2%} "
            f"death_rate={death_rate:.2%}"
        )

        if mean_coverage > self.best_mean_coverage:
            self.best_mean_coverage = mean_coverage
            best_model_path = self.run_dir / "best_model"
            self.model.save(str(best_model_path))
            if isinstance(self.train_env, VecNormalize):
                self.train_env.save(str(self.run_dir / "best_vecnormalize.pkl"))
            print(f"Saved new best model to: {best_model_path}.zip")


def main():
    args = parse_args()
    if args.list_versions:
        available_observations, available_rewards = available_versions()
        print(f"Available observation versions: {available_observations}")
        print(f"Available reward versions: {available_rewards}")
        return

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import VecNormalize

    configure_custom_versions(args)
    print(
        "Configured custom.py settings:",
        f"observation_space={custom.ACTIVE_OBSERVATION_SPACE},",
        f"observation={custom.ACTIVE_OBSERVATION},",
        f"reward={custom.ACTIVE_REWARD}",
    )
    run_dir = make_run_dir(args.save_dir, args.env_id, args.run_name)
    tb_dir = run_dir / "tensorboard"
    checkpoints_dir = run_dir / "checkpoints"

    for directory in (tb_dir, checkpoints_dir):
        directory.mkdir(parents=True, exist_ok=True)

    save_run_metadata(run_dir, args)

    vec_env = build_vec_env(make_vec_env, Monitor, args, training=True)
    eval_env = build_vec_env(make_vec_env, Monitor, args, training=False) if args.eval_freq > 0 else None

    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // args.num_envs, 1),
        save_path=str(checkpoints_dir),
        name_prefix="ppo_model",
    )

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        seed=args.seed,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        clip_range=args.clip_range,
        n_epochs=args.n_epochs,
        tensorboard_log=str(tb_dir),
        policy_kwargs={"net_arch": [256, 256]},
    )

    callbacks = [checkpoint_callback]
    if eval_env is not None:
        coverage_eval = CoverageEvalCallback(
            model=model,
            train_env=vec_env,
            eval_env=eval_env,
            run_dir=run_dir,
            eval_freq=args.eval_freq,
            eval_episodes=args.eval_episodes,
        )
        callbacks.append(coverage_eval.callback)

    print(f"Training PPO on '{args.env_id}' with {args.num_envs} parallel environments.")
    print(f"Run artifacts will be saved to: {run_dir}")
    print(
        "Training wrappers:",
        f"frame_stack={args.frame_stack},",
        f"normalize_observations={args.normalize_observations},",
        f"normalize_rewards={args.normalize_rewards}",
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=CallbackList(callbacks),
        progress_bar=args.progress_bar,
    )

    final_model_path = run_dir / "final_model"
    model.save(str(final_model_path))
    print(f"Final model saved to: {final_model_path}.zip")

    if isinstance(vec_env, VecNormalize):
        vecnormalize_path = run_dir / "vecnormalize.pkl"
        vec_env.save(str(vecnormalize_path))
        print(f"Saved VecNormalize statistics to: {vecnormalize_path}")

    vec_env.close()
    if eval_env is not None:
        eval_env.close()


if __name__ == "__main__":
    main()
