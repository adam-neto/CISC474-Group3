import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import gymnasium as gym

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_PACKAGE_ROOT = PROJECT_ROOT / "coverage-gridworld"
if str(LOCAL_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCAL_PACKAGE_ROOT))

import coverage_gridworld  # noqa: F401 - registers the env IDs with Gymnasium
from coverage_gridworld import custom


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
        "--eval-env-id",
        default=None,
        help="Environment ID used during evaluation. Defaults to the training environment.",
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
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=25_000,
        help="Evaluate every N timesteps.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes per evaluation run.",
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


def make_run_dir(save_dir: str, env_id: str, run_name: str | None) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    resolved_name = run_name or f"ppo_{env_id}_{timestamp}"
    run_dir = Path(save_dir) / resolved_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def build_env_kwargs():
    return {
        "render_mode": None,
        "predefined_map_list": None,
    }


def save_run_metadata(run_dir: Path, args):
    metadata = vars(args).copy()
    metadata["active_observation_space"] = custom.ACTIVE_OBSERVATION_SPACE
    metadata["active_observation"] = custom.ACTIVE_OBSERVATION
    metadata["active_reward"] = custom.ACTIVE_REWARD

    with (run_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def main():
    args = parse_args()
    if args.list_versions:
        available_observations = [
            version
            for version in range(10)
            if version == 0
            or (
                hasattr(custom, f"observation_space{version}")
                and hasattr(custom, f"observation{version}")
            )
        ]
        available_rewards = [
            version for version in range(10) if version == 0 or hasattr(custom, f"reward{version}")
        ]
        print(f"Available observation versions: {available_observations}")
        print(f"Available reward versions: {available_rewards}")
        return

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.monitor import Monitor

    configure_custom_versions(args)
    print(
        "Configured custom.py settings:",
        f"observation_space={custom.ACTIVE_OBSERVATION_SPACE},",
        f"observation={custom.ACTIVE_OBSERVATION},",
        f"reward={custom.ACTIVE_REWARD}",
    )
    eval_env_id = args.eval_env_id or args.env_id
    run_dir = make_run_dir(args.save_dir, args.env_id, args.run_name)
    tb_dir = run_dir / "tensorboard"
    checkpoints_dir = run_dir / "checkpoints"
    best_model_dir = run_dir / "best_model"
    eval_logs_dir = run_dir / "eval_logs"

    for directory in (tb_dir, checkpoints_dir, best_model_dir, eval_logs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    save_run_metadata(run_dir, args)

    env_kwargs = build_env_kwargs()
    vec_env = make_vec_env(
        args.env_id,
        n_envs=args.num_envs,
        seed=args.seed,
        env_kwargs=env_kwargs,
        wrapper_class=Monitor,
    )

    eval_env = Monitor(gym.make(eval_env_id, **env_kwargs))

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(best_model_dir),
        log_path=str(eval_logs_dir),
        eval_freq=max(args.eval_freq // args.num_envs, 1),
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
    )

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
        tensorboard_log=str(tb_dir),
        policy_kwargs={"net_arch": [256, 256]},
    )

    print(f"Training PPO on '{args.env_id}' with {args.num_envs} parallel environments.")
    print(f"Run artifacts will be saved to: {run_dir}")

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=args.progress_bar,
    )

    final_model_path = run_dir / "final_model"
    model.save(str(final_model_path))
    print(f"Final model saved to: {final_model_path}.zip")

    eval_env.close()
    vec_env.close()


if __name__ == "__main__":
    main()
