import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_PACKAGE_ROOT = PROJECT_ROOT / "coverage-gridworld"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(LOCAL_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCAL_PACKAGE_ROOT))

import coverage_gridworld  # noqa: F401 - registers env IDs with Gymnasium
from train_sb3 import configure_custom_versions, make_run_dir, save_run_metadata

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a PPO agent across multiple env IDs in sequence."
    )
    parser.add_argument(
        "--stage",
        action="append",
        required=True,
        help="Format: name:timesteps:env_id",
    )
    parser.add_argument(
        "--env-id",
        default="curriculum",
        help="Folder naming only; each stage uses its own env_id from --stage.",
    )
    parser.add_argument("--observation-version", type=int, default=0)
    parser.add_argument("--reward-version", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--checkpoint-freq", type=int, default=50_000)
    parser.add_argument("--save-dir", default="trained_agents")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--progress-bar", action="store_true")
    return parser.parse_args()


def parse_stages(stage_args):
    stages = []
    for stage_str in stage_args:
        parts = stage_str.split(":", maxsplit=2)
        if len(parts) != 3:
            raise ValueError(
                f"Invalid --stage '{stage_str}'. Expected format: name:timesteps:env_id"
            )

        name, timesteps_str, env_id = parts

        try:
            timesteps = int(timesteps_str)
        except ValueError as exc:
            raise ValueError(
                f"Invalid timesteps in stage '{stage_str}': '{timesteps_str}' is not an integer."
            ) from exc

        if timesteps <= 0:
            raise ValueError(
                f"Invalid timesteps in stage '{stage_str}': timesteps must be > 0."
            )

        stages.append(
            {
                "name": name,
                "timesteps": timesteps,
                "env_id": env_id,
            }
        )

    return stages


def build_env(env_id, num_envs, seed):
    env_kwargs = {"render_mode": None, "predefined_map_list": None}
    return make_vec_env(
        env_id,
        n_envs=num_envs,
        seed=seed,
        env_kwargs=env_kwargs,
        wrapper_class=Monitor,
    )


def main():
    args = parse_args()
    stages = parse_stages(args.stage)

    configure_custom_versions(args)
    print(
        "Configured custom.py settings:",
        f"observation_space={args.observation_version},",
        f"observation={args.observation_version},",
        f"reward={args.reward_version}",
    )

    run_dir = make_run_dir(args.save_dir, args.env_id, args.run_name)
    tb_dir = run_dir / "tensorboard"
    checkpoints_dir = run_dir / "checkpoints"

    for directory in (tb_dir, checkpoints_dir):
        directory.mkdir(parents=True, exist_ok=True)

    save_run_metadata(run_dir, args)

    model = None
    current_env = None

    print(f"Run artifacts will be saved to: {run_dir}")

    for i, stage in enumerate(stages):
        print(f"\n=== Stage {i + 1}/{len(stages)}: {stage['name']} ({stage['env_id']}) ===")

        new_env = build_env(stage["env_id"], args.num_envs, args.seed + i)

        if model is None:
            model = PPO(
                "MlpPolicy",
                new_env,
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
        else:
            model.set_env(new_env)

        checkpoint_callback = CheckpointCallback(
            save_freq=max(args.checkpoint_freq // args.num_envs, 1),
            save_path=str(checkpoints_dir),
            name_prefix="ppo_model",
        )

        print(
            f"Training PPO on '{stage['env_id']}' with {args.num_envs} parallel environments "
            f"for {stage['timesteps']} timesteps."
        )

        model.learn(
            total_timesteps=stage["timesteps"],
            callback=checkpoint_callback,
            progress_bar=args.progress_bar,
            reset_num_timesteps=(i == 0),
        )

        if current_env is not None:
            current_env.close()
        current_env = new_env

    final_model_path = run_dir / "final_model"
    model.save(str(final_model_path))
    print(f"Final model saved to: {final_model_path}.zip")

    if current_env is not None:
        current_env.close()


if __name__ == "__main__":
    main()