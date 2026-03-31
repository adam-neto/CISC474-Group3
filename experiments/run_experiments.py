import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from pathlib import Path

from config import EVAL_MAPS


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train_sb3.py"
EVAL_SCRIPT = PROJECT_ROOT / "scripts" / "eval_sb3.py"
MODELS_DIR = PROJECT_ROOT / "experiments" / "models"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"

CHECKPOINT_PATTERN = re.compile(r"ppo_model_(\d+)_steps\.zip$")


def parse_version_list(raw_value: str):
    return [int(part.strip()) for part in raw_value.split(",") if part.strip()]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the observation/reward experiment matrix for Coverage Gridworld."
    )
    parser.add_argument("--train-env-id", default="standard", help="Environment used for training.")
    parser.add_argument("--total-timesteps", type=int, default=300_000, help="Training timesteps per run.")
    parser.add_argument("--num-envs", type=int, default=4, help="Parallel training environments.")
    parser.add_argument(
        "--n-steps",
        type=int,
        default=1024,
        help="PPO rollout length per environment. Used for training and x-axis step labels.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="PPO batch size passed through to the training script.",
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=100,
        help="Save training checkpoints every N timesteps so plots have intermediate points.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=1,
        help="Episodes per fixed-map evaluation point.",
    )
    parser.add_argument(
        "--standard-eval-episodes",
        type=int,
        default=1,
        help="Episodes per evaluation point on the standard map.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Number of parallel evaluation jobs to run after each training job.",
    )
    parser.add_argument(
        "--observations",
        default="1,2,3",
        help="Comma-separated observation versions to include.",
    )
    parser.add_argument(
        "--rewards",
        default="1,2,3",
        help="Comma-separated reward versions to include.",
    )
    parser.add_argument(
        "--maps",
        default=",".join(EVAL_MAPS),
        help="Comma-separated evaluation maps to include.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training and only evaluate existing runs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    return parser.parse_args()


def run_command(command, dry_run: bool):
    print(" ".join(command))
    if not dry_run:
        subprocess.run(command, check=True, cwd=PROJECT_ROOT)


def run_dir_for(obs_version: int, reward_version: int) -> Path:
    return MODELS_DIR / f"obs{obs_version}_rew{reward_version}"


def results_file_for(obs_version: int, reward_version: int) -> Path:
    return RESULTS_DIR / f"obs{obs_version}_rew{reward_version}.json"


def build_train_command(args, obs_version: int, reward_version: int):
    return [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--env-id",
        args.train_env_id,
        "--observation-version",
        str(obs_version),
        "--reward-version",
        str(reward_version),
        "--total-timesteps",
        str(args.total_timesteps),
        "--num-envs",
        str(args.num_envs),
        "--n-steps",
        str(args.n_steps),
        "--batch-size",
        str(args.batch_size),
        "--checkpoint-freq",
        str(args.checkpoint_freq),
        "--seed",
        str(args.seed),
        "--save-dir",
        str(MODELS_DIR),
        "--run-name",
        f"obs{obs_version}_rew{reward_version}",
    ]


def actual_total_timesteps(args):
    rollout_size = args.n_steps * args.num_envs
    if rollout_size <= 0:
        return args.total_timesteps
    num_rollouts = (args.total_timesteps + rollout_size - 1) // rollout_size
    return num_rollouts * rollout_size


def collect_models(run_dir: Path, args):
    models = []
    checkpoints_dir = run_dir / "checkpoints"
    if checkpoints_dir.exists():
        for checkpoint in sorted(checkpoints_dir.glob("*.zip")):
            match = CHECKPOINT_PATTERN.match(checkpoint.name)
            if match:
                models.append(
                    {
                        "label": f"checkpoint_{match.group(1)}",
                        "step": int(match.group(1)),
                        "path": checkpoint,
                    }
                )

    final_model = run_dir / "final_model.zip"
    if final_model.exists():
        final_step = actual_total_timesteps(args)
        models.append(
            {
                "label": f"final_{final_step}",
                "step": final_step,
                "path": final_model,
            }
        )

    return models


def build_eval_command(args, model_info, obs_version: int, reward_version: int, map_id: str):
    return [
        sys.executable,
        str(EVAL_SCRIPT),
        str(model_info["path"]),
        "--env-id",
        map_id,
        "--observation-version",
        str(obs_version),
        "--reward-version",
        str(reward_version),
        "--episodes",
        str(episodes_for_map(args, map_id)),
        "--seed",
        str(args.seed),
    ]


def episodes_for_map(args, map_id: str):
    if map_id == "standard":
        return args.standard_eval_episodes
    return args.eval_episodes


def run_eval_jobs(eval_jobs, dry_run: bool, jobs: int):
    for job in eval_jobs:
        print(" ".join(job["command"]))

    if dry_run:
        return []

    def run_eval_job(job):
        temp_dir = RESULTS_DIR / ".tmp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            prefix="eval_",
            dir=temp_dir,
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)

        command = job["command"] + ["--output-json", str(temp_path)]
        try:
            subprocess.run(command, check=True, cwd=PROJECT_ROOT)
            with temp_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        finally:
            if temp_path.exists():
                temp_path.unlink()

        return {
            "label": job["model_info"]["label"],
            "step": job["model_info"]["step"],
            "map_id": job["map_id"],
            "summary": payload["summary"],
            "episodes": payload["episodes"],
            "seed": payload["seed"],
        }

    max_workers = max(1, jobs)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_eval_job, job) for job in eval_jobs]
        done, not_done = wait(futures, return_when=FIRST_EXCEPTION)

        results = [future.result() for future in done]

        if not_done:
            for future in not_done:
                future.cancel()

        return results


def write_combo_results(args, obs_version: int, reward_version: int, output_path: Path, entries):
    payload = {
        "observation_version": obs_version,
        "reward_version": reward_version,
        "train_env_id": args.train_env_id,
        "total_timesteps_requested": args.total_timesteps,
        "total_timesteps_actual": actual_total_timesteps(args),
        "checkpoint_freq": args.checkpoint_freq,
        "eval_episodes": args.eval_episodes,
        "standard_eval_episodes": args.standard_eval_episodes,
        "seed": args.seed,
        "results": sorted(entries, key=lambda item: (item["step"], item["map_id"])),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main():
    args = parse_args()
    observations = parse_version_list(args.observations)
    rewards = parse_version_list(args.rewards)
    maps = [item.strip() for item in args.maps.split(",") if item.strip()]

    for obs_version in observations:
        for reward_version in rewards:
            run_dir = run_dir_for(obs_version, reward_version)
            results_file = results_file_for(obs_version, reward_version)
            run_dir.parent.mkdir(parents=True, exist_ok=True)

            if not args.skip_train:
                run_command(build_train_command(args, obs_version, reward_version), args.dry_run)

            eval_jobs = []
            for model_info in collect_models(run_dir, args):
                for map_id in maps:
                    eval_jobs.append(
                        {
                            "model_info": model_info,
                            "map_id": map_id,
                            "command": build_eval_command(
                                args,
                                model_info,
                                obs_version,
                                reward_version,
                                map_id,
                            ),
                        }
                    )

            entries = run_eval_jobs(
                eval_jobs,
                args.dry_run,
                args.jobs,
            )

            if not args.dry_run:
                write_combo_results(
                    args,
                    obs_version,
                    reward_version,
                    results_file,
                    entries,
                )


if __name__ == "__main__":
    main()
