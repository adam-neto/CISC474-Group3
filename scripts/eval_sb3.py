import argparse
import json
import sys
import time
from pathlib import Path

import gymnasium as gym

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


def parse_map_list(raw_value: str | None) -> list[str]:
    if not raw_value:
        return []
    return [part.strip() for part in raw_value.split(",") if part.strip()]


def resolve_predefined_maps(map_ids: list[str]):
    if not map_ids:
        return None

    predefined_maps = []
    for map_id in map_ids:
        spec = gym.spec(map_id)
        predefined_map = spec.kwargs.get("predefined_map")
        if predefined_map is None:
            raise ValueError(
                f"Environment '{map_id}' does not define a fixed predefined_map and cannot be used in --map-list."
            )
        predefined_maps.append(predefined_map)

    return predefined_maps


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a saved PPO agent on the Coverage Gridworld environment."
    )
    parser.add_argument(
        "model_path",
        help="Path to the saved SB3 model (.zip file or path without the extension).",
    )
    parser.add_argument(
        "--env-id",
        default="sneaky_enemies",
        help="Gymnasium environment ID used for evaluation.",
    )
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
        "--map-list",
        default=None,
        help="Optional comma-separated list of registered fixed-map env IDs to cycle through during evaluation.",
    )
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to run.")
    parser.add_argument(
        "--frame-stack",
        type=int,
        default=4,
        help="Number of observations to stack together. Must match training if frame stacking was used.",
    )
    parser.add_argument(
        "--vecnormalize-path",
        default=None,
        help="Optional path to VecNormalize statistics (.pkl). If omitted, the script tries to infer it.",
    )
    parser.add_argument(
        "--normalize-observations",
        type=str2bool,
        default=True,
        help="Whether to wrap the evaluation env with observation normalization when no stats file is found.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic actions instead of deterministic actions.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the environment with pygame. Leave this off for headless evaluation.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Optional sleep in seconds between rendered steps.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write a JSON summary of the evaluation.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
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


def infer_vecnormalize_path(model_path: str) -> Path | None:
    path = Path(model_path)
    candidates = []

    if path.suffix == ".zip":
        candidates.append(path.with_name("best_vecnormalize.pkl" if path.stem == "best_model" else "vecnormalize.pkl"))
        candidates.append(path.with_name("vecnormalize.pkl"))
        candidates.append(path.with_name("best_vecnormalize.pkl"))
    else:
        candidates.append(path.parent / ("best_vecnormalize.pkl" if path.name == "best_model" else "vecnormalize.pkl"))
        candidates.append(path.parent / "vecnormalize.pkl")
        candidates.append(path.parent / "best_vecnormalize.pkl")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def main():
    args = parse_args()
    if args.list_versions:
        available_observations, available_rewards = available_versions()
        print(f"Available observation versions: {available_observations}")
        print(f"Available reward versions: {available_rewards}")
        return

    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize

    configure_custom_versions(args)
    print(
        "Configured custom.py settings:",
        f"observation_space={custom.ACTIVE_OBSERVATION_SPACE},",
        f"observation={custom.ACTIVE_OBSERVATION},",
        f"reward={custom.ACTIVE_REWARD}",
    )
    if args.map_list:
        print(f"Using rotating predefined map list: {parse_map_list(args.map_list)}")
    deterministic = not args.stochastic
    render_mode = "human" if args.render else None

    env = gym.make(
        args.env_id,
        render_mode=render_mode,
        predefined_map_list=resolve_predefined_maps(parse_map_list(args.map_list)),
    )
    env = Monitor(env)
    vec_env = DummyVecEnv([lambda: env])

    if args.frame_stack > 1:
        vec_env = VecFrameStack(vec_env, n_stack=args.frame_stack)

    resolved_vecnormalize_path = None
    if args.vecnormalize_path:
        resolved_vecnormalize_path = Path(args.vecnormalize_path)
    else:
        resolved_vecnormalize_path = infer_vecnormalize_path(args.model_path)

    if resolved_vecnormalize_path is not None and resolved_vecnormalize_path.exists():
        vec_env = VecNormalize.load(str(resolved_vecnormalize_path), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
        print(f"Loaded VecNormalize statistics from: {resolved_vecnormalize_path}")
    elif args.normalize_observations:
        vec_env = VecNormalize(vec_env, training=False, norm_obs=True, norm_reward=False)
        print("No VecNormalize statistics found; using fresh observation normalization wrapper.")

    model = PPO.load(args.model_path)
    episode_summaries = []

    for episode in range(1, args.episodes + 1):
        vec_env.seed(args.seed + episode - 1)
        obs = vec_env.reset()
        done = False
        episode_reward = 0.0
        final_info = {}

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, rewards, dones, infos = vec_env.step(action)
            episode_reward += float(rewards[0])
            final_info = infos[0]
            done = bool(dones[0])

            if args.render and args.delay > 0:
                time.sleep(args.delay)

        coverable_cells = final_info.get("coverable_cells", 0)
        total_covered_cells = final_info.get("total_covered_cells", 0)
        coverage = 0.0
        if coverable_cells:
            coverage = 100.0 * total_covered_cells / coverable_cells

        steps_left = int(final_info.get("steps_remaining", 0))
        game_over = bool(final_info.get("game_over", False))
        coverable_cells = int(coverable_cells)
        total_covered_cells = int(total_covered_cells)
        coverage = float(coverage)
        episode_reward = float(episode_reward)
        completed = coverable_cells > 0 and total_covered_cells == coverable_cells
        timed_out = steps_left == 0 and not game_over and not completed

        episode_summary = {
            "episode": episode,
            "reward": episode_reward,
            "coverage": coverage,
            "steps_left": steps_left,
            "game_over": game_over,
            "completed": completed,
            "timed_out": timed_out,
            "coverable_cells": coverable_cells,
            "total_covered_cells": total_covered_cells,
        }
        episode_summaries.append(episode_summary)

        print(
            f"Episode {episode}: reward={episode_reward:.2f}, "
            f"coverage={coverage:.1f}%, "
            f"steps_left={steps_left}, "
            f"game_over={game_over}"
        )

    vec_env.close()

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        mean_reward = sum(item["reward"] for item in episode_summaries) / len(episode_summaries)
        mean_coverage = sum(item["coverage"] for item in episode_summaries) / len(episode_summaries)
        completion_rate = sum(item["completed"] for item in episode_summaries) / len(episode_summaries)
        death_rate = sum(item["game_over"] for item in episode_summaries) / len(episode_summaries)
        timeout_rate = sum(item["timed_out"] for item in episode_summaries) / len(episode_summaries)

        payload = {
            "model_path": str(Path(args.model_path)),
            "env_id": args.env_id,
            "observation_version": args.observation_version,
            "reward_version": args.reward_version,
            "episodes": args.episodes,
            "seed": args.seed,
            "summary": {
                "mean_reward": float(mean_reward),
                "mean_coverage": float(mean_coverage),
                "completion_rate": float(completion_rate),
                "death_rate": float(death_rate),
                "timeout_rate": float(timeout_rate),
            },
            "episodes_detail": episode_summaries,
        }

        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)


if __name__ == "__main__":
    main()
