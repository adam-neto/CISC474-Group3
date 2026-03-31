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
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to run.")
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

    configure_custom_versions(args)
    print(
        "Configured custom.py settings:",
        f"observation_space={custom.ACTIVE_OBSERVATION_SPACE},",
        f"observation={custom.ACTIVE_OBSERVATION},",
        f"reward={custom.ACTIVE_REWARD}",
    )
    deterministic = not args.stochastic
    render_mode = "human" if args.render else None

    env = gym.make(args.env_id, render_mode=render_mode, predefined_map_list=None)
    model = PPO.load(args.model_path)
    episode_summaries = []

    for episode in range(1, args.episodes + 1):
        obs, _ = env.reset(seed=args.seed + episode - 1)
        done = False
        episode_reward = 0.0
        final_info = {}

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            final_info = info
            done = terminated or truncated

            if args.render and args.delay > 0:
                time.sleep(args.delay)

        coverable_cells = final_info.get("coverable_cells", 0)
        total_covered_cells = final_info.get("total_covered_cells", 0)
        coverage = 0.0
        if coverable_cells:
            coverage = 100.0 * total_covered_cells / coverable_cells

        steps_left = final_info.get("steps_remaining", 0)
        game_over = final_info.get("game_over", False)
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

    env.close()

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
                "mean_reward": mean_reward,
                "mean_coverage": mean_coverage,
                "completion_rate": completion_rate,
                "death_rate": death_rate,
                "timeout_rate": timeout_rate,
            },
            "episodes_detail": episode_summaries,
        }

        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)


if __name__ == "__main__":
    main()
