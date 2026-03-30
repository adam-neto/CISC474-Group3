import argparse
import sys
import time
from pathlib import Path

import gymnasium as gym

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_PACKAGE_ROOT = PROJECT_ROOT / "coverage-gridworld"
if str(LOCAL_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCAL_PACKAGE_ROOT))

import coverage_gridworld  # noqa: F401 - registers the env IDs with Gymnasium
from stable_baselines3 import PPO


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
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    return parser.parse_args()


def main():
    args = parse_args()
    deterministic = not args.stochastic
    render_mode = "human" if args.render else None

    env = gym.make(args.env_id, render_mode=render_mode, predefined_map_list=None)
    model = PPO.load(args.model_path)

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

        print(
            f"Episode {episode}: reward={episode_reward:.2f}, "
            f"coverage={coverage:.1f}%, "
            f"steps_left={final_info.get('steps_remaining', 'n/a')}, "
            f"game_over={final_info.get('game_over', 'n/a')}"
        )

    env.close()


if __name__ == "__main__":
    main()
