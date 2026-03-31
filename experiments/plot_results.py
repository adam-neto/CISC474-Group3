import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import MAP_COLORS, OBSERVATION_VERSIONS, REWARD_VERSIONS


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
PLOTS_DIR = PROJECT_ROOT / "experiments" / "plots"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot coverage curves for each observation/reward combination."
    )
    parser.add_argument(
        "--output",
        default=str(PLOTS_DIR / "coverage_grid.png"),
        help="Output image path for the 3x3 plot grid.",
    )
    return parser.parse_args()


def load_results():
    data = {}
    malformed_files = []
    for result_path in RESULTS_DIR.glob("obs*_rew*.json"):
        try:
            with result_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except json.JSONDecodeError:
            malformed_files.append(result_path)
            continue

        obs_version = payload["observation_version"]
        reward_version = payload["reward_version"]
        for item in payload.get("results", []):
            key = (obs_version, reward_version, item["map_id"])
            data.setdefault(key, []).append((item["step"], item["summary"]["mean_coverage"]))

    # Backward compatibility for older per-checkpoint result files.
    for result_path in RESULTS_DIR.glob("obs*_rew*/*.json"):
        combo_name = result_path.parent.name
        parts = combo_name.replace("obs", "").split("_rew")
        obs_version = int(parts[0])
        reward_version = int(parts[1])

        try:
            with result_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except json.JSONDecodeError:
            malformed_files.append(result_path)
            continue

        model_name, map_id = result_path.stem.split("__", 1)
        if model_name.startswith("checkpoint_"):
            step = int(model_name.split("_", 1)[1])
        elif model_name.startswith("final_"):
            step = int(model_name.split("_", 1)[1])
        else:
            continue

        key = (obs_version, reward_version, map_id)
        data.setdefault(key, []).append((step, payload["summary"]["mean_coverage"]))

    for key in data:
        data[key] = sorted(set(data[key]), key=lambda item: item[0])

    return data, malformed_files


def main():
    args = parse_args()
    data, malformed_files = load_results()
    if not data:
        raise SystemExit("No valid result files were found in experiments/results.")

    max_step = max(points[-1][0] for points in data.values() if points)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        len(OBSERVATION_VERSIONS),
        len(REWARD_VERSIONS),
        figsize=(15, 12),
        sharex=True,
        sharey=True,
    )

    for row_index, obs_version in enumerate(OBSERVATION_VERSIONS):
        for col_index, reward_version in enumerate(REWARD_VERSIONS):
            ax = axes[row_index][col_index]
            for map_id, color in MAP_COLORS.items():
                points = data.get((obs_version, reward_version, map_id), [])
                if not points:
                    continue
                x_values = [item[0] for item in points]
                y_values = [item[1] for item in points]
                ax.plot(
                    x_values,
                    y_values,
                    label=map_id,
                    color=color,
                    linewidth=2,
                    marker="o",
                    markersize=5,
                )

            ax.set_title(f"Obs {obs_version} / Rew {reward_version}")
            ax.set_ylim(0, 100)
            ax.set_xlim(0, max_step)
            ax.grid(True, alpha=0.3)

            if row_index == len(OBSERVATION_VERSIONS) - 1:
                ax.set_xlabel("Training Timesteps")
            if col_index == 0:
                ax.set_ylabel("Average Coverage (%)")

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=4)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    expected_series = {
        (obs_version, reward_version, map_id)
        for obs_version in OBSERVATION_VERSIONS
        for reward_version in REWARD_VERSIONS
        for map_id in MAP_COLORS
    }
    missing_series = sorted(expected_series - set(data.keys()))

    if malformed_files:
        print(f"Skipped {len(malformed_files)} malformed result file(s):")
        for path in malformed_files:
            print(f"  - {path}")
    if missing_series:
        print(f"Missing {len(missing_series)} expected result series:")
        for obs_version, reward_version, map_id in missing_series:
            print(f"  - obs{obs_version}_rew{reward_version} / {map_id}")
    else:
        print("All expected result series are present.")

    print(f"Saved plot grid to: {output_path}")


if __name__ == "__main__":
    main()
