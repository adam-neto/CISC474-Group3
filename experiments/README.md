# Experiments

This directory contains a simple workflow for Task 4 experiments.

Goal:

- train one agent for each observation/reward combination
- evaluate saved checkpoints on multiple maps
- plot one panel per observation/reward combination
- organize the final report figures as a 3x3 grid
  - rows = observation spaces
  - columns = reward functions

Layout:

- `config.py`: default experiment matrix and plot colors
- `run_experiments.py`: trains runs and aggregates checkpoint evaluations
- `plot_results.py`: builds the 3x3 coverage plot grid
- `models/`: saved agents for experiment runs
- `results/`: one JSON summary per observation/reward combination, such as `obs1_rew2.json`
- `plots/`: generated figures

Notes:

- The experiment scripts use the same `--observation-version` and `--reward-version` flags as the main SB3 scripts.
- Coverage percentage is the primary metric for plotting.
- The default training environment is `standard`, which means each observation/reward pair is trained once on random maps and then evaluated on each map listed in `--maps`.
- The script default for `--checkpoint-freq` is `100`, but larger values such as `50000` are more practical for long report-scale runs.
- The current plot workflow reads the aggregate `obsX_rewY.json` files in `results/`.
- Generated files under `models/`, `results/`, and `plots/` are ignored by Git.

Typical workflow:

```bash
python3 experiments/run_experiments.py --observations 1 --rewards 1 --total-timesteps 100000
python3 experiments/plot_results.py
```

Report-scale workflow:

```bash
python3 experiments/run_experiments.py --observations 1,2,3 --rewards 1,2,3 --total-timesteps 1000000 --checkpoint-freq 50000
python3 experiments/plot_results.py
```

If you only want to test your own combination after changing one observation space or reward function, run just that
pair:

```bash
python3 experiments/run_experiments.py --observations 2 --rewards 3 --total-timesteps 4096
python3 experiments/plot_results.py
```

Run the full 3x3 matrix:

```bash
python3 experiments/run_experiments.py --observations 1,2,3 --rewards 1,2,3 --total-timesteps 300000
python3 experiments/plot_results.py
```
