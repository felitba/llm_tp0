import argparse
import itertools
import copy
import torch

from config.config import load_config
from main import train_one_experiment, write_experiment_summary
from plots.experiment_plots import plot_runs_combined


def merged_config(base_config: dict, overrides: dict) -> dict:
    """Return a copy of base_config with one grid combination applied."""
    config = copy.deepcopy(base_config)
    config.update(overrides)
    return config


def main():
    parser = argparse.ArgumentParser(description="Run a grid search over model parameters.")
    parser.add_argument(
        "--config",
        help="Base config file. Defaults to config/text_vs_feature_tokens.json",
        default="config/feature_tokens.json"
    )
    args = parser.parse_args()

    # 1. Load the base configuration (ignoring the manual experiments array)
    base_config = load_config(args.config)

    # 2. Define your Grid here
    param_grid = {
        "d_model": [96, 128],
        "n_heads": [4,6],
        "num_layers": [2,4],
        "dim_feedforward": [256, 512],
    }

    # 3. Generate all combinations
    keys = list(param_grid.keys())
    combinations = list(itertools.product(*(param_grid[k] for k in keys)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    runs = []

    for combo in combinations:
        overrides = dict(zip(keys, combo))

        # Transformer architecture constraint: d_model must be divisible by n_heads
        if overrides["d_model"] % overrides["n_heads"] != 0:
            print(
                f"Skipping {overrides}: d_model ({overrides['d_model']}) is not divisible by n_heads ({overrides['n_heads']}).")
            continue

        # Create a unique name for the output folder (e.g., grid_d_model96_lr0.0003_n_heads4)
        name_parts = [f"{k}{v}" for k, v in overrides.items()]
        exp_name = "grid_" + "_".join(name_parts)

        config = merged_config(base_config, overrides)

        print(f"\n=== Starting Grid Run: {exp_name} ===")
        # Call the exact same training function main.py uses
        run_result = train_one_experiment(
            name=exp_name,
            config=config,
            device=device,
            save_plots=True,
            save_weights=True
        )
        runs.append(run_result)

    # 4. Generate the final combined summary and comparison plots
    summary_path = write_experiment_summary([run.summary for run in runs])
    print(f"\nGrid search summary written to: {summary_path}")

    for combined_path in plot_runs_combined(runs, suffix="grid_search"):
        print(f"Combined plot written to: {combined_path}")


if __name__ == "__main__":
    main()