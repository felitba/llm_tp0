"""The step-1 "selection" slide: two panels, four arms, deck palette.

    python scripts/slide_seleccion.py --config config/01_entrada.json [--arms a,b,c,d]

Left: validation loss per epoch for a handful of arms, each with its minimum
marked. Right: PR-AUC on test for the same arms, reported as the mean of the
per-epoch predictions (epochs 10..50), drawn over the min..max band of the
single-epoch values so the reader sees what a lucky or unlucky checkpoint would
have claimed. Four arms, not twelve: the all-configs figures are for the
appendix; this one has to be read from the back of the room.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.config import resolve_path  # noqa: E402
from config.experiments import seed_list  # noqa: E402
from metrics.run_results import experiments_dir, load_run, output_dir_from_config, run_dir  # noqa: E402
from plots.plot_theme import apply_theme  # noqa: E402
from plots.pr_auc import pr_auc_score  # noqa: E402

BLUE, ORANGE, GREEN, GREY, INK, LIGHT = "#3B5BFD", "#E69F00", "#009E73", "#9CA3AF", "#1F2937", "#E5E7EB"
DEFAULT_ARMS = "s1_02_etiqueta_nombre,s1_04_5col_tokens,s1_05_5col_texto,s1_11_12col_texto"
COLORS = [GREY, BLUE, ORANGE, GREEN]
FIRST_EPOCH = 10
BASELINE = 0.679


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	parser.add_argument("--config", required=True)
	parser.add_argument("--arms", default=DEFAULT_ARMS, help="comma-separated experiment names, in draw order")
	args = parser.parse_args()
	with resolve_path(args.config).open(encoding="utf-8") as file:
		config = json.load(file)
	output_dir_from_config(config)

	arms = []
	for name in args.arms.split(","):
		# With a seeds batch the curves come from the first seed's run.
		candidates = [name.strip()] + [f"{name.strip()}_seed{seed}" for seed in seed_list(config)]
		run = next(load_run(c) for c in candidates if (run_dir(c) / "run.json").exists())
		stored = run.epoch_predictions
		labels = stored["test_labels"].astype(int)
		probs = stored["test_probs"].astype(float)
		epochs = [int(m["epoch"]) for m in run.epoch_metrics]
		val_loss = np.array([m["val_loss"] for m in run.epoch_metrics])
		per_epoch = np.array([pr_auc_score(labels, p) for p in probs])
		arms.append({
			"label": run.name.replace("s1_", ""),
			"epochs": epochs, "val_loss": val_loss,
			"band": (per_epoch[FIRST_EPOCH - 1:].min(), per_epoch[FIRST_EPOCH - 1:].max()),
			"ensemble": pr_auc_score(labels, probs[FIRST_EPOCH - 1:].mean(axis=0)),
			"at_min_loss": per_epoch[val_loss.argmin()],
		})

	apply_theme()
	figure, (left, right) = plt.subplots(1, 2, figsize=(12.5, 4.4), gridspec_kw={"width_ratios": [1.25, 1]})

	for arm, color in zip(arms, COLORS):
		left.plot(arm["epochs"], arm["val_loss"], color=color, linewidth=2, label=arm["label"])
		best = int(arm["val_loss"].argmin())
		left.plot(arm["epochs"][best], arm["val_loss"][best], marker="o", color=color,
		          markersize=7, markeredgecolor="white", markeredgewidth=1.2)
	left.set_xlabel("época", color=INK)
	left.set_ylabel("pérdida de validación (BCE)", color=INK)
	left.grid(False)
	for side in ("top", "right"):
		left.spines[side].set_visible(False)
	left.spines["left"].set_color(GREY); left.spines["bottom"].set_color(GREY)
	left.legend(loc="upper left", frameon=False, fontsize=9)
	left.text(0.98, 0.04, "punto = mínima val_loss", transform=left.transAxes, ha="right",
	          fontsize=8.5, color=GREY)

	positions = np.arange(len(arms))[::-1]
	for y, arm, color in zip(positions, arms, COLORS):
		lo, hi = arm["band"]
		right.plot([lo, hi], [y, y], color=LIGHT, linewidth=10, solid_capstyle="butt", zorder=1)
		right.plot(arm["at_min_loss"], y, marker="|", color=GREY, markersize=16, markeredgewidth=1.6, zorder=2)
		right.plot(arm["ensemble"], y, marker="D", color=color, markersize=9, zorder=3,
		           markeredgecolor="white", markeredgewidth=1.0)
		right.text(hi + 0.006, y, f"{arm['ensemble']:.3f}", va="center", fontsize=11,
		           family="monospace", color=INK)
	right.axvline(BASELINE, color=GREY, linestyle=(0, (4, 3)), linewidth=1)
	right.text(BASELINE - 0.003, len(arms) - 0.45, "baseline 0.679", ha="right", fontsize=8.5, color=GREY)
	right.set_yticks(positions)
	right.set_yticklabels([arm["label"] for arm in arms], fontsize=10, color=INK)
	right.set_xlabel("PR-AUC en test  ·  ◆ promedio de predicciones (ép. 10–50)  ·  banda = rango por época  ·  | = mínima val_loss",
	                 fontsize=8.5, color=INK)
	right.set_xlim(0.62, 0.80)
	right.set_ylim(-0.7, len(arms) - 0.3)
	right.grid(False)
	for side in ("top", "right", "left"):
		right.spines[side].set_visible(False)
	right.spines["bottom"].set_color(GREY)
	right.tick_params(axis="y", length=0)

	figure.tight_layout(w_pad=2.5)
	out = experiments_dir() / "slides"
	out.mkdir(parents=True, exist_ok=True)
	path = out / "X1b_seleccion.png"
	figure.savefig(path, dpi=200, bbox_inches="tight")
	print(path)


if __name__ == "__main__":
	main()
