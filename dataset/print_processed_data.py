from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd


def write_processed_data_report(
	splits: Dict[str, pd.DataFrame],
	output_path: Path | None = None,
) -> Path:
	"""Write a text report for processed dataset splits and return the report path."""
	report_lines: list[str] = []

	for key, value in splits.items():
		report_lines.append("=" * 100)
		report_lines.append(f"{key.upper()} | rows={len(value):,} cols={value.shape[1]}")
		if key == "train":
			mapping = value.attrs.get("one_hot_encoding_mapping", {})
			report_lines.append("One-hot encoding mapping (vector index -> category):")
			if mapping:
				for column, column_mapping in mapping.items():
					report_lines.append(f"  {column}: {column_mapping}")
			else:
				report_lines.append("  No one-hot encoded columns")
		if "bought" in value.columns:
			bought_ratio = value["bought"].astype(bool).mean()
			report_lines.append(
				f"bought=True ratio (bought=True rows / total rows): {bought_ratio:.2%}"
			)
		report_lines.append("-" * 100)
		preview = value.head(50)
		first_columns = [
			column for column in ("product_name", "title_tag") if column in preview.columns
		]
		remaining_columns = [
			column for column in preview.columns if column not in first_columns
		]
		report_lines.append(preview[first_columns + remaining_columns].to_string(index=False))
		report_lines.append("")

	if output_path is None:
		output_path = Path(__file__).parent / "output" / "preprocess_dataset_report.txt"

	output_path.parent.mkdir(parents=True, exist_ok=True)
	output_path.write_text("\n".join(report_lines), encoding="utf-8")
	return output_path

