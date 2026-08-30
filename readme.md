# Transformers
TP1 - 73.69 Large Language Models - 2026

## Project Structure

```text
main.py                     train every experiment in a config file
replot.py                   redraw every figure from saved results, no retrain

config/     config.py       repo-root-relative paths + config loading
            config.json     base hyperparameters and the experiments[] list
            *.json          one file per ablation (text vs. feature tokens, ...)
dataset/    supermarket_products.csv
            preprocess_dataset.py   title split, drops, split-by-query, id encoding
            product_dataset.py      DataFrame -> tensors -> DataLoaders
            print_processed_data.py the split report writer
model/      feature_tokenizer.py    a row -> [CLS] + title + numeric + categorical
            encoding_block.py       the TransformerEncoder stack
            encoder_only_model.py   tokenizer -> encoder -> CLS -> MLP head
            checkpoint.py           save/load trained weights (opt-in)
            positional_encoding.py  tokenizer.py
metrics/    metrics.py      precision / recall / fall-out from confusion counts
            run_results.py  what a run writes to disk, and how to read it back
plots/      plot_theme.py   the deck palette: colors, fonts, line weights
            threshold_curves.py     curves by sweeping every distinct score
            pr_auc.py  roc_auc.py   the two axis choices
            train_vs_val_error.py   the loss curve
            experiment_plots.py     the plotting path main.py and replot.py share
scripts/    eda_columns.py   eda_dataset.py   count_bought_by_title_tag.py
baselines/  bert_model.py  visualize_trained_model.py   (side comparison)
output/     every generated artifact; the only directory git ignores
docs/       comentarios_para_el_equipo/
```

Source and generated files never share a directory: anything a run writes goes
under `output/` (experiment runs, figures, EDA, reports), and `output/` is what
`.gitignore` covers.


## Setup
Requirements may include:
- Python 3.10+
- PyTorch or relevant ML framework
- pandas
- numpy
- scikit-learn
- matplotlib / seaborn for analysis and reporting

Create and activate a virtual environment:

```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Windows (CMD)
python -m venv .venv
.venv\Scripts\activate.bat

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

Install the project dependencies from the requirements file:

```bash
pip install -r requirements.txt
```

## Usage
From the project root, activate your virtual environment and run the training entry point:

```bash
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
python main.py

# Windows (CMD)
.\.venv\Scripts\activate.bat
python main.py

# macOS / Linux
source .venv/bin/activate
python main.py
```

If you are using the Python launcher on Windows, this is also valid:

```bash
py main.py
```

The script expects to be launched from the repository root so that relative paths such as `config/config.json` and `dataset/supermarket_products.csv` resolve correctly.

Configuration is loaded from:

```text
config/config.json
```

Optional experiment and training controls are available via CLI arguments, for example:

```bash
python main.py --experiment base
python main.py --epochs 1 --no-plots
python main.py --experiment base --save-weights
```

## Running Experiments
Experiment definitions are stored in `config/config.json` under the `experiments` array. Each experiment has a unique `name` and an `overrides` object for model or training settings.

Run all configured experiments:

```bash
python main.py
```

Run one experiment by name:

```bash
python main.py --experiment small_d64_l2
python main.py --experiment medium_d96_l2
python main.py --experiment small_d64_l3
```

For a quick smoke test, override the epoch count and optionally skip plots:

```bash
python main.py --experiment small_d64_l2 --epochs 1 --no-plots
```

Training plots are saved under `output/experiments/<experiment_name>/`, and metrics for all selected experiments are written to `output/experiments/summary.csv`.

### Saved results and replotting

Every experiment also writes what it produced next to its figures, so no figure ever needs a retrain:

| file | contents |
| --- | --- |
| `output/experiments/<name>/run.json` | merged config, per-epoch train/val loss and AUCs, test metrics, summary row |
| `output/experiments/<name>/test_predictions.csv` | one row per test product, `label,probability` |

The ROC and PR curves are integrated over every distinct score, so keeping the scores keeps every curve and threshold reproducible. `replot.py` reads those files back and redraws (it never imports torch):

```bash
python replot.py                                       # every saved run, plus the all-configs figures
python replot.py medium_d96_l2_baseline medium_d128_l2 # only these
python replot.py --config config/text_vs_feature_tokens.json --suffix text_vs_feature_tokens
```

`--suffix` names the combined figures `roc_auc_all_configs_<suffix>.jpg` / `pr_auc_all_configs_<suffix>.jpg`, so one config file's comparison does not overwrite another's.

All figures — training, curves and EDA — take their colors, fonts and line weights from `plots/plot_theme.py`, the same palette as the slide deck. Change a hex there and `python replot.py` restyles the whole deck in seconds. Axis titles are off by default (the title lives on the slide); set `SHOW_AXES_TITLES = True` in `plots/plot_theme.py` to get them back while iterating locally.

### Saved weights

Weights are **not** saved by default. Ask for them per run:

```bash
python main.py --experiment medium_d96_l2_baseline --save-weights
```

That adds `output/experiments/<name>/model.pt` — the state early stopping restored, i.e. the weights the reported test metrics came from, together with the `cardinalities`, merged `config` and `categorical_id_mapping` needed to rebuild the model:

### Feature tokens vs. plain text

`config/text_vs_feature_tokens.json` asks whether the encoder does better reading a row as feature tokens (one token per column, FT-Transformer style) or as a serialized `name: value | name: value` string handed to the text tokenizer alone. Architecture, optimizer and splits are identical across its three experiments, so only the representation differs:

| experiment | how a row reaches the encoder |
| --- | --- |
| `feature_tokens` | control: `product_name` as text, plus 2 numeric and 5 categorical tokens |
| `all_text_matched_columns` | the same 8 columns, serialized as text, no numeric or categorical tokens |
| `all_text_every_column` | every column left after `drop_columns` (minus the label and `query_id`), serialized as text |

```bash
python main.py --config config/text_vs_feature_tokens.json
```

The combined `output/experiments/roc_auc_all_configs.jpg` then holds exactly those three curves. `all_text_every_column` also sees columns the control never does, so read it against `all_text_matched_columns` to tell "more columns" apart from "text instead of feature tokens".

`text_column` in any config accepts a single column name, a list of column names, or `"all"` (everything `drop_columns` left, minus `text_exclude_columns`, which defaults to `bought` and `query_id`). Set `numeric_columns` and `categorical_columns` to `[]` to remove the tabular tokens entirely, and size `max_title_len` to the longest serialized row so nothing is truncated.
