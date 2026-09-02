# BTR prediction with an encoder-only Transformer

**TP1 — 73.69 Large Language Models — ITBA, 2026**

Predict **BTR (Buy Through Rate)** — will an impressed product be bought — from
`dataset/supermarket_products.csv` (10,000 impressions, 22 columns, 2,012 queries,
base BTR 0.1301), using an encoder-only Transformer whose row representation is
FT-Transformer style: **one token per feature**.

> **The single project document is [`docs/INFORME.md`](docs/INFORME.md)** — problem
> definition, EDA, assumptions, architecture, evaluation protocol, results, and what is
> still open. This README only covers how to run the code.
> The frozen model-selection protocol is [`docs/PROTOCOL.md`](docs/PROTOCOL.md).

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate                 # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> **`requirements.txt` is incomplete and UTF-16/CRLF encoded.** It is missing `tiktoken`
> (needed by `model/tokenizer.py`) and `scikit-learn` (needed by `plots/pr_auc.py` and
> `plots/roc_auc.py`), plus `transformers` and `datasets`, which only `baselines/` needs.
> Until it is regenerated:
>
> ```bash
> pip install tiktoken scikit-learn
> ```

`python` is not on the PATH outside the venv — activate it, or prefix every command with
`.venv/bin/python`. Everything expects to be launched from the repository root.

## Running experiments

The live experiment plan is six ordered config files, one per ablation step. Each one
declares its own `output_dir` and carries a `_carry_forward` note saying what to copy into
the next step.

```bash
python main.py --config config/01_entrada.json     # what the encoder reads, and how
python main.py --config config/02_transformer.json # is the attention worth anything?
python main.py --config config/03_capacidad.json   # d_model / layers / heads / FFN ratio
python main.py --config config/04_regularizacion.json  # dropout / weight decay / lr
python main.py --config config/05_modulos.json     # ReLU vs GELU, pre-LN, linear probe
python main.py --config config/06_semillas.json    # error bars on the final model
```

`main.py` reads `experiments[]` from the config file and, for each entry, deep-copies the
base config and applies its `overrides`. A top-level `"seeds": [42, 7, 1234]` multiplies
every entry into one run per seed (`<name>_seed<n>`). **An ablation is a JSON edit, not a
code change.**

```bash
python main.py --config config/01_entrada.json --experiment s1_04_5col_tokens        # its 3 seeds
python main.py --config config/01_entrada.json --experiment s1_04_5col_tokens_seed7  # just one
python main.py --epochs 1 --no-plots                 # smoke test
python main.py --config ... --save-weights           # also write model.pt (~20 MB per run)
python main.py --config ... --no-error-patterns      # skip the batch-level error report
python main.py --print-cls-btr                       # dump per-product CLS logit/prob/label
```

Steps 02 and 05 are **not runnable yet**: `use_encoder` / `pooling` (step 02) and
`norm_first` / head depth (step 05) are declared in the configs but not implemented in the
model. See `docs/INFORME.md` §9.

`config/config.json` (the default when `--config` is omitted) and
`config/text_vs_feature_tokens.json`, `config/feature_tokens*.json`,
`config/no_signal_columns.json` are **legacy**: they predate the protocol, select on
`val_pr_auc`, and their numbers are not comparable with steps 01–06.

## What a run writes

Under `<output_dir>/<name>/`, where `<output_dir>` is the config's `"output_dir"` key
(`config/01_entrada.json` → `output/01_entrada/`), falling back to `output/experiments/`:

| file | contents |
| --- | --- |
| `run.json` | merged config, per-epoch train/val loss and AUCs, the `selection` block (which epoch the test row describes and by what rule), test metrics, summary row |
| `test_predictions.csv` | one row per test product, `label,probability` |
| `epoch_predictions.npz` | validation **and** test probabilities at every epoch (<1 MB) |
| `test_error_analysis.csv` | the test rows as a person can read them, worst prediction first, ids decoded back to names |
| `loss.jpg`, `pr_auc.jpg`, `roc_auc.jpg`, … | the figures |

Per batch: `summary.csv` (one row per run), the `*_all_configs.jpg` comparison figures,
`error_patterns.txt` / `.jpg`, and — only with `reselect --final` — `resultados_finales.csv`.

The ROC and PR curves are integrated over every distinct score, so keeping the scores keeps
every curve and threshold reproducible without a forward pass. `epoch_predictions.npz` is
what makes the checkpoint rule a post-hoc decision.

## Reporting, without retraining

```bash
# compare the arms of a step on VALIDATION (test stays closed)
python scripts/reselect.py --config config/01_entrada.json
python scripts/reselect.py --config config/01_entrada.json --rule val_pr_auc   # another rule

# once, when the model is already chosen
python scripts/reselect.py --config config/01_entrada.json --apply --final

# figures
python replot.py --config config/01_entrada.json           # redraw a step from disk
python replot.py                                           # every saved run
python replot.py s1_04_5col_tokens_seed42                  # only these
python replot.py --config ... --suffix mystep --no-combined

# analysis
python scripts/error_patterns.py --config config/01_entrada.json  # where the batch fails
python scripts/epoch_band.py --config config/01_entrada.json      # appendix: PR-AUC per epoch
python scripts/step1_figures.py                                   # the step-1 slide figures
python scripts/slide_seleccion.py --config config/01_entrada.json
python scripts/attention_map.py <name> --output-dir output/01_entrada   # needs --save-weights
```

`replot.py` rebuilds every figure from what the runs wrote to disk and **never imports
torch** — which is why `metrics/run_results.py` stays torch-free and weight I/O lives in
`model/checkpoint.py`.

Figure styling — training curves, metric curves and EDA alike — lives in
`plots/plot_theme.py` (Okabe-Ito palette, chosen to survive all three kinds of colour
blindness). Change a hex there and `python replot.py` restyles the whole deck in seconds.
Axis titles are off by default because the title lives on the slide; set
`SHOW_AXES_TITLES = True` to get them back while iterating.

## EDA

```bash
python scripts/eda_columns.py    # stdlib only; output/eda/00*–12*
python scripts/eda_dataset.py    # output/eda/13*–17*
python scripts/eda_slides.py     # output/eda/slides/*.png, in the deck palette
```

`eda_columns.py` deliberately avoids pandas: `read_csv`'s default `na_values` turns the
literal string `"None"` in `allergens` — a real category meaning "no allergens" — into
`NaN`, inventing a 44.5% missing rate in a column with no missing values. It regenerates
every number quoted in §2 of `docs/INFORME.md`.

## Modules inside packages

They import `config.config`, so `python dataset/preprocess_dataset.py` fails. Use `-m`:

```bash
python -m dataset.preprocess_dataset   # writes output/preprocess_dataset_report.txt
python -m config.config                # print the resolved config
python -m model.tokenizer              # tiktoken smoke test
```

## Layout

```text
main.py                     train every experiment a config file declares
replot.py                   redraw every figure from saved results, no retrain

config/     config.py       repo-root-relative paths + config loading
            experiments.py  experiments[] x seeds -> the runs a config means
            0[1-6]_*.json   the live ablation plan, one file per step
            config.json     legacy default; *.json others are legacy too
dataset/    supermarket_products.csv
            preprocess_dataset.py   title split, drops, split-by-query, binning,
                                    train-only normalisation and id encoding
            product_dataset.py      DataFrame -> tensors -> DataLoaders
            print_processed_data.py the split report writer
model/      feature_tokenizer.py    a row -> [CLS] + title + numeric + categorical
            positional_encoding.py  sinusoids, applied to the text tokens only
            encoding_block.py       the nn.TransformerEncoder stack
            encoder_only_model.py   tokenizer -> encoder -> [CLS] -> MLP head
            attention.py            recover the weights nn.TransformerEncoder discards
            tokenizer.py            tiktoken GPT-2
            checkpoint.py           save/load trained weights (opt-in)
metrics/    metrics.py      precision / recall / fall-out from confusion counts
            run_results.py  what a run writes to disk, and how to read it back
            error_analysis.py       the test rows, decoded, worst first
            uncertainty.py  query-level bootstrap CI
            final_table.py  one row per configuration, the three seeds visible
plots/      plot_theme.py   the palette: colors, fonts, line weights
            threshold_curves.py     curves by sweeping every distinct score
            pr_auc.py  roc_auc.py   the two axis choices
            pr_auc_by_epoch.py  train_vs_val_error.py   the per-epoch curves
            config_comparison.py    several runs on the same axes
            calibration.py  reliability, score histogram, BTR by query
            attention_map.py  error_patterns_figure.py
            experiment_plots.py     the plotting path main.py and replot.py share
scripts/    reselect.py  error_patterns.py  epoch_band.py  attention_map.py
            eda_columns.py  eda_dataset.py  eda_slides.py
            step1_figures.py  slide_seleccion.py  count_bought_by_title_tag.py
baselines/  bert_model.py  visualize_trained_model.py   (side comparison, standalone)
docs/       INFORME.md      the single project document
            PROTOCOL.md     the frozen model-selection and evaluation protocol
output/     every generated artifact
```

**Source and generated files never share a directory.** Anything a run writes goes under
`output/`; a new figure goes there too, never next to the code that draws it.

> ⚠ `.gitignore` currently also ignores several source files and `docs/PROTOCOL.md`.
> Already-tracked files keep being versioned, but **new files in those paths are
> invisible to git**. See `docs/INFORME.md` §9 before the final commit.

There is no test suite and no linter config.
