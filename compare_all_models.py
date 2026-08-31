import json
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification

# Import plotting functions from your repo
from plots.roc_auc import plot_roc_auc_by_config
from plots.pr_auc import plot_pr_auc_by_config
from plots.plot_theme import save


def get_bert_predictions():
    """Runs inference for BERT on the test split and returns (y_true, y_probs)."""
    model_path = 'baselines/bert-tabular-model-final'

    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load holdout test data
    df = pd.read_csv('dataset/test_supermarket_products.csv')
    df['bought'] = df['bought'].astype(str).str.strip().str.lower()
    df['bought'] = df['bought'].map({'true': 1, 'false': 0, '1': 1, '0': 0})
    df = df.dropna(subset=['bought'])
    df['bought'] = df['bought'].astype(int)

    def serialize_row(row):
        features = []
        for col in df.columns:
            if col == 'bought':
                continue
            val = str(row[col]).strip() if pd.notna(row[col]) else "None"
            features.append(f"{col}: {val}")
        return " | ".join(features)

    df['serialized_text'] = df.apply(serialize_row, axis=1)

    bert_probs = []
    print("Extracting probabilities from BERT...")
    for text in df['serialized_text']:
        inputs = tokenizer(text, return_tensors="pt", padding='max_length', truncation=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        probs = F.softmax(outputs.logits, dim=-1)
        bert_probs.append(probs[0][1].item())  # Probability of class 1

    return df['bought'].values, np.array(bert_probs)


def discover_experiment_directories(base_dir: Path) -> list[str]:
    """Scans base_dir and returns names of folders containing experiment predictions."""
    if not base_dir.exists():
        return []

    experiment_names = []
    for entry in sorted(base_dir.iterdir()):
        if entry.is_dir():
            # Check if directory contains a test predictions file or a run.json
            if (entry / "test_predictions.csv").exists() or (entry / "run.json").exists():
                experiment_names.append(entry.name)

    return experiment_names


def load_custom_experiment(exp_name: str):
    """Reads test labels and probabilities from test_predictions.csv."""
    exp_dir = Path(f"output/experiments/{exp_name}")
    csv_path = exp_dir / "test_predictions.csv"

    # Fallback to check run.json if test_predictions.csv filename differs
    json_path = exp_dir / "run.json"
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "predictions_file" in data:
            csv_path = exp_dir / data["predictions_file"]

    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find test predictions file at: {csv_path}")

    df = pd.read_csv(csv_path)

    # Dynamically find the label and probability columns
    label_col = next((col for col in ["label", "labels", "bought", "target", "y_true"] if col in df.columns), None)
    prob_col = next((col for col in ["prob", "probs", "probability", "pred", "score", "y_score"] if col in df.columns),
                    None)

    if label_col is None or prob_col is None:
        raise KeyError(
            f"Could not automatically locate label/prob columns in {csv_path}. "
            f"Available columns: {list(df.columns)}"
        )

    labels = df[label_col].values.astype(int)
    probs = df[prob_col].values.astype(float)
    return labels, probs


def main():
    experiments_dir = Path("output/experiments")

    # 1. Automatically find all custom experiment folders
    custom_experiment_names = discover_experiment_directories(experiments_dir)

    if not custom_experiment_names:
        print(f"No valid experiment folders found in {experiments_dir}")
        return

    print(f"Found {len(custom_experiment_names)} custom experiment(s):")
    for name in custom_experiment_names:
        print(f" - {name}")

    results_by_config = []

    # 2. Load probabilities from all discovered Custom Transformer Models
    for exp_name in custom_experiment_names:
        try:
            labels, probs = load_custom_experiment(exp_name)
            results_by_config.append((exp_name, labels, probs))
            print(f"Successfully loaded: {exp_name}")
        except Exception as e:
            print(f"Skipping {exp_name}: {e}")

    # 3. Get probabilities from BERT Baseline
    try:
        bert_labels, bert_probs = get_bert_predictions()
        results_by_config.append(("BERT Baseline", bert_labels, bert_probs))
    except Exception as e:
        print(f"Could not load BERT baseline: {e}")

    if not results_by_config:
        print("No valid predictions loaded. Exiting...")
        return

    # 4. Generate and Save Combined Plots
    print("\nGenerating combined ROC and PR curves...")

    roc_fig, _ = plot_roc_auc_by_config(results_by_config)
    roc_path = save(roc_fig, experiments_dir / "roc_auc_bert_vs_custom.jpg")
    print(f"Saved ROC comparison plot to: {roc_path}")

    pr_fig, _ = plot_pr_auc_by_config(results_by_config)
    pr_path = save(pr_fig, experiments_dir / "pr_auc_bert_vs_custom.jpg")
    print(f"Saved PR comparison plot to: {pr_path}")


if __name__ == "__main__":
    main()