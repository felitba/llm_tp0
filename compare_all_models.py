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
    custom_experiment_names = [
        "grid_d_model64_n_heads2_num_layers2_dim_feedforward128",
        "grid_d_model64_n_heads2_num_layers2_dim_feedforward256",
        "grid_d_model64_n_heads2_num_layers2_dim_feedforward512"
    ]

    results_by_config = []

    # 1. Load probabilities from Custom Transformer Models
    for exp_name in custom_experiment_names:
        labels, probs = load_custom_experiment(exp_name)
        results_by_config.append((exp_name, labels, probs))
        print(f"Loaded custom model predictions: {exp_name}")

    # 2. Get probabilities from BERT
    bert_labels, bert_probs = get_bert_predictions()
    results_by_config.append(("BERT Baseline", bert_labels, bert_probs))

    # 3. Generate and Save Combined Plots
    print("\nGenerating combined ROC and PR curves...")

    output_dir = Path("output/experiments")
    output_dir.mkdir(parents=True, exist_ok=True)

    roc_fig, _ = plot_roc_auc_by_config(results_by_config)
    roc_path = save(roc_fig, output_dir / "roc_auc_bert_vs_custom.jpg")
    print(f"Saved ROC comparison plot to: {roc_path}")

    pr_fig, _ = plot_pr_auc_by_config(results_by_config)
    pr_path = save(pr_fig, output_dir / "pr_auc_bert_vs_custom.jpg")
    print(f"Saved PR comparison plot to: {pr_path}")


if __name__ == "__main__":
    main()