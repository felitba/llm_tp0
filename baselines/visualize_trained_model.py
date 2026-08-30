import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report

def visualize_model():
    # 1. Load the fine-tuned model and tokenizer
    model_path = 'bert-tabular-model-final'
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)

    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 2. Load the completely unseen holdout data
    csv_filepath = 'dataset/test_supermarket_products.csv'
    df = pd.read_csv(csv_filepath)

    # Note: The 'bought' column is already cleaned since we split the data AFTER
    # cleaning it in the training script, but leaving this here is perfectly safe.
    df['bought'] = df['bought'].astype(str).str.strip().str.lower()
    df['bought'] = df['bought'].map({'true': 1, 'false': 0, '1': 1, '0': 0})
    df = df.dropna(subset=['bought'])
    df['bought'] = df['bought'].astype(int)

    # Take a random sample of 30 rows from the unseen data
    # (Or remove .sample() entirely to evaluate the whole test dataset)
    sample_df = df.sample(n=min(30, len(df)), random_state=37).copy()

    # Serialize the sample rows
    def serialize_row(row):
        features = []
        for col in df.columns:
            if col == 'bought': continue
            val = str(row[col]).strip() if pd.notna(row[col]) else "None"
            features.append(f"{col}: {val}")
        return " | ".join(features)

    sample_df['serialized_text'] = sample_df.apply(serialize_row, axis=1)

    # 3. Run Inference
    predictions = []

    print("Running predictions on unseen data...")
    for text in sample_df['serialized_text']:
        inputs = tokenizer(text, return_tensors="pt", padding='max_length', truncation=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=-1).item()
        predictions.append(predicted_class_id)

    sample_df['predicted_bought'] = predictions

    # 4. Visualize the Results
    print("\n--- Model Predictions vs Actual ---")
    sample_df['Actual'] = sample_df['bought'].map({1: 'True', 0: 'False'})
    sample_df['Predicted'] = sample_df['predicted_bought'].map({1: 'True', 0: 'False'})

    pd.set_option('display.max_colwidth', 50)
    print(sample_df[['title', 'Actual', 'Predicted']].to_string(index=False))

    # 5. Quick Evaluation Metrics
    accuracy = accuracy_score(sample_df['bought'], sample_df['predicted_bought'])
    print(f"\nSample Accuracy: {accuracy * 100:.2f}%")

    print(classification_report(sample_df['bought'], sample_df['predicted_bought']))

if __name__ == '__main__':
    visualize_model()