import pandas as pd
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.model_selection import train_test_split

def train_bert():
    # 1. Load your real CSV data
    csv_filepath = '../dataset/supermarket_products.csv'
    df = pd.read_csv(csv_filepath)

    # 2. Preprocess the target column ('bought')
    df['bought'] = df['bought'].astype(str).str.strip().str.lower()
    df['bought'] = df['bought'].map({'true': 1, 'false': 0})
    df = df.dropna(subset=['bought'])
    df['bought'] = df['bought'].astype(int)

    # NEW STEP: Create the holdout test set BEFORE training
    # We save 15% of the dataset to disk for the visualization script to use later
    train_df, holdout_df = train_test_split(df, test_size=0.10, random_state=39)
    holdout_df.to_csv('../dataset/test_supermarket_products.csv', index=False)
    print(f"Saved {len(holdout_df)} rows to test_supermarket_products.csv for evaluation.")

    # We overwrite 'df' with 'train_df' so the model never sees the holdout data
    df = train_df.copy()

    # 3. Text-Serialization
    def serialize_row(row):
        features = []
        for col in df.columns:
            if col == 'bought':
                continue
            val = str(row[col]).strip() if pd.notna(row[col]) else "None"
            features.append(f"{col}: {val}")
        return " | ".join(features)

    df['serialized_text'] = df.apply(serialize_row, axis=1)

    # 4. Convert to Hugging Face Dataset format
    hf_dataset = Dataset.from_pandas(df[['serialized_text', 'bought']])
    hf_dataset = hf_dataset.rename_column("bought", "label")

    # 5. Tokenization
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def tokenize_function(examples):
        return tokenizer(examples['serialized_text'], padding='max_length', truncation=True, max_length=128)

    tokenized_datasets = hf_dataset.map(tokenize_function, batched=True)

    # 6. Train/Eval Split (This eval is just for calculating loss during training)
    split_dataset = tokenized_datasets.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']

    # 7. Model Initialization
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased', num_labels=2,
        hidden_dropout_prob=0.1,attention_probs_dropout_prob=0.1)

    # 8. Training Arguments & Trainer Setup
    training_args = TrainingArguments(
        output_dir='./bert-tabular-model',
        learning_rate=5e-5,  # Your learning_rate
        num_train_epochs=5,
        per_device_train_batch_size=64,  # Your batch_size
        weight_decay=0.01,  # Your weight_decay
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,  # Required for early stopping
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=100,
                early_stopping_threshold=0.0001
            )
        ]
    )

    # 9. Start Fine-Tuning
    trainer.train()

    # 10. Save the fine-tuned model and tokenizer
    trainer.save_model('./bert-tabular-model-final')
    tokenizer.save_pretrained('./bert-tabular-model-final')

if __name__ == '__main__':
    train_bert()