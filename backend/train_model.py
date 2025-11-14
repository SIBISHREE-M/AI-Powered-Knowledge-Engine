
import torch
import pandas as pd
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)


def fetch_data():
    """Load the source dataset and return a cleaned DataFrame."""
    print("[INFO] Loading HuggingFace dataset...")
    raw = load_dataset("Tobi-Bueck/customer-support-tickets")["train"]

    rows = []
    for entry in raw:
        text = entry.get("body")
        pr = entry.get("priority")
        if text and pr:
            rows.append({"text": text, "priority": pr})

    df = pd.DataFrame(rows)
    return df


def prepare_labels(df):
    """Map priority strings to numeric labels and return filtered DataFrame."""
    mapping = {"low": 0, "medium": 1, "high": 2}
    df = df[df["priority"].isin(mapping)]
    df["label"] = df["priority"].map(mapping)
    return df


def tokenize_function(tokenizer):
    """Return a tokenizer function for Dataset.map()."""
    def _tok(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=128
        )
    return _tok


def make_hf_dataset(train_df, val_df, tokenizer):
    """Convert Pandas DataFrames to tokenized HuggingFace Datasets."""
    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)

    encode = tokenize_function(tokenizer)
    train_ds = train_ds.map(encode, batched=True)
    val_ds = val_ds.map(encode, batched=True)

    cols = ["input_ids", "attention_mask", "label"]
    train_ds.set_format("torch", columns=cols)
    val_ds.set_format("torch", columns=cols)

    return train_ds, val_ds


def build_model():
    """Initialize sequence classification model."""
    return AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=3
    )


def training_config():
    """Return TrainingArguments configuration."""
    return TrainingArguments(
        output_dir="./model",
        evaluation_strategy="epoch",
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        save_total_limit=1,
        logging_steps=50,
    )


def main():
    # Step 1: Load + clean
    df = fetch_data()
    df = prepare_labels(df)

    # Step 2: Train/val split
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    # Step 3: Tokenizer + datasets
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    train_ds, val_ds = make_hf_dataset(train_df, val_df, tokenizer)

    # Step 4: Model + Trainer
    model = build_model()
    args = training_config()

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
    )

    # Step 5: Train + save
    trainer.train()
    model.save_pretrained("ticket_priority_model")
    tokenizer.save_pretrained("ticket_priority_model")

    print("🎉 Training completed and model saved!")

