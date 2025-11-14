
import pandas as pd
from datasets import load_dataset


def load_source():
    """Fetch dataset from HuggingFace."""
    print("[INFO] Loading remote dataset...")
    data = load_dataset("Tobi-Bueck/customer-support-tickets")
    return data["train"]


def to_dataframe(hf_split):
    """Convert HF split to Pandas DataFrame."""
    df = pd.DataFrame(hf_split)
    print(f"[INFO] Columns found: {list(df.columns)}")
    return df


def clean_dataset(df):
    """
    Extract relevant fields and clean rows.
    Adjust column names here if dataset changes.
    """
    selected = df.loc[:, ["body", "answer"]].copy()
    selected.columns = ["ticket_text", "solution"]

    cleaned = selected.dropna()
    print(f"[INFO] Remaining rows after cleaning: {len(cleaned)}")
    return cleaned


def save_csv(df, out_file):
    """Save cleaned DataFrame to CSV."""
    df.to_csv(out_file, index=False)
    print(f"[SUCCESS] Saved → {out_file}")


def main():
    out_file = "tickets_with_solutions.csv"

    raw_split = load_source()
    frame = to_dataframe(raw_split)
    cleaned = clean_dataset(frame)
    save_csv(cleaned, out_file)

