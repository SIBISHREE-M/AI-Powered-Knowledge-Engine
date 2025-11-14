

import pickle
import pandas as pd
from sentence_transformers import SentenceTransformer


def load_data(csv_file):
    """Load the CSV file containing tickets and solutions."""
    print("[INFO] Reading dataset...")
    return pd.read_csv(csv_file, engine="python", on_bad_lines="skip")


def load_embedder(model_name):
    """Initialize the embedding model."""
    print(f"[INFO] Loading embedding model: {model_name}")
    return SentenceTransformer(model_name)


def create_embeddings(model, texts):
    """Generate embeddings from the given list of texts."""
    print("[INFO] Generating embeddings...")
    vectors = model.encode(
        texts,
        convert_to_tensor=True
    )
    return vectors.cpu()


def save_pickle(obj, file_path):
    """Save Python object as pickle."""
    with open(file_path, "wb") as handle:
        pickle.dump(obj, handle)
    print(f"[SUCCESS] Embeddings stored at: {file_path}")


def main():
    data_file = "tickets_with_solutions.csv"
    output_file = "ticket_embeddings.pkl"
    model_name = "all-MiniLM-L6-v2"

    df = load_data(data_file)
    embedder = load_embedder(model_name)

    text_list = df["ticket_text"].astype(str).tolist()
    embedding_vectors = create_embeddings(embedder, text_list)

    save_pickle(embedding_vectors, output_file)

