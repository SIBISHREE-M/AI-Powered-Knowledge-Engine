

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm


def select_device():
    """Return CUDA if available, else CPU."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(name, device):
    """Load the embedding model."""
    print(f"[INFO] Loading model: {name}")
    return SentenceTransformer(name, device=device)


def read_texts(csv_path, column):
    """Read the dataset and extract text column."""
    print(f"[INFO] Reading data from: {csv_path}")
    df = pd.read_csv(csv_path)

    if column not in df.columns:
        raise KeyError(f"Missing column '{column}'. Available: {list(df.columns)}")

    texts = df[column].astype(str).tolist()
    print(f"[INFO] Found {len(texts)} text entries.")
    return texts, df


def compute_embeddings(model, texts, batch_size=64):
    """Generate embeddings for all texts."""
    print("[INFO] Encoding texts...")
    return model.encode(
        texts,
        batch_size=batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )


def save_outputs(embeds, mapping_df, embed_file, mapping_file):
    """Save embeddings and text mapping."""
    torch.save(embeds.cpu(), embed_file)
    print(f"[INFO] Embeddings saved → {embed_file}")

    mapping_df.to_csv(mapping_file, index=False)
    print(f"[INFO] Mapping saved → {mapping_file}")


def main():
    # Paths & configs
    csv_path = "tickets_with_solutions.csv"
    text_column = "ticket_text"
    model_name = "intfloat/e5-small-v2"
    out_embeddings = "ticket_embeddings.pkl"
    out_mapping = "ticket_mapping.csv"

    # Device + Model
    device = select_device()
    print(f"[INFO] Using device: {device}")
    model = load_model(model_name, device)

    # Load dataset
    texts, df = read_texts(csv_path, text_column)

    # Generate embeddings
    embeddings = compute_embeddings(model, texts)

    # Prepare mapping file
    df_map = pd.DataFrame({"id": range(len(texts)), text_column: texts})

    # Save results
    save_outputs(embeddings, df_map, out_embeddings, out_mapping)

    print("[SUCCESS] Embedding pipeline completed!")

