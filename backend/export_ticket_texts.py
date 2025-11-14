
import pickle
import pandas as pd


def load_csv(filepath):
    """Read a CSV file and return its DataFrame."""
    return pd.read_csv(filepath)


def extract_column(df, column_name):
    """Convert a text column to a clean Python list."""
    return [str(item) for item in df.get(column_name, [])]


def save_as_pickle(data, output_file):
    """Serialize data into a pickle file."""
    with open(output_file, "wb") as file:
        pickle.dump(data, file)


def main():
    source_file = "tickets_with_solutions.csv"
    output_pickle = "ticket_texts.pkl"
    column = "ticket_text"

    # Load, process, save
    dataframe = load_csv(source_file)
    texts = extract_column(dataframe, column)
    save_as_pickle(texts, output_pickle)

    print(f"{len(texts)} records saved to {output_pickle}")

