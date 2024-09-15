import pandas as pd

def load_data(file_path):
    """Load dataset from a CSV file."""
    return pd.read_csv(file_path)
