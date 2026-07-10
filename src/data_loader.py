import pandas as pd
from pathlib import Path
import os

def load_raw_data(file_path):
    file_path = Path("data/raw/listings.csv")
    df = pd.read_csv(file_path)
    return df

def save_processed_data(df, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)