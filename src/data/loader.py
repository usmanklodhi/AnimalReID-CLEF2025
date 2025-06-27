import pandas as pd

def load_metadata(csv_path):
    """Load metadata CSV as DataFrame."""
    return pd.read_csv(csv_path)

def show_info(df):
    """Print DataFrame info."""
    print(df.info()) 