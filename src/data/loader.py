import os

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
import torchvision.transforms as T

from src.data.dataset import AnimalDataset


def load_metadata(csv_path):
    """Load metadata CSV as DataFrame."""
    return pd.read_csv(csv_path)

def show_info(df):
    """Print DataFrame info."""
    print(df.info())

def get_dataloader(data_dir, batch_size, split='test', num_workers=4):
    """
    Creates a DataLoader for a specific data split.
    Filters data based on the 'split' column in metadata.csv.
    """
    csv_path = os.path.join(data_dir, 'metadata.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"metadata.csv not found in {data_dir}")

    df_full = pd.read_csv(csv_path)
    
    # Filter the dataframe for the specified split
    df = df_full[df_full['split'] == split].copy()

    if df.empty:
        raise ValueError(f"No data found for split '{split}' in {csv_path}")
    
    # Fit the label encoder on all identities to ensure consistent encoding
    le = LabelEncoder()
    le.fit(df_full['identity'])
    
    # Create a dictionary mapping from string label to integer
    label_encoder_dict = {label: i for i, label in enumerate(le.classes_)}

    # Apply the encoding to the current split's identities
    # Using a direct map is safer than transform if some identities are missing in a split
    df['identity_encoded'] = df['identity'].map(label_encoder_dict)

    # Assuming a standard transformation for testing
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = AnimalDataset(
        csv_file_or_df=df,
        root_dir=data_dir,
        label_encoder=label_encoder_dict,
        transform=transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader