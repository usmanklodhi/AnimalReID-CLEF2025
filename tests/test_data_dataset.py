import pandas as pd
import os
from src.data.dataset import AnimalDataset
from PIL import Image
import torch

def test_animal_dataset(tmp_path):
    # Create dummy image
    img_path = tmp_path / "img1.jpg"
    img = Image.new('RGB', (10, 10))
    img.save(img_path)
    # Create dummy CSV
    csv_path = tmp_path / "meta.csv"
    df = pd.DataFrame({
        'path': [os.path.basename(img_path)],
        'identity': ['cat']
    })
    df.to_csv(csv_path, index=False)
    label_encoder = {'cat': 0}
    ds = AnimalDataset(str(csv_path), str(tmp_path), label_encoder)
    image, label = ds[0]
    assert isinstance(image, Image.Image) or (torch.is_tensor(image) and image.shape[-1] == 3)
    assert label == 0 