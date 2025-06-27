import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

class AnimalDataset(Dataset):
    def __init__(self, csv_file, root_dir, label_encoder, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.label_encoder = label_encoder
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.data.iloc[idx]['path'])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label_str = self.data.iloc[idx]['identity']
        label = self.label_encoder[label_str]
        return image, label 