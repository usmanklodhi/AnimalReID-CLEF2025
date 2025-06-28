import argparse
import torch
from src.data.dataset import AnimalDataset
from src.models.backbones import MultiBackboneClassifier
from src.train.train import train_model
from src.utils.early_stopping import EarlyStopping
import torchvision.transforms as T
from torch.utils.data import DataLoader
from transformers.optimization import get_scheduler
import timm
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple

def main():
    parser = argparse.ArgumentParser(description='Train Animal ReID Model')
    parser.add_argument('--metadata_csv', type=str, default='animal-clef-2025/metadata.csv')
    parser.add_argument('--root', type=str, default='animal-clef-2025')
    parser.add_argument('--model', type=str, default='multi', choices=['multi'])
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--output', type=str, default='output')
    parser.add_argument('--val_split', type=float, default=0.2)
    parser.add_argument('--random_state', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load metadata and create train/val splits
    metadata = pd.read_csv(args.metadata_csv)
    
    # Filter out rows without identity (query images)
    trainable_data = metadata[metadata['identity'].notna()].copy()
    
    # Create train/val splits
    train_data, val_data = train_test_split(  # type: ignore
        trainable_data, 
        test_size=args.val_split, 
        random_state=args.random_state,
        stratify=trainable_data['identity']
    )
    
    # Save splits for reference
    train_data.to_csv(os.path.join(args.output, 'train_split.csv'), index=False)  # type: ignore
    val_data.to_csv(os.path.join(args.output, 'val_split.csv'), index=False)  # type: ignore

    # Label encoder - use ALL unique identities from the entire dataset
    all_identities = trainable_data['identity'].unique()  # type: ignore
    label_encoder = {identity: idx for idx, identity in enumerate(sorted(all_identities))}
    
    print(f"Total unique identities: {len(label_encoder)}")
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")

    # Transforms
    train_transform = T.Compose([
        T.RandomResizedCrop(224, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
    ])
    val_transform = T.Compose([
        T.Resize([224, 224]),
        T.ToTensor(),
    ])

    # Create datasets using the splits
    train_ds = AnimalDataset(train_data, args.root, label_encoder, train_transform)
    val_ds = AnimalDataset(val_data, args.root, label_encoder, val_transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # Model
    model_names = ["resnet18", "resnet18"]
    embedding_dims = [512, 512]
    model = MultiBackboneClassifier(model_names, embedding_dims, len(label_encoder))
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    num_training_steps = len(train_loader) * args.epochs
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps
    )
    early_stopping = EarlyStopping(patience=3, min_delta=0.001)

    train_model(
        model, train_loader, val_loader, optimizer, scheduler, criterion,
        model.device, args.epochs, label_encoder, metrics_csv=os.path.join(args.output, 'metrics.csv'), early_stopping=early_stopping
    )

if __name__ == '__main__':
    main() 