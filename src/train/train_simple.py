#!/usr/bin/env python3
"""
Simple training script for Animal ReID with the CLEF 2025 dataset
"""

import torch
import pandas as pd
from src.data.dataset import AnimalDataset
from src.models.backbones import MultiBackboneClassifier
from src.train.train import train_model
from src.utils.early_stopping import EarlyStopping
import torchvision.transforms as T
from torch.utils.data import DataLoader
from transformers import get_scheduler
import os
from sklearn.model_selection import train_test_split

def main():
    # Configuration
    metadata_csv = 'animal-clef-2025/metadata.csv'
    root_dir = 'animal-clef-2025'
    output_dir = 'output'
    epochs = 5
    batch_size = 8
    val_split = 0.2
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading metadata...")
    metadata = pd.read_csv(metadata_csv)
    
    # Filter out rows without identity (query images)
    trainable_data = metadata[metadata['identity'].notna()].copy()
    print(f"Total trainable samples: {len(trainable_data)}")
    print(f"Unique identities: {len(trainable_data['identity'].unique())}")
    
    # Create train/val splits
    train_data, val_data = train_test_split(
        trainable_data, 
        test_size=val_split, 
        random_state=42,
        stratify=trainable_data['identity']
    )
    
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    
    # Save splits for reference
    train_data.to_csv(os.path.join(output_dir, 'train_split.csv'), index=False)
    val_data.to_csv(os.path.join(output_dir, 'val_split.csv'), index=False)
    
    # Label encoder
    train_identities = train_data['identity'].unique()
    label_encoder = {identity: idx for idx, identity in enumerate(sorted(train_identities))}
    print(f"Number of classes: {len(label_encoder)}")
    
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
    
    # Create datasets
    train_ds = AnimalDataset(train_data, root_dir, label_encoder, train_transform)
    val_ds = AnimalDataset(val_data, root_dir, label_encoder, val_transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    
    # Model
    model_names = ["resnet18", "resnet18"]
    embedding_dims = [512, 512]
    model = MultiBackboneClassifier(model_names, embedding_dims, len(label_encoder))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    print(f"Using device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    num_training_steps = len(train_loader) * epochs
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps
    )
    early_stopping = EarlyStopping(patience=3, min_delta=0.001)
    
    # Train
    print("Starting training...")
    train_model(
        model, train_loader, val_loader, optimizer, scheduler, criterion,
        device, epochs, label_encoder, metrics_csv=os.path.join(output_dir, 'metrics.csv'), early_stopping=early_stopping
    )
    
    print("Training completed!")

if __name__ == '__main__':
    main() 