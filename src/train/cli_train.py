import argparse
import torch
from src.data.dataset import AnimalDataset
from src.data.loader import load_metadata
from src.models.backbones import MultiBackboneClassifier
from src.train.train import train_model
from src.utils.early_stopping import EarlyStopping
import torchvision.transforms as T
from torch.utils.data import DataLoader
from transformers import get_scheduler
import timm
import os
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description='Train Animal ReID Model')
    parser.add_argument('--train_csv', type=str, required=True)
    parser.add_argument('--val_csv', type=str, required=True)
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--model', type=str, default='multi', choices=['multi'])
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--output', type=str, default='output')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Label encoder
    train_identities = pd.read_csv(args.train_csv)['identity'].unique()
    label_encoder = {identity: idx for idx, identity in enumerate(sorted(train_identities))}

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

    train_ds = AnimalDataset(args.train_csv, args.root, label_encoder, train_transform)
    val_ds = AnimalDataset(args.val_csv, args.root, label_encoder, val_transform)
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