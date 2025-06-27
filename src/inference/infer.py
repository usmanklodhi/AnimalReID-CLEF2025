import torch
import pandas as pd
from src.data.dataset import AnimalDataset
from torch.utils.data import DataLoader
from torchvision import transforms as T
import timm
import os
from PIL import Image

def load_model(model_path, model_arch, num_classes, device):
    model = timm.create_model(model_arch, pretrained=False, num_classes=num_classes)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model

def predict(model, image_path, transform, device):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(image)
        probs = torch.softmax(logits, dim=1)
        pred = probs.argmax(dim=1).item()
        conf = probs.max().item()
    return pred, conf

def run_inference(model_path, model_arch, label_encoder, csv_path, root_dir, output_csv, device):
    df = pd.read_csv(csv_path)
    transform = T.Compose([
        T.Resize([224, 224]),
        T.ToTensor(),
    ])
    model = load_model(model_path, model_arch, len(label_encoder), device)
    idx_to_label = {v: k for k, v in label_encoder.items()}
    preds, confs = [], []
    for _, row in df.iterrows():
        img_path = os.path.join(root_dir, row['path'])
        pred, conf = predict(model, img_path, transform, device)
        preds.append(idx_to_label[pred])
        confs.append(conf)
    df['predicted_identity'] = preds
    df['confidence'] = confs
    df.to_csv(output_csv, index=False)
    print(f"Saved predictions to {output_csv}") 