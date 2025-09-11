import torch
import pandas as pd
from src.data.dataset import AnimalDataset
from torch.utils.data import DataLoader
from torchvision import transforms as T
import timm
import os
from src.models.backbones import MultiBackboneClassifier
from PIL import Image

def load_model(model_path, model_names, embedding_dims, num_classes, device):
    """Loads the MultiBackboneClassifier model."""
    model = MultiBackboneClassifier(model_names, embedding_dims, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
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

def run_inference(model_path, model_names, embedding_dims, label_encoder, csv_path, root_dir, output_csv, device):
    """Runs inference on a set of images using the MultiBackboneClassifier model."""
    df = pd.read_csv(csv_path)
    transform = T.Compose([
        T.Resize([224, 224]),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    num_classes = len(label_encoder.classes_)
    model = load_model(model_path, model_names, embedding_dims, num_classes, device)
    
    idx_to_label = {i: label for i, label in enumerate(label_encoder.classes_)}
    
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