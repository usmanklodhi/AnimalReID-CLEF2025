import pandas as pd
import os
import torch
from src.inference.infer import run_inference
import timm

def test_run_inference(tmp_path):
    # Create dummy image
    from PIL import Image
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
    # Create and save dummy model
    model_arch = "resnet18"
    num_classes = 1
    model = timm.create_model(model_arch, pretrained=False, num_classes=num_classes)
    for p in model.parameters():
        p.data.fill_(1.0)
    model_path = tmp_path / "model.pth"
    torch.save({'model_state_dict': model.state_dict()}, model_path)
    # Run inference
    output_csv = tmp_path / "out.csv"
    run_inference(str(model_path), model_arch, label_encoder, str(csv_path), str(tmp_path), str(output_csv), "cpu")
    out_df = pd.read_csv(output_csv)
    assert 'predicted_identity' in out_df.columns
    assert 'confidence' in out_df.columns 