import torch
import tempfile
from src.models.soup import weighted_average_models
import timm
import os

def test_weighted_average_models(tmp_path):
    # Create two dummy models and save their state dicts
    model_arch = "resnet18"
    num_classes = 2
    m1 = timm.create_model(model_arch, pretrained=False, num_classes=num_classes)
    m2 = timm.create_model(model_arch, pretrained=False, num_classes=num_classes)
    for p in m1.parameters():
        p.data.fill_(1.0)
    for p in m2.parameters():
        p.data.fill_(3.0)
    p1 = tmp_path / "m1.pth"
    p2 = tmp_path / "m2.pth"
    torch.save({'model_state_dict': m1.state_dict()}, p1)
    torch.save({'model_state_dict': m2.state_dict()}, p2)
    # Weighted average: 0.75 * m1 + 0.25 * m2
    model = weighted_average_models([str(p1), str(p2)], [0.75, 0.25], model_arch, num_classes, "cpu")
    for p in model.parameters():
        # Should be 1.5 (0.75*1 + 0.25*3)
        assert torch.allclose(p, torch.full_like(p, 1.5)) 