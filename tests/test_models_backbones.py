import torch
from src.models.backbones import MultiBackboneClassifier, EnsembleNet

def test_multi_backbone_classifier_forward():
    # Use small dummy models for test
    model_names = ["resnet18", "resnet18"]
    embedding_dims = [512, 512]
    num_classes = 2
    model = MultiBackboneClassifier(model_names, embedding_dims, num_classes)
    x = torch.randn(4, 3, 224, 224)
    out = model(x)
    assert out.shape == (4, num_classes)

def test_ensemble_net_forward():
    num_classes = 2
    model = EnsembleNet(num_classes)
    x = torch.randn(4, 3, 224, 224)
    out = model(x)
    assert out.shape == (4, num_classes) 