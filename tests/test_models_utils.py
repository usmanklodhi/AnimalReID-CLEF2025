import torch.nn as nn
from src.models import utils

class DummyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbones = nn.ModuleList([DummyBackbone() for _ in range(2)])

def test_freeze_and_unfreeze():
    model = DummyModel()
    utils.freeze_backbones(model)
    for backbone in model.backbones:
        for param in backbone.parameters():
            assert not param.requires_grad
    utils.unfreeze_backbones(model)
    for backbone in model.backbones:
        for param in backbone.parameters():
            assert param.requires_grad 