import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.models.ensemble import evaluate_soft_ensemble

def test_evaluate_soft_ensemble():
    class DummyModel(nn.Module):
        def forward(self, x):
            return torch.ones(x.size(0), 2)
    model1 = DummyModel()
    model2 = DummyModel()
    X = torch.randn(8, 3, 32, 32)
    y = torch.zeros(8, dtype=torch.long)
    ds = TensorDataset(X, y)
    loader = DataLoader(ds, batch_size=4)
    criterion = nn.NLLLoss()
    device = "cpu"
    loss, acc = evaluate_soft_ensemble(model1, model2, loader, criterion, device, alpha=0.5)
    assert 0 <= acc <= 1 