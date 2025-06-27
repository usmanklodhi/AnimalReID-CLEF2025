import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class GLUBlock(nn.Module):
    def __init__(self, dim_in, dim_out, dropout=0.3):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out * 2)
        self.norm = nn.LayerNorm(dim_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_proj = self.linear(x)
        out, gate = x_proj.chunk(2, dim=-1)
        return self.norm(out * torch.sigmoid(gate))

class MultiBackboneClassifier(nn.Module):
    def __init__(self, model_names, embedding_dims, num_classes):
        super().__init__()
        assert len(model_names) == len(embedding_dims)
        self.backbones = nn.ModuleList([
            timm.create_model(name, pretrained=True, num_classes=0)
            for name in model_names
        ])
        total_dim = sum(embedding_dims)
        self.classifier = nn.Sequential(
            nn.LayerNorm(total_dim),
            GLUBlock(total_dim, 2048, dropout=0.4),
            GLUBlock(2048, 1024, dropout=0.3),
            GLUBlock(1024, 512, dropout=0.2),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        feats = [F.normalize(backbone(x), p=2, dim=1) for backbone in self.backbones]
        fused = torch.cat(feats, dim=1)
        return self.classifier(fused)

class EnsembleNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone1 = timm.create_model('swin_large_patch4_window12_384', pretrained=True, num_classes=0)
        self.backbone2 = timm.create_model('convnext_xlarge.fb_in22k_ft_in1k_384', pretrained=True, num_classes=0)
        self.backbone3 = timm.create_model('beit_large_patch16_384.in22k_ft_in22k_in1k', pretrained=True, num_classes=0)
        self.feat_dim1 = self.backbone1.num_features
        self.feat_dim2 = self.backbone2.num_features
        self.feat_dim3 = self.backbone3.num_features
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feat_dim1 + self.feat_dim2 + self.feat_dim3),
            nn.Linear(self.feat_dim1 + self.feat_dim2 + self.feat_dim3, 4096),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(4096, 2048),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(2048, num_classes)
        )
    def forward(self, x):
        f1 = self.backbone1(x)
        f2 = self.backbone2(x)
        f3 = self.backbone3(x)
        fused = torch.cat([f1, f2, f3], dim=1)
        return self.classifier(fused) 