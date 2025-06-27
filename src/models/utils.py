def freeze_backbones(model):
    for backbone in model.backbones:
        for param in backbone.parameters():
            param.requires_grad = False

def unfreeze_backbones(model):
    for backbone in model.backbones:
        for param in backbone.parameters():
            param.requires_grad = True 