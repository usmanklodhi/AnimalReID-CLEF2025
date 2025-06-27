import torch
import timm

def load_state_dict(path):
    checkpoint = torch.load(path, map_location='cpu')
    return checkpoint['model_state_dict']

def weighted_average_models(model_paths, weights, model_arch, num_classes, device):
    assert len(model_paths) == len(weights), "Number of models and weights must match."
    weights = [float(w) for w in weights]
    total_weight = sum(weights)
    norm_weights = [w / total_weight for w in weights]
    base_model = timm.create_model(model_arch, pretrained=False, num_classes=num_classes).to(device)
    avg_state_dict = None
    for i, path in enumerate(model_paths):
        state_dict = load_state_dict(path)
        weight = norm_weights[i]
        if avg_state_dict is None:
            avg_state_dict = {k: v.clone().float() * weight for k, v in state_dict.items()}
        else:
            for k in avg_state_dict:
                avg_state_dict[k] += state_dict[k].float() * weight
    base_model.load_state_dict(avg_state_dict)
    return base_model 