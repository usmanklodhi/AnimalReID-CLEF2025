import torch
import pandas as pd
from src.data.dataset import AnimalDataset
from torch.utils.data import DataLoader
from torchvision import transforms as T
import timm
import os
from src.models.backbones import MultiBackboneClassifier
from PIL import Image
import os
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms as T

# --- Loader ---
def load_model(model_path, model_names, embedding_dims, num_classes=None, device="cpu"):
    """
    Loads MultiBackboneClassifier with the correct num_classes inferred from the checkpoint
    (label_encoder or classifier weight shape), then loads weights. Returns (model, label_encoder).
    """
    ckpt = torch.load(model_path, map_location=device)

    # Extract the saved state dict whether raw or wrapped
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

    # Try to infer num_classes
    le = None
    if isinstance(ckpt, dict):
        le = ckpt.get("label_encoder", None)

    inferred_num_classes = None
    if le is not None and hasattr(le, "classes_"):
        inferred_num_classes = len(le.classes_)
    else:
        # Fallback: infer from classifier layer shape in the checkpoint
        # adjust the key if your final linear layer name differs
        for k in ("classifier.4.weight", "classifier.fc.weight", "classifier[-1].weight"):
            if k in state:
                inferred_num_classes = state[k].shape[0]
                break

    if num_classes is None:
        if inferred_num_classes is None:
            raise ValueError("num_classes not provided and cannot be inferred from checkpoint.")
        num_classes = inferred_num_classes

    # Now build the model with the correct class count
    model = MultiBackboneClassifier(model_names, embedding_dims, num_classes)
    model.load_state_dict(state, strict=True)  # should match now that num_classes is correct
    model = model.to(device).eval()
    return model, le


# --- Single image prediction ---
def predict(model, image_path, transform, device):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(image)
        probs = torch.softmax(logits, dim=1)
        pred = probs.argmax(dim=1).item()
        conf = probs.max().item()
    return pred, conf

# --- Inference for submission ---
def run_inference(model_path, model_names, embedding_dims, metadata_csv, sample_csv, root_dir, output_csv, device="cpu"):
    """
    Runs inference using MultiBackboneClassifier.
    - Uses sample_submission.csv for output format + order
    - Uses metadata.csv for image paths
    """
    # Load references
    meta = pd.read_csv(metadata_csv)
    sample = pd.read_csv(sample_csv)

    # Build lookup: image_id -> path
    id2path = dict(zip(meta["image_id"].astype(str), meta["path"].astype(str)))

    # # Load model + label encoder
    # model, le = load_model(model_path, model_names, embedding_dims, device=device)
    # if le is None or not hasattr(le, "classes_"):
    #     raise ValueError("No valid label_encoder found in checkpoint.")
    # idx_to_label = {i: lab for i, lab in enumerate(le.classes_)}

    # after loading:
    model, classes = load_model(model_path, model_names, embedding_dims, device=device)
    if classes is None:
        raise ValueError("Could not recover class names from checkpoint. Save a label encoder or classes list.")

    idx_to_label = {i: lab for i, lab in enumerate(classes)}

    # Preprocessing
    transform = T.Compose([
        T.Resize([224, 224]),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Iterate in sample_submission order
    preds, confs = [], []
    for _, row in sample.iterrows():
        image_id = str(row["image_id"])
        rel_path = id2path.get(image_id)
        if rel_path is None:
            raise FileNotFoundError(f"image_id {image_id} not found in metadata.csv")
        img_fp = os.path.join(root_dir, rel_path)
        if not os.path.isfile(img_fp):
            raise FileNotFoundError(f"Image not found on disk: {img_fp}")

        pred_idx, conf = predict(model, img_fp, transform, device)
        preds.append(idx_to_label[pred_idx])
        confs.append(conf)

    # Build submission
    submission = sample.copy()
    submission["identity"] = preds
    submission.to_csv(output_csv, index=False)
    print(f"Submission saved to {output_csv}")

    # Optional confidence file
    aux_path = os.path.splitext(output_csv)[0] + "_with_conf.csv"
    submission["confidence"] = confs
    submission.to_csv(aux_path, index=False)
    print(f"Submission+confidence saved to {aux_path}")
