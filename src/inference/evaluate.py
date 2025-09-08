import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from tqdm import tqdm

from src.data.loader import get_dataloader
from src.models.backbones import get_model


def get_embeddings(model, dataloader, device):
    """Extract embeddings for all images in a dataloader."""
    model.eval()
    all_embeddings = []
    all_labels = []
    all_filepaths = []

    with torch.no_grad():
        for images, labels, filepaths in tqdm(dataloader, desc="Extracting embeddings"):
            images = images.to(device)
            embeddings = model(images)
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_filepaths.extend(filepaths)

    return np.vstack(all_embeddings), np.array(all_labels), all_filepaths


def calculate_map(query_embeddings, query_labels, gallery_embeddings, gallery_labels):
    """Calculate mean Average Precision (mAP)."""
    num_queries = query_embeddings.shape[0]
    aps = []

    for i in range(num_queries):
        query_embedding = query_embeddings[i]
        query_label = query_labels[i]

        distances = np.linalg.norm(gallery_embeddings - query_embedding, axis=1)
        sorted_indices = np.argsort(distances)

        relevant_mask = gallery_labels[sorted_indices] == query_label
        
        if not np.any(relevant_mask):
            continue

        precision_at_k = np.cumsum(relevant_mask) / (np.arange(len(relevant_mask)) + 1)
        ap = np.sum(precision_at_k * relevant_mask) / np.sum(relevant_mask)
        aps.append(ap)

    return np.mean(aps) if aps else 0.0


def plot_tsne(embeddings, labels, save_path="tsne.png"):
    """Generate and save a t-SNE plot."""
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
    tsne_results = tsne.fit_transform(embeddings)

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='viridis', alpha=0.5)
    plt.legend(handles=scatter.legend_elements()[0], labels=np.unique(labels).tolist())
    plt.title("t-SNE Visualization of Image Embeddings")
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.savefig(save_path)
    plt.close()
    print(f"t-SNE plot saved to {save_path}")


def plot_qualitative_results(query_embeddings, query_labels, query_filepaths, gallery_embeddings, gallery_labels, gallery_filepaths, save_dir="qualitative_results", num_queries=5, top_k=5):
    """Save qualitative results for a few queries."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i in range(num_queries):
        query_embedding = query_embeddings[i]
        query_label = query_labels[i]
        query_filepath = query_filepaths[i]

        distances = np.linalg.norm(gallery_embeddings - query_embedding, axis=1)
        sorted_indices = np.argsort(distances)[:top_k]

        fig, axes = plt.subplots(1, top_k + 1, figsize=(15, 3))
        
        # Show query image
        query_img = plt.imread(query_filepath)
        axes[0].imshow(query_img)
        axes[0].set_title(f"Query: {query_label}")
        axes[0].axis('off')

        # Show top_k gallery images
        for j, idx in enumerate(sorted_indices):
            gallery_filepath = gallery_filepaths[idx]
            gallery_label = gallery_labels[idx]
            gallery_img = plt.imread(gallery_filepath)
            axes[j + 1].imshow(gallery_img)
            is_correct = gallery_label == query_label
            title_color = 'green' if is_correct else 'red'
            axes[j + 1].set_title(f"Rank {j+1}: {gallery_label}", color=title_color)
            axes[j + 1].axis('off')
        
        plt.savefig(os.path.join(save_dir, f"query_{i}.png"))
        plt.close()
    print(f"Qualitative results saved to {save_dir}")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = get_model(args.model_name, pretrained=False)
    model.load_state_dict(torch.load(args.model_path))
    model = model.to(device)

    # Get dataloaders
    # Assuming you have a function to get query and gallery dataloaders
    # You might need to adapt this based on your `loader.py`
    query_loader = get_dataloader(args.data_dir, batch_size=args.batch_size, split='test')
    gallery_loader = get_dataloader(args.data_dir, batch_size=args.batch_size, split='test') # Or a separate gallery set

    # Extract embeddings
    query_embeddings, query_labels, query_filepaths = get_embeddings(model, query_loader, device)
    gallery_embeddings, gallery_labels, gallery_filepaths = get_embeddings(model, gallery_loader, device)

    # Calculate mAP
    mean_ap = calculate_map(query_embeddings, query_labels, gallery_embeddings, gallery_labels)
    print(f"Mean Average Precision (mAP): {mean_ap:.4f}")

    # Generate t-SNE plot
    if args.tsne:
        plot_tsne(gallery_embeddings, gallery_labels, save_path=args.tsne_path)

    # Generate qualitative results
    if args.qualitative:
        plot_qualitative_results(query_embeddings, query_labels, query_filepaths, gallery_embeddings, gallery_labels, gallery_filepaths, save_dir=args.qualitative_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a Re-ID model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model file.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument("--model_name", type=str, default="resnet50", help="Name of the model architecture.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference.")
    
    parser.add_argument("--tsne", action="store_true", help="Generate a t-SNE plot.")
    parser.add_argument("--tsne_path", type=str, default="tsne.png", help="Path to save the t-SNE plot.")
    
    parser.add_argument("--qualitative", action="store_true", help="Generate qualitative results.")
    parser.add_argument("--qualitative_path", type=str, default="qualitative_results", help="Directory to save qualitative results.")

    args = parser.parse_args()
    main(args)
