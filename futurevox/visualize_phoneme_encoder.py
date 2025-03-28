import os
import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import h5py
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
import pandas as pd
from tqdm import tqdm

# Import your existing modules
from models.singer_modules.phoneme_encoder import EnhancedPhonemeEncoder
from data.singer_dataset import SingingVoxDataset


def read_config(config_path):
    """Read configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def extract_phoneme_embeddings(model, dataset, device='cuda', max_samples=1000):
    """
    Extract phoneme embeddings for visualization.
    
    Args:
        model: Phoneme encoder model
        dataset: Dataset containing samples
        device: Device to run model on
        max_samples: Maximum number of samples to process
        
    Returns:
        embeddings: List of embedding vectors
        labels: List of phoneme labels
    """
    # Set model to evaluation mode
    model.eval()
    
    all_embeddings = []
    all_labels = []
    
    # Process limited number of samples
    num_samples = min(max_samples, len(dataset))
    
    with torch.no_grad():
        for i in tqdm(range(num_samples), desc="Extracting embeddings"):
            # Get sample
            sample = dataset[i]
            
            # Prepare inputs
            phone_indices = sample['phone_indices'].unsqueeze(0).to(device)
            
            # Get phoneme labels
            phones = sample['phones']
            
            # Forward pass through phoneme encoder
            encoded_phonemes, _ = model(phone_indices)
            
            # Extract embedding for each phoneme
            for j, phone in enumerate(phones):
                if j < encoded_phonemes.size(1):  # Check if within sequence length
                    embedding = encoded_phonemes[0, j].cpu().numpy()
                    all_embeddings.append(embedding)
                    all_labels.append(phone)
    
    return all_embeddings, all_labels


def visualize_embeddings_by_category(embeddings, labels, output_path, 
                                     method='tsne', title="Phoneme Embeddings"):
    """
    Visualize embeddings with coloring by phoneme categories.
    
    Args:
        embeddings: List of embedding vectors
        labels: List of phoneme labels
        output_path: Path to save visualization
        method: Dimensionality reduction method ('tsne', 'pca', or 'umap')
        title: Title for the plot
    """
    # Convert to numpy array
    embeddings_array = np.array(embeddings)
    
    # Define phoneme categories
    vowels = ['iy', 'ih', 'eh', 'ae', 'aa', 'ah', 'ao', 'uh', 'uw', 'ux', 'er', 'ax', 'ix', 'axr', 'ey', 'ay', 'oy', 'aw', 'ow']
    nasals = ['m', 'n', 'ng', 'em', 'en', 'eng']
    fricatives = ['f', 'v', 'th', 'dh', 's', 'z', 'sh', 'zh', 'hh']
    stops = ['p', 'b', 't', 'd', 'k', 'g']
    affricates = ['ch', 'jh']
    liquids = ['l', 'r', 'el']
    glides = ['w', 'y']
    silence = ['pau', 'sil', 'sp']
    
    # Create category mapping
    category_map = {}
    for phone in vowels:
        category_map[phone] = "Vowel"
    for phone in nasals:
        category_map[phone] = "Nasal"
    for phone in fricatives:
        category_map[phone] = "Fricative"
    for phone in stops:
        category_map[phone] = "Stop"
    for phone in affricates:
        category_map[phone] = "Affricate"
    for phone in liquids:
        category_map[phone] = "Liquid"
    for phone in glides:
        category_map[phone] = "Glide"
    for phone in silence:
        category_map[phone] = "Silence"
    
    # Assign categories to labels
    categories = [category_map.get(label, "Other") for label in labels]
    
    # Reduce dimensionality for visualization
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        embeddings_2d = reducer.fit_transform(embeddings_array)
    elif method == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        embeddings_2d = reducer.fit_transform(embeddings_array)
    elif method == 'umap':
        reducer = umap.UMAP(n_components=2, random_state=42)
        embeddings_2d = reducer.fit_transform(embeddings_array)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Create DataFrame for plotting
    df = pd.DataFrame({
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1],
        'category': categories,
        'label': labels
    })
    
    # Plot with categorical colors
    plt.figure(figsize=(12, 10))
    sns.scatterplot(data=df, x='x', y='y', hue='category', style='category', alpha=0.7)
    plt.title(f"{title} ({method.upper()} projection)")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend(title="Phoneme Type")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Embedding visualization saved to {output_path}")
    
    # Also create an interactive HTML version with hover labels
    try:
        import plotly.express as px
        fig = px.scatter(df, x='x', y='y', color='category', hover_data=['label'],
                        title=f"{title} ({method.upper()} projection)")
        html_path = output_path.replace('.png', '.html')
        fig.write_html(html_path)
        print(f"Interactive visualization saved to {html_path}")
    except ImportError:
        print("Plotly not installed, skipping interactive visualization")


def visualize_embedding_similarity(embeddings, labels, output_path, title="Phoneme Similarity"):
    """
    Visualize similarity matrix between phoneme embeddings.
    
    Args:
        embeddings: List of embedding vectors
        labels: List of phoneme labels
        output_path: Path to save visualization
        title: Title for the plot
    """
    # Convert to numpy array
    embeddings_array = np.array(embeddings)
    
    # Get unique phonemes
    unique_labels = sorted(set(labels))
    
    # Create a dictionary to store average embedding for each phoneme
    phoneme_embeddings = {label: [] for label in unique_labels}
    
    # Collect embeddings for each phoneme
    for i, label in enumerate(labels):
        phoneme_embeddings[label].append(embeddings[i])
    
    # Compute average embedding for each phoneme
    average_embeddings = {}
    for label in unique_labels:
        if phoneme_embeddings[label]:
            average_embeddings[label] = np.mean(phoneme_embeddings[label], axis=0)
    
    # Create similarity matrix
    num_phonemes = len(average_embeddings)
    similarity_matrix = np.zeros((num_phonemes, num_phonemes))
    phoneme_list = list(average_embeddings.keys())
    
    for i, p1 in enumerate(phoneme_list):
        for j, p2 in enumerate(phoneme_list):
            # Compute cosine similarity
            emb1 = average_embeddings[p1]
            emb2 = average_embeddings[p2]
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            similarity_matrix[i, j] = similarity
    
    # Plot similarity matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(similarity_matrix, cmap='viridis', xticklabels=phoneme_list, yticklabels=phoneme_list,
               annot=False, fmt='.2f', cbar=True)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Similarity matrix visualization saved to {output_path}")


def visualize_attention_weights(model, dataset, output_dir, device='cuda', num_samples=5):
    """
    Visualize attention patterns in the phoneme encoder.
    
    Args:
        model: Phoneme encoder model
        dataset: Dataset containing samples
        output_dir: Directory to save visualizations
        device: Device to run model on
        num_samples: Number of samples to visualize
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set model to evaluation mode
    model.eval()
    
    # Copy the model and modify it to return attention weights
    class AttentionCapturingModel(torch.nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model
            
        def forward(self, phone_indices, note_indices=None, phone_masks=None):
            # Get the original output
            encoded_phonemes, _ = self.base_model(phone_indices, note_indices, phone_masks)
            
            # Collect attention weights from all layers
            attention_weights = {}
            
            # Each transformer layer has a MultiHeadAttention module
            for i, layer in enumerate(self.base_model.layers):
                # Get self attention module
                self_attn = layer.self_attn
                
                # Create synthetic inputs to compute attention
                batch_size, seq_len = phone_indices.size()
                d_model = self.base_model.config.get('phoneme_encoder', {}).get('d_model', 256)
                
                # Create a dummy input tensor
                dummy_input = torch.randn(batch_size, seq_len, d_model, device=phone_indices.device)
                
                # Forward pass through attention module
                with torch.no_grad():
                    _, attn = self_attn(dummy_input, dummy_input, dummy_input, mask=None)
                    attention_weights[f'layer_{i}'] = attn
            
            return encoded_phonemes, attention_weights
    
    # Create attention capturing model
    attention_model = AttentionCapturingModel(model).to(device)
    
    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            # Get sample
            sample = dataset[i]
            
            # Prepare inputs
            phone_indices = sample['phone_indices'].unsqueeze(0).to(device)
            phones = sample['phones']
            
            # Forward pass through model
            _, attention_weights = attention_model(phone_indices)
            
            # Visualize attention weights for each layer
            for layer_name, weights in attention_weights.items():
                # Get first batch, first head
                attn = weights[0, 0].cpu().numpy()
                
                # Create heatmap
                plt.figure(figsize=(10, 8))
                sns.heatmap(attn, cmap='viridis', annot=False, fmt='.2f', 
                           xticklabels=phones, yticklabels=phones)
                plt.title(f"Attention Weights - {layer_name}")
                plt.xlabel("Target Phonemes")
                plt.ylabel("Source Phonemes")
                plt.tight_layout()
                
                # Save figure
                output_path = os.path.join(output_dir, f"sample_{i}_{layer_name}_attention.png")
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"Attention visualization saved to {output_path}")


def main(args):
    """Main function."""
    # Read configuration
    config = read_config(args.config)
    
    # Load dataset
    print(f"Loading dataset from {args.h5_path}...")
    dataset = SingingVoxDataset(args.h5_path, config, split=args.split)
    
    # Get number of phonemes
    with h5py.File(args.h5_path, 'r') as f:
        # Get file list
        file_list = f['metadata']['file_list'][:]
        file_ids = [name.decode('utf-8') for name in file_list]
        
        # Collect all phonemes
        all_phonemes = set()
        for sample_id in file_ids:
            if sample_id in f:
                sample = f[sample_id]
                if 'phonemes' in sample and 'phones' in sample['phonemes']:
                    phones_bytes = sample['phonemes']['phones'][:]
                    phones = [p.decode('utf-8') for p in phones_bytes]
                    all_phonemes.update(phones)
        
        num_phonemes = len(all_phonemes)
        print(f"Found {num_phonemes} unique phonemes in the dataset")
    
    # Initialize phoneme encoder
    print("Initializing phoneme encoder...")
    phoneme_encoder = EnhancedPhonemeEncoder(
        config,
        num_phonemes=num_phonemes,
        num_notes=128
    )
    
    # Load checkpoint if provided
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}...")
        state_dict = torch.load(args.checkpoint, map_location='cpu')
        phoneme_encoder.load_state_dict(state_dict)
    
    # Move model to device
    device = args.device
    phoneme_encoder = phoneme_encoder.to(device)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Extract embeddings
    print("Extracting phoneme embeddings...")
    embeddings, labels = extract_phoneme_embeddings(
        phoneme_encoder, dataset, device, max_samples=args.max_samples
    )
    
    # Visualize embeddings
    print("Visualizing embeddings...")
    visualize_embeddings_by_category(
        embeddings, labels,
        os.path.join(args.output_dir, "phoneme_embeddings_tsne.png"),
        method='tsne',
        title="Phoneme Embeddings"
    )
    
    visualize_embeddings_by_category(
        embeddings, labels,
        os.path.join(args.output_dir, "phoneme_embeddings_pca.png"),
        method='pca',
        title="Phoneme Embeddings"
    )
    
    try:
        visualize_embeddings_by_category(
            embeddings, labels,
            os.path.join(args.output_dir, "phoneme_embeddings_umap.png"),
            method='umap',
            title="Phoneme Embeddings"
        )
    except:
        print("UMAP visualization failed. Make sure UMAP is installed.")
    
    # Visualize similarity matrix
    print("Visualizing similarity matrix...")
    visualize_embedding_similarity(
        embeddings, labels,
        os.path.join(args.output_dir, "phoneme_similarity.png"),
        title="Phoneme Embedding Similarity"
    )
    
    # Visualize attention weights
    print("Visualizing attention weights...")
    visualize_attention_weights(
        phoneme_encoder, dataset,
        os.path.join(args.output_dir, "attention"),
        device, num_samples=args.num_attention_samples
    )
    
    print("Visualization complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Phoneme Encoder")
    parser.add_argument("--config", type=str, default="config/default.yaml",
                        help="Path to config file")
    parser.add_argument("--h5_path", type=str, required=True,
                        help="Path to HDF5 dataset file")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="outputs/phoneme_visualizations",
                        help="Output directory for visualizations")
    parser.add_argument("--split", type=str, default="val",
                        help="Dataset split to use (train, val, test)")
    parser.add_argument("--max_samples", type=int, default=100,
                        help="Maximum number of samples to process for embeddings")
    parser.add_argument("--num_attention_samples", type=int, default=5,
                        help="Number of samples to use for attention visualization")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run model on")
    args = parser.parse_args()
    
    main(args)