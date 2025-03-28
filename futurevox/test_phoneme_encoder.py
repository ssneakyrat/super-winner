import os
import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import h5py
from tqdm import tqdm

# Import your existing modules
from models.singer_modules.phoneme_encoder import EnhancedPhonemeEncoder
from data.singer_dataset import SingingVoxDataset


def read_config(config_path):
    """Read configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def visualize_embeddings(embeddings, output_path, title="Phoneme Embeddings"):
    """
    Visualize embeddings using PCA projection.
    
    Args:
        embeddings: Tensor of embeddings [batch_size, seq_len, d_model]
        output_path: Path to save visualization
        title: Title for the plot
    """
    # Reshape to 2D: [batch_size * seq_len, d_model]
    embeddings_flat = embeddings.reshape(-1, embeddings.shape[-1])
    
    # Use PCA to reduce to 2D for visualization
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings_flat.detach().cpu().numpy())
    
    # Plot scatter
    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.5)
    plt.title(f"{title} (PCA projection)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Embedding visualization saved to {output_path}")


def visualize_attention(attention_weights, output_path, title="Attention Weights"):
    """
    Visualize attention weights as a heatmap.
    
    Args:
        attention_weights: Tensor of attention weights [batch_size, num_heads, seq_len, seq_len]
        output_path: Path to save visualization
        title: Title for the plot
    """
    if attention_weights is None or len(attention_weights) == 0:
        print("No attention weights to visualize")
        return
    
    # Select first batch and average across heads
    if isinstance(attention_weights, dict):
        # Handle dictionary of attention weights
        for layer_name, weights in attention_weights.items():
            layer_output_path = output_path.replace('.png', f'_{layer_name}.png')
            plt.figure(figsize=(10, 8))
            sns.heatmap(weights[0].mean(0).detach().cpu().numpy(), 
                        cmap='viridis', annot=False, fmt='.2f')
            plt.title(f"{title} - {layer_name}")
            plt.tight_layout()
            plt.savefig(layer_output_path, dpi=300, bbox_inches='tight')
            plt.close()
    else:
        # Handle tensor of attention weights
        plt.figure(figsize=(10, 8))
        sns.heatmap(attention_weights[0].mean(0).detach().cpu().numpy(), 
                    cmap='viridis', annot=False, fmt='.2f')
        plt.title(title)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Attention visualization saved to {output_path}")


def visualize_distance_matrix(embeddings, output_path, title="Embedding Distances"):
    """
    Visualize pairwise distances between embeddings.
    
    Args:
        embeddings: Tensor of embeddings [batch_size, seq_len, d_model]
        output_path: Path to save visualization
        title: Title for the plot
    """
    # Take just the first sample in the batch
    emb = embeddings[0].detach().cpu().numpy()
    
    # Compute pairwise distances
    from sklearn.metrics.pairwise import cosine_distances
    distances = cosine_distances(emb)
    
    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(distances, cmap='coolwarm', annot=False, fmt='.2f')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Distance matrix visualization saved to {output_path}")


def test_phoneme_encoder(dataset, phoneme_encoder, output_dir, num_samples=5, batch_size=2, device='cuda'):
    """
    Test phoneme encoder with dataset samples.
    
    Args:
        dataset: Dataset containing samples
        phoneme_encoder: EnhancedPhonemeEncoder model
        output_dir: Directory to save visualizations
        num_samples: Number of samples to process
        batch_size: Batch size for processing
        device: Device to run model on
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Move model to device and set to evaluation mode
    phoneme_encoder = phoneme_encoder.to(device)
    phoneme_encoder.eval()
    
    # Create dataloader for the subset of samples
    indices = list(range(min(num_samples, len(dataset))))
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(indices),
        collate_fn=dataset.collate_fn
    )
    
    print(f"Testing phoneme encoder with {num_samples} samples...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            # Prepare inputs
            phone_indices = batch['phone_indices'].to(device)
            phone_masks = batch['phone_masks'].to(device) if 'phone_masks' in batch else None
            note_indices = batch['note_indices'].to(device) if 'note_indices' in batch else None
            
            # Forward pass through phoneme encoder
            encoded_phonemes, attention_weights = phoneme_encoder(
                phone_indices, note_indices, phone_masks
            )
            
            # Visualize results
            batch_output_dir = os.path.join(output_dir, f"batch_{batch_idx}")
            os.makedirs(batch_output_dir, exist_ok=True)
            
            # Visualize embeddings
            visualize_embeddings(
                encoded_phonemes, 
                os.path.join(batch_output_dir, "phoneme_embeddings.png"),
                "Phoneme Embeddings"
            )
            
            # Visualize attention weights if available
            if attention_weights:
                visualize_attention(
                    attention_weights,
                    os.path.join(batch_output_dir, "attention_weights.png"),
                    "Attention Weights"
                )
            
            # Visualize distance matrix
            visualize_distance_matrix(
                encoded_phonemes,
                os.path.join(batch_output_dir, "distance_matrix.png"),
                "Phoneme Embedding Distances"
            )
            
            # Save tensor statistics
            with open(os.path.join(batch_output_dir, "stats.txt"), "w") as f:
                f.write(f"Encoded shape: {encoded_phonemes.shape}\n")
                f.write(f"Mean: {encoded_phonemes.mean().item()}\n")
                f.write(f"Std: {encoded_phonemes.std().item()}\n")
                f.write(f"Min: {encoded_phonemes.min().item()}\n")
                f.write(f"Max: {encoded_phonemes.max().item()}\n")
                
                # Check for NaN or inf values
                has_nan = torch.isnan(encoded_phonemes).any().item()
                has_inf = torch.isinf(encoded_phonemes).any().item()
                f.write(f"Has NaN: {has_nan}\n")
                f.write(f"Has Inf: {has_inf}\n")
            
            print(f"Batch {batch_idx} processed and saved to {batch_output_dir}")


def forward_consistency_test(phoneme_encoder, input_dim=100, seq_len=20, batch_size=2, device='cuda'):
    """
    Test that consecutive forward passes with the same input give the same output.
    
    Args:
        phoneme_encoder: EnhancedPhonemeEncoder model
        input_dim: Size of input dimension (num_phonemes)
        seq_len: Sequence length
        batch_size: Batch size
        device: Device to run model on
    
    Returns:
        success: True if test passes, False otherwise
    """
    # Move model to device and set to evaluation mode
    phoneme_encoder = phoneme_encoder.to(device)
    phoneme_encoder.eval()
    
    # Create dummy inputs
    phone_indices = torch.randint(0, input_dim, (batch_size, seq_len)).to(device)
    
    # Run forward pass twice with the same input
    with torch.no_grad():
        encoded1, _ = phoneme_encoder(phone_indices)
        encoded2, _ = phoneme_encoder(phone_indices)
    
    # Check if outputs are the same
    diff = (encoded1 - encoded2).abs().max().item()
    
    print(f"Forward consistency test: max difference = {diff:.8f}")
    return diff < 1e-6


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
    
    # Print model information
    num_params = sum(p.numel() for p in phoneme_encoder.parameters())
    print(f"Phoneme Encoder has {num_params:,} parameters")
    
    # Run forward consistency test
    print("Running forward consistency test...")
    consistency_success = forward_consistency_test(
        phoneme_encoder, 
        input_dim=num_phonemes,
        device=args.device
    )
    if not consistency_success:
        print("Warning: Forward consistency test failed. The model may be non-deterministic.")
    
    # Test with dataset samples
    print(f"Testing with {args.num_samples} samples...")
    test_phoneme_encoder(
        dataset,
        phoneme_encoder,
        args.output_dir,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        device=args.device
    )
    
    print("Testing complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Phoneme Encoder")
    parser.add_argument("--config", type=str, default="config/default.yaml",
                        help="Path to config file")
    parser.add_argument("--h5_path", type=str, required=True,
                        help="Path to HDF5 dataset file")
    parser.add_argument("--output_dir", type=str, default="outputs/phoneme_encoder_test",
                        help="Output directory for visualizations")
    parser.add_argument("--split", type=str, default="val",
                        help="Dataset split to use (train, val, test)")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of samples to test")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size for testing")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run model on")
    args = parser.parse_args()
    
    main(args)