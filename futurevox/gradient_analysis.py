import os
import argparse
import yaml
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import h5py

# Import your modules
from models.singer_modules.phoneme_encoder import EnhancedPhonemeEncoder
from data.singer_dataset import SingingVoxDataset


def read_config(config_path):
    """Read configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def analyze_model_gradients(model, device, input_phonemes, input_notes=None, masks=None):
    """Analyze gradients by parameter group and visualize them."""
    model.to(device)
    model.train()
    
    # Move inputs to device
    input_phonemes = input_phonemes.to(device)
    if input_notes is not None:
        input_notes = input_notes.to(device)
    if masks is not None:
        masks = masks.to(device)
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Forward pass
    encoded_phonemes, _ = model(input_phonemes, input_notes, masks)
    
    # Mean pooling over sequence length
    pooled = encoded_phonemes.mean(dim=1)
    
    # Simple loss: make the embeddings have unit norm
    loss = ((pooled.norm(dim=1) - 1.0) ** 2).mean()
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Collect gradients by layer type
    gradient_dict = {
        "embedding": [],
        "attention": [],
        "feed_forward": [],
        "layernorm": [],
        "projection": [],
        "other": []
    }
    
    param_dict = {
        "embedding": [],
        "attention": [],
        "feed_forward": [],
        "layernorm": [],
        "projection": [],
        "other": []
    }
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            param_norm = param.norm().item()
            
            if "embedding" in name:
                gradient_dict["embedding"].append(grad_norm)
                param_dict["embedding"].append(param_norm)
            elif "self_attn" in name:
                gradient_dict["attention"].append(grad_norm)
                param_dict["attention"].append(param_norm)
            elif "feed_forward" in name:
                gradient_dict["feed_forward"].append(grad_norm)
                param_dict["feed_forward"].append(param_norm)
            elif "norm" in name:
                gradient_dict["layernorm"].append(grad_norm)
                param_dict["layernorm"].append(param_norm)
            elif "projection" in name:
                gradient_dict["projection"].append(grad_norm)
                param_dict["projection"].append(param_norm)
            else:
                gradient_dict["other"].append(grad_norm)
                param_dict["other"].append(param_norm)
    
    # Compute statistics for each group
    stats = {}
    for key in gradient_dict:
        if gradient_dict[key]:
            stats[key] = {
                "grad_mean": np.mean(gradient_dict[key]),
                "grad_std": np.std(gradient_dict[key]),
                "grad_min": np.min(gradient_dict[key]),
                "grad_max": np.max(gradient_dict[key]),
                "param_mean": np.mean(param_dict[key]),
                "param_std": np.std(param_dict[key]),
                "param_min": np.min(param_dict[key]),
                "param_max": np.max(param_dict[key]),
            }
    
    return loss.item(), stats, gradient_dict, param_dict


def visualize_gradient_statistics(stats, gradient_dict, output_dir):
    """Create detailed visualizations of gradient statistics."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create bar charts of mean gradient norms by layer type
    plt.figure(figsize=(12, 6))
    
    # Extract mean gradient norms for each layer type
    layer_types = list(stats.keys())
    grad_means = [stats[key]["grad_mean"] for key in layer_types]
    
    # Create bar chart
    bars = plt.bar(layer_types, grad_means)
    
    # Add text labels above each bar
    for bar, mean in zip(bars, grad_means):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.00001,
                f'{mean:.6f}',
                ha='center', va='bottom', rotation=0, fontsize=8)
    
    plt.title('Mean Gradient Norm by Layer Type')
    plt.ylabel('Mean Gradient Norm')
    plt.xlabel('Layer Type')
    plt.yscale('log')  # Use log scale to better visualize different magnitudes
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "gradient_mean_by_layer.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a detail table of statistics
    with open(os.path.join(output_dir, "gradient_statistics.txt"), "w") as f:
        f.write("Gradient Statistics by Layer Type\n")
        f.write("================================\n\n")
        
        for layer_type, layer_stats in stats.items():
            f.write(f"{layer_type.upper()}\n")
            f.write("-" * len(layer_type) + "\n")
            f.write(f"Gradient Mean: {layer_stats['grad_mean']:.8f}\n")
            f.write(f"Gradient Std:  {layer_stats['grad_std']:.8f}\n")
            f.write(f"Gradient Min:  {layer_stats['grad_min']:.8f}\n")
            f.write(f"Gradient Max:  {layer_stats['grad_max']:.8f}\n")
            f.write(f"Parameter Mean: {layer_stats['param_mean']:.8f}\n")
            f.write(f"Parameter Std:  {layer_stats['param_std']:.8f}\n")
            f.write(f"Parameter Min:  {layer_stats['param_min']:.8f}\n")
            f.write(f"Parameter Max:  {layer_stats['param_max']:.8f}\n\n")
            
            # Calculate ratio of gradient norm to parameter norm
            grad_param_ratio = layer_stats['grad_mean'] / layer_stats['param_mean'] if layer_stats['param_mean'] > 0 else 0
            f.write(f"Gradient/Parameter Ratio: {grad_param_ratio:.8f}\n\n")
    
    # Create histograms of gradient norms for each layer type
    for layer_type, gradient_norms in gradient_dict.items():
        if gradient_norms:
            plt.figure(figsize=(10, 6))
            plt.hist(gradient_norms, bins=30, alpha=0.7)
            plt.title(f'Gradient Norm Distribution: {layer_type}')
            plt.xlabel('Gradient Norm')
            plt.ylabel('Count')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"gradient_hist_{layer_type}.png"), dpi=300, bbox_inches='tight')
            plt.close()


def analyze_embedding_weight_statistics(model, output_dir):
    """Analyze and visualize embedding weight statistics."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract embedding weights
    embedding_weight = None
    for name, param in model.named_parameters():
        if "phoneme_embedding.weight" in name:
            embedding_weight = param.detach().cpu().numpy()
            break
    
    if embedding_weight is None:
        print("No embedding weights found")
        return
    
    # Calculate statistics
    mean = np.mean(embedding_weight)
    std = np.std(embedding_weight)
    min_val = np.min(embedding_weight)
    max_val = np.max(embedding_weight)
    
    # Create a histogram of embedding values
    plt.figure(figsize=(10, 6))
    plt.hist(embedding_weight.flatten(), bins=50, alpha=0.7)
    plt.title(f'Embedding Weight Distribution\nMean={mean:.4f}, Std={std:.4f}, Min={min_val:.4f}, Max={max_val:.4f}')
    plt.xlabel('Weight Value')
    plt.ylabel('Count')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "embedding_weight_hist.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a heatmap of embedding weights
    plt.figure(figsize=(12, 8))
    plt.imshow(embedding_weight, cmap='viridis', aspect='auto')
    plt.colorbar(label='Weight Value')
    plt.title('Embedding Weight Matrix')
    plt.xlabel('Embedding Dimension')
    plt.ylabel('Phoneme Index')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "embedding_weight_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Compute row-wise statistics (per phoneme)
    row_means = np.mean(embedding_weight, axis=1)
    row_stds = np.std(embedding_weight, axis=1)
    row_norms = np.linalg.norm(embedding_weight, axis=1)
    
    # Plot row norms (phoneme norms)
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(row_norms)), row_norms)
    plt.title('Embedding Norm by Phoneme')
    plt.xlabel('Phoneme Index')
    plt.ylabel('L2 Norm')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "embedding_norms_by_phoneme.png"), dpi=300, bbox_inches='tight')
    plt.close()


def main(args):
    """Main function."""
    # Read configuration
    config = read_config(args.config)
    
    # Determine device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load dataset
    print(f"Loading dataset from {args.h5_path}...")
    dataset = SingingVoxDataset(args.h5_path, config, split='train')
    
    # Get number of phonemes from dataset
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
    
    # Initialize model
    print("Initializing phoneme encoder...")
    model = EnhancedPhonemeEncoder(
        config,
        num_phonemes=num_phonemes,
        num_notes=128
    )
    
    # Load model checkpoint if provided
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        state_dict = torch.load(args.checkpoint, map_location='cpu')
        # Handle different checkpoint formats
        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        # Remove 'model.' prefix if present
        if all(k.startswith('model.') for k in state_dict.keys()):
            state_dict = {k[6:]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
    
    # Get a batch from dataset
    batch = dataset[0]
    
    # Prepare inputs for model
    input_phonemes = batch['phone_indices'].unsqueeze(0)  # Add batch dimension
    input_notes = batch['note_indices'].unsqueeze(0) if 'note_indices' in batch else None
    masks = batch['phone_masks'].unsqueeze(0) if 'phone_masks' in batch else None
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Analyze gradients
    print("Analyzing gradients...")
    loss, stats, gradient_dict, param_dict = analyze_model_gradients(
        model, device, input_phonemes, input_notes, masks
    )
    
    # Visualize gradient statistics
    print("Visualizing gradient statistics...")
    visualize_gradient_statistics(stats, gradient_dict, args.output_dir)
    
    # Analyze embedding weights
    print("Analyzing embedding weights...")
    analyze_embedding_weight_statistics(model, args.output_dir)
    
    print(f"Analysis complete! All visualizations saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gradient Analysis for Phoneme Encoder")
    parser.add_argument("--config", type=str, default="config/default.yaml",
                        help="Path to config file")
    parser.add_argument("--h5_path", type=str, required=True,
                        help="Path to HDF5 dataset file")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (optional)")
    parser.add_argument("--output_dir", type=str, default="outputs/gradient_analysis",
                        help="Output directory for visualizations")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run model on")
    args = parser.parse_args()
    
    main(args)