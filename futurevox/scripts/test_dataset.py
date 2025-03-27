"""
Test script to validate LightSinger dataset loading.
Checks if phonemes, durations, and F0 are properly loaded.
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from typing import Dict, List, Any, Optional

# Add parent directory to path to import config and dataset
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.model_config import FutureVoxConfig, DataConfig
from data.dataset import LightSingerDataset


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Test LightSinger dataset loading")
    
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Path to data directory"
    )
    
    parser.add_argument(
        "--sample_index", type=int, default=0,
        help="Index of sample to visualize"
    )
    
    parser.add_argument(
        "--output_dir", type=str, default="./dataset_test_output",
        help="Directory to save visualization outputs"
    )
    
    return parser.parse_args()


def plot_and_save(item: Dict[str, Any], output_dir: str, item_id: str) -> None:
    """
    Plot and save visualizations of the dataset item.
    
    Args:
        item: Dataset item
        output_dir: Output directory
        item_id: Item ID
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data
    phonemes = item["text"]
    durations = item["durations"].numpy()
    f0 = item["f0"].numpy()
    mel = item["mel"].numpy()
    
    # Plot durations
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(durations)), durations)
    plt.title(f"Phoneme Durations - {item_id}")
    plt.xlabel("Phoneme Index")
    plt.ylabel("Duration (frames)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{item_id}_durations.png"))
    plt.close()
    
    # Plot F0 contour
    plt.figure(figsize=(10, 4))
    plt.plot(f0)
    plt.title(f"F0 Contour - {item_id}")
    plt.xlabel("Frame Index")
    plt.ylabel("Frequency (Hz)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{item_id}_f0.png"))
    plt.close()
    
    # Plot mel spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        mel,
        y_axis='mel',
        x_axis='time',
        sr=22050,
        hop_length=256
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"Mel Spectrogram - {item_id}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{item_id}_mel.png"))
    plt.close()
    
    # Save phoneme and duration alignment
    phoneme_list = phonemes.split()
    duration_list = durations.tolist()
    
    with open(os.path.join(output_dir, f"{item_id}_alignment.txt"), "w") as f:
        f.write("Phoneme\tDuration (frames)\n")
        for p, d in zip(phoneme_list, duration_list):
            f.write(f"{p}\t{d}\n")


def validate_dataset_item(item: Dict[str, Any]) -> bool:
    """
    Validate dataset item.
    
    Args:
        item: Dataset item
        
    Returns:
        True if valid, False otherwise
    """
    # Check if all required keys exist
    required_keys = ["id", "phonemes", "durations", "f0", "mel", "text"]
    for key in required_keys:
        if key not in item:
            print(f"Missing key: {key}")
            return False
    
    # Check if phonemes and durations have the same length
    if len(item["phonemes"]) != len(item["durations"]):
        print(f"Phoneme length ({len(item['phonemes'])}) doesn't match duration length ({len(item['durations'])})")
        return False
    
    # Check if f0 and mel have valid values
    if torch.isnan(item["f0"]).any() or torch.isinf(item["f0"]).any():
        print("F0 contains NaN or inf values")
        return False
    
    if torch.isnan(item["mel"]).any() or torch.isinf(item["mel"]).any():
        print("Mel spectrogram contains NaN or inf values")
        return False
    
    return True


def print_item_stats(item: Dict[str, Any]) -> None:
    """
    Print statistics about the dataset item.
    
    Args:
        item: Dataset item
    """
    print(f"Item ID: {item['id']}")
    print(f"Phoneme text: {item['text']}")
    print(f"Number of phonemes: {len(item['phonemes'])}")
    print(f"Total duration (frames): {item['durations'].sum().item()}")
    print(f"F0 range: {item['f0'].min().item():.2f} - {item['f0'].max().item():.2f} Hz")
    print(f"Mel spectrogram shape: {item['mel'].shape}")
    
    # Print phoneme and duration details
    print("\nPhoneme alignment:")
    phoneme_list = item['text'].split()
    duration_list = item['durations'].tolist()
    
    for i, (p, d) in enumerate(zip(phoneme_list, duration_list)):
        print(f"  {i}: {p} - {d} frames")


def main():
    """Main test function."""
    args = parse_args()
    
    # Load configuration
    if args.config:
        config = FutureVoxConfig.from_yaml(args.config)
    else:
        # Use default configuration
        config = FutureVoxConfig()
    
    print(f"Testing dataset loading from directory: {args.data_dir}")
    
    # Create dataset
    dataset = LightSingerDataset(
        data_dir=args.data_dir,
        config=config.data,
        split="train"  # Use train split for testing
    )
    
    # Check if dataset is empty
    if len(dataset) == 0:
        print("Error: Dataset is empty!")
        sys.exit(1)
    
    # Get sample index
    sample_index = args.sample_index % len(dataset)
    print(f"Testing sample {sample_index} of {len(dataset)}")
    
    # Get sample
    item = dataset[sample_index]
    
    # Validate item
    is_valid = validate_dataset_item(item)
    print(f"Item validation: {'PASSED' if is_valid else 'FAILED'}")
    
    if is_valid:
        # Print statistics
        print("\nItem Statistics:")
        print("-" * 40)
        print_item_stats(item)
        
        # Plot and save visualizations
        print(f"\nSaving visualizations to {args.output_dir}")
        plot_and_save(item, args.output_dir, item["id"])
    
    print("\nAdditional validation:")
    print("-" * 40)
    
    # Check duration sum matches mel length
    duration_sum = item["durations"].sum().item()
    mel_length = item["mel"].shape[1]
    duration_match = abs(duration_sum - mel_length) <= 10  # Allow small margin of error
    
    print(f"Duration sum: {duration_sum}, Mel length: {mel_length}")
    print(f"Duration matches mel length: {'YES' if duration_match else 'NO (possible alignment issue)'}")
    
    # Check F0 values
    voiced_ratio = (item["f0"] > 0).float().mean().item() * 100
    print(f"Voiced frames: {voiced_ratio:.2f}% (should be reasonable for singing)")
    
    # Check overall dataset
    print("\nOverall Dataset Statistics:")
    print("-" * 40)
    print(f"Total samples: {len(dataset)}")
    print(f"Phoneme dictionary size: {len(dataset.phoneme_dict)}")
    
    print("\nTest completed!")


if __name__ == "__main__":
    main()