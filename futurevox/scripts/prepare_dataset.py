"""
Dataset preparation script for LightSinger.
Processes a directory of .lab and .wav files, creates train/val/test splits,
and generates necessary metadata files.
"""

import os
import argparse
import random
import json
import glob
import shutil
import librosa
from tqdm import tqdm
from typing import Dict, List, Tuple, Set

# Fix imports for running from futurevox/ directory
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.model_config import FutureVoxConfig


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Prepare dataset for LightSinger")
    
    parser.add_argument(
        "--config", type=str, default="./config/default.yaml",
        help="Path to configuration file (default: ./config/default.yaml)"
    )
    
    parser.add_argument(
        "--val_ratio", type=float, default=0.1,
        help="Ratio of data to use for validation"
    )
    
    parser.add_argument(
        "--test_ratio", type=float, default=0.1,
        help="Ratio of data to use for testing"
    )
    
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--copy_files", action="store_true",
        help="Copy files to output directory instead of using paths"
    )
    
    return parser.parse_args()


def find_lab_wav_pairs(input_dir: str) -> List[Tuple[str, str]]:
    """
    Find pairs of .lab and .wav files.
    
    Args:
        input_dir: Input directory
        
    Returns:
        List of (lab_path, wav_path) pairs
    """
    pairs = []
    
    # Find all .lab files
    lab_files = glob.glob(os.path.join(input_dir, "**/*.lab"), recursive=True)
    
    for lab_path in lab_files:
        # Generate expected wav path by replacing .lab with .wav
        wav_path = lab_path.replace(".lab", ".wav")
        
        if os.path.exists(wav_path):
            pairs.append((lab_path, wav_path))
    
    return pairs


def split_dataset(
    pairs: List[Tuple[str, str]],
    val_ratio: float,
    test_ratio: float,
    seed: int
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        pairs: List of (lab_path, wav_path) pairs
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        seed: Random seed
        
    Returns:
        Tuple of (train_pairs, val_pairs, test_pairs)
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Shuffle pairs
    pairs_copy = pairs.copy()
    random.shuffle(pairs_copy)
    
    # Calculate split indices
    n_samples = len(pairs_copy)
    n_test = int(n_samples * test_ratio)
    n_val = int(n_samples * val_ratio)
    n_train = n_samples - n_test - n_val
    
    # Split dataset
    train_pairs = pairs_copy[:n_train]
    val_pairs = pairs_copy[n_train:n_train + n_val]
    test_pairs = pairs_copy[n_train + n_val:]
    
    return train_pairs, val_pairs, test_pairs


def create_filelists(
    output_dir: str,
    train_pairs: List[Tuple[str, str]],
    val_pairs: List[Tuple[str, str]],
    test_pairs: List[Tuple[str, str]],
    base_dir: str = None
):
    """
    Create filelists for train, validation, and test sets.
    
    Args:
        output_dir: Output directory
        train_pairs: List of training (lab_path, wav_path) pairs
        val_pairs: List of validation (lab_path, wav_path) pairs
        test_pairs: List of test (lab_path, wav_path) pairs
        base_dir: Base directory to make paths relative to
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    def get_relative_path(path, base):
        if base:
            return os.path.relpath(path, base)
        return path
    
    # Write train filelist
    with open(os.path.join(output_dir, "train_filelist.txt"), "w") as f:
        for lab_path, _ in train_pairs:
            f.write(f"{get_relative_path(lab_path, base_dir)}\n")
    
    # Write validation filelist
    with open(os.path.join(output_dir, "val_filelist.txt"), "w") as f:
        for lab_path, _ in val_pairs:
            f.write(f"{get_relative_path(lab_path, base_dir)}\n")
    
    # Write test filelist
    with open(os.path.join(output_dir, "test_filelist.txt"), "w") as f:
        for lab_path, _ in test_pairs:
            f.write(f"{get_relative_path(lab_path, base_dir)}\n")


def copy_files(
    output_dir: str,
    train_pairs: List[Tuple[str, str]],
    val_pairs: List[Tuple[str, str]],
    test_pairs: List[Tuple[str, str]]
):
    """
    Copy files to output directory with train/val/test subdirectories.
    
    Args:
        output_dir: Output directory
        train_pairs: List of training (lab_path, wav_path) pairs
        val_pairs: List of validation (lab_path, wav_path) pairs
        test_pairs: List of test (lab_path, wav_path) pairs
    """
    # Create subdirectories
    os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "val"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "test"), exist_ok=True)
    
    # Copy training files
    print("Copying training files...")
    for lab_path, wav_path in tqdm(train_pairs):
        lab_name = os.path.basename(lab_path)
        wav_name = os.path.basename(wav_path)
        
        shutil.copy2(lab_path, os.path.join(output_dir, "train", lab_name))
        shutil.copy2(wav_path, os.path.join(output_dir, "train", wav_name))
    
    # Copy validation files
    print("Copying validation files...")
    for lab_path, wav_path in tqdm(val_pairs):
        lab_name = os.path.basename(lab_path)
        wav_name = os.path.basename(wav_path)
        
        shutil.copy2(lab_path, os.path.join(output_dir, "val", lab_name))
        shutil.copy2(wav_path, os.path.join(output_dir, "val", wav_name))
    
    # Copy test files
    print("Copying test files...")
    for lab_path, wav_path in tqdm(test_pairs):
        lab_name = os.path.basename(lab_path)
        wav_name = os.path.basename(wav_path)
        
        shutil.copy2(lab_path, os.path.join(output_dir, "test", lab_name))
        shutil.copy2(wav_path, os.path.join(output_dir, "test", wav_name))
    
    # Update filelists to use relative paths
    create_filelists(
        output_dir,
        [(os.path.join("train", os.path.basename(p[0])), os.path.join("train", os.path.basename(p[1]))) for p in train_pairs],
        [(os.path.join("val", os.path.basename(p[0])), os.path.join("val", os.path.basename(p[1]))) for p in val_pairs],
        [(os.path.join("test", os.path.basename(p[0])), os.path.join("test", os.path.basename(p[1]))) for p in test_pairs],
        None
    )


def extract_phonemes_from_lab(lab_path: str) -> Set[str]:
    """
    Extract unique phonemes from a .lab file.
    
    Args:
        lab_path: Path to .lab file
        
    Returns:
        Set of unique phonemes
    """
    phonemes = set()
    
    with open(lab_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                phoneme = parts[2]
                phonemes.add(phoneme)
    
    return phonemes


def create_phoneme_dict(
    output_dir: str,
    pairs: List[Tuple[str, str]]
):
    """
    Create phoneme dictionary from all .lab files.
    
    Args:
        output_dir: Output directory
        pairs: List of (lab_path, wav_path) pairs
    """
    # Collect all unique phonemes
    all_phonemes = set()
    
    print("Extracting phonemes from .lab files...")
    for lab_path, _ in tqdm(pairs):
        phonemes = extract_phonemes_from_lab(lab_path)
        all_phonemes.update(phonemes)
    
    # Create phoneme to ID mapping
    phoneme_dict = {p: i + 1 for i, p in enumerate(sorted(all_phonemes))}
    # 0 is reserved for padding/unknown
    
    # Save phoneme dictionary
    with open(os.path.join(output_dir, "phoneme_dict.json"), "w") as f:
        json.dump(phoneme_dict, f, indent=2)
    
    print(f"Created phoneme dictionary with {len(phoneme_dict)} phonemes")


def validate_wav_files(pairs: List[Tuple[str, str]], sample_rate: int = 22050):
    """
    Validate WAV files to ensure they can be loaded properly.
    
    Args:
        pairs: List of (lab_path, wav_path) pairs
        sample_rate: Expected sample rate
    """
    print("Validating WAV files...")
    invalid_files = []
    
    for _, wav_path in tqdm(pairs):
        try:
            # Try to load audio file
            y, sr = librosa.load(wav_path, sr=sample_rate)
            
            # Check if audio is too short
            if len(y) < 1000:  # Arbitrary threshold for "too short"
                invalid_files.append((wav_path, "too short"))
        except Exception as e:
            invalid_files.append((wav_path, str(e)))
    
    # Report invalid files
    if invalid_files:
        print(f"Found {len(invalid_files)} invalid WAV files:")
        for path, reason in invalid_files:
            print(f"  {path}: {reason}")
    else:
        print("All WAV files are valid")


def generate_stats(pairs: List[Tuple[str, str]], output_dir: str):
    """
    Generate statistics about the dataset.
    
    Args:
        pairs: List of (lab_path, wav_path) pairs
        output_dir: Output directory
    """
    # Collect statistics
    total_duration = 0
    phoneme_counts = {}
    num_utterances = len(pairs)
    
    print("Generating dataset statistics...")
    for lab_path, wav_path in tqdm(pairs):
        # Get audio duration
        y, sr = librosa.load(wav_path, sr=None)
        duration = len(y) / sr
        total_duration += duration
        
        # Count phonemes
        phonemes = extract_phonemes_from_lab(lab_path)
        for p in phonemes:
            phoneme_counts[p] = phoneme_counts.get(p, 0) + 1
    
    # Save statistics
    stats = {
        "num_utterances": num_utterances,
        "total_duration_hours": total_duration / 3600,
        "avg_utterance_duration": total_duration / num_utterances,
        "num_unique_phonemes": len(phoneme_counts),
        "phoneme_frequency": phoneme_counts
    }
    
    with open(os.path.join(output_dir, "dataset_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    
    # Print summary
    print(f"Dataset Statistics:")
    print(f"  Total utterances: {num_utterances}")
    print(f"  Total duration: {total_duration / 3600:.2f} hours")
    print(f"  Average utterance duration: {total_duration / num_utterances:.2f} seconds")
    print(f"  Number of unique phonemes: {len(phoneme_counts)}")


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = FutureVoxConfig.from_yaml(args.config)
    
    # Get directories from config
    output_dir = config.data.datasets_root
    
    # By convention, use a "raw" subdirectory in the same parent directory as datasets_root
    parent_dir = os.path.dirname(os.path.abspath(output_dir))
    input_dir = parent_dir #os.path.join(parent_dir, "raw")
    
    # Ensure input directory exists
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}. Please create a 'raw' directory with your .lab and .wav files in {parent_dir}")
    
    print(f"Using input directory: {input_dir}")
    print(f"Using output directory: {output_dir}")
    
    # Find lab/wav pairs
    print(f"Scanning directory: {input_dir}")
    pairs = find_lab_wav_pairs(input_dir)
    print(f"Found {len(pairs)} lab/wav pairs")
    
    # Validate pairs
    if len(pairs) == 0:
        print("Error: No valid lab/wav pairs found!")
        return
    
    # Validate WAV files
    validate_wav_files(pairs)
    
    # Split dataset
    train_pairs, val_pairs, test_pairs = split_dataset(
        pairs, args.val_ratio, args.test_ratio, args.seed
    )
    
    print(f"Dataset split:")
    print(f"  Training: {len(train_pairs)} samples")
    print(f"  Validation: {len(val_pairs)} samples")
    print(f"  Test: {len(test_pairs)} samples")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create phoneme dictionary
    create_phoneme_dict(output_dir, pairs)
    
    # Generate dataset statistics
    generate_stats(pairs, output_dir)
    
    # Process files
    if args.copy_files:
        # Copy files to output directory
        copy_files(
            output_dir,
            train_pairs,
            val_pairs,
            test_pairs
        )
    else:
        # Create filelists with relative paths
        create_filelists(
            output_dir,
            train_pairs,
            val_pairs,
            test_pairs,
            input_dir
        )
    
    print(f"Dataset preparation complete!")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()