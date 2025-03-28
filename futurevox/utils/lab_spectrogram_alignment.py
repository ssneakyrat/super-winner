#!/usr/bin/env python3
# lab_spectrogram_alignment.py

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse
from pathlib import Path

def read_lab_file(lab_file):
    """
    Read a .lab file and return a dataframe with start_time, end_time, and label.
    """
    data = []
    with open(lab_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                start_time, end_time, label = parts
                data.append({
                    'start_time': int(start_time),
                    'end_time': int(end_time),
                    'label': label
                })
    
    return pd.DataFrame(data)

def convert_lab_time_to_seconds(df, sample_rate=None, time_unit='samples'):
    """
    Convert lab file timings to seconds.
    
    Args:
        df: DataFrame with 'start_time' and 'end_time'
        sample_rate: Sample rate of the audio (needed if time_unit is 'samples')
        time_unit: Can be 'samples' or 'milliseconds'
    """
    df = df.copy()
    
    if time_unit == 'samples' and sample_rate:
        df['start_sec'] = df['start_time'] / sample_rate
        df['end_sec'] = df['end_time'] / sample_rate
    elif time_unit == 'milliseconds':
        df['start_sec'] = df['start_time'] / 1000
        df['end_sec'] = df['end_time'] / 1000
    else:
        raise ValueError("time_unit must be 'samples' or 'milliseconds' and sample_rate must be provided for 'samples'")
    
    return df

def visualize_alignment(wav_file, lab_file, time_unit='samples', n_mels=128, n_fft=2048, hop_length=512, 
                        output_file=None, figsize=(20, 10)):
    """
    Visualize the alignment between a .lab file and a mel spectrogram.
    
    Args:
        wav_file: Path to the WAV file
        lab_file: Path to the .lab file
        time_unit: Can be 'samples', 'milliseconds', or 'auto'
        n_mels, n_fft, hop_length: Parameters for mel spectrogram extraction
        output_file: If provided, save the plot to this file
        figsize: Figure size for the plot
    """
    # Load the audio file
    y, sr = librosa.load(wav_file, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    print(f"Audio duration: {duration:.2f} seconds, Sample rate: {sr} Hz")
    
    # Extract mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    # Read lab file
    lab_df = read_lab_file(lab_file)
    
    # Auto-detect time unit if specified
    if time_unit == 'auto':
        # Check if times are beyond what's reasonable for a sample rate
        max_time = lab_df['end_time'].max()
        if max_time > duration * sr * 1.5:  # If more than 150% of audio samples
            print(f"Auto-detected time unit as milliseconds (max time: {max_time})")
            time_unit = 'milliseconds'
        else:
            print(f"Auto-detected time unit as samples (max time: {max_time}, samples: {duration * sr})")
            time_unit = 'samples'
    
    # Convert lab times to seconds
    lab_df = convert_lab_time_to_seconds(lab_df, sr, time_unit)
    
    # Create the plot
    plt.figure(figsize=figsize)
    
    # Plot mel spectrogram
    librosa.display.specshow(
        log_mel_spectrogram, 
        sr=sr, 
        hop_length=hop_length, 
        x_axis='time', 
        y_axis='mel'
    )
    plt.colorbar(format='%+2.0f dB')
    
    # Get axis limits
    ax = plt.gca()
    y_min, y_max = ax.get_ylim()
    
    # Add phoneme boundaries and labels
    for i, row in lab_df.iterrows():
        # Add vertical line at boundary
        plt.axvline(x=row['start_sec'], color='r', linestyle='--', alpha=0.7)
        
        # Add label text
        label_x = (row['start_sec'] + row['end_sec']) / 2
        plt.text(label_x, y_max*0.9, row['label'], 
                 horizontalalignment='center', 
                 verticalalignment='center',
                 fontsize=10, 
                 bbox=dict(facecolor='white', alpha=0.7))
    
    # Add final boundary
    plt.axvline(x=lab_df.iloc[-1]['end_sec'], color='r', linestyle='--', alpha=0.7)
    
    plt.title(f"Alignment: {os.path.basename(wav_file)} with {os.path.basename(lab_file)}")
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved to {output_file}")
    else:
        plt.show()
    
    return lab_df

def find_corresponding_wav(lab_file, data_dir=None):
    """
    Find the corresponding WAV file for a given lab file.
    
    Args:
        lab_file: Path to the lab file
        data_dir: Base directory for data (optional)
    
    Returns:
        Path to the WAV file
    """
    lab_path = Path(lab_file)
    file_name = lab_path.stem
    
    # Try different directory structures
    potential_paths = []
    
    # If data_dir is provided
    if data_dir:
        # Structure: data_dir/wav/filename.wav
        potential_paths.append(Path(data_dir) / "wav" / f"{file_name}.wav")
        
        # Structure: same directory as lab but in wav subdirectory
        if "lab" in str(lab_path):
            wav_path = str(lab_path).replace("lab", "wav").replace(".lab", ".wav")
            potential_paths.append(Path(wav_path))
    
    # Try the same directory
    potential_paths.append(lab_path.parent / f"{file_name}.wav")
    
    # Try parent directory
    potential_paths.append(lab_path.parent.parent / "wav" / f"{file_name}.wav")
    
    for path in potential_paths:
        if path.exists():
            return str(path)
    
    return None

def main():
    parser = argparse.ArgumentParser(description="Visualize alignment between .lab files and mel spectrograms")
    parser.add_argument("lab_file", help="Path to the .lab file")
    parser.add_argument("--wav", help="Path to the WAV file (optional, will try to find automatically)")
    parser.add_argument("--data-dir", help="Base directory for data files")
    parser.add_argument("--time-unit", choices=["samples", "milliseconds", "auto"], default="auto", 
                       help="Time unit used in the .lab file")
    parser.add_argument("--output", help="Path to save the output image (optional)")
    parser.add_argument("--n-mels", type=int, default=128, help="Number of mel bands")
    parser.add_argument("--n-fft", type=int, default=2048, help="FFT window size")
    parser.add_argument("--hop-length", type=int, default=512, help="Hop length for FFT")
    
    args = parser.parse_args()
    
    # Find WAV file if not specified
    wav_file = args.wav
    if not wav_file:
        wav_file = find_corresponding_wav(args.lab_file, args.data_dir)
        if not wav_file:
            print(f"ERROR: Could not find corresponding WAV file for {args.lab_file}")
            print("Please specify WAV file path using --wav option")
            return
    
    print(f"Using WAV file: {wav_file}")
    print(f"Using LAB file: {args.lab_file}")
    
    # Visualize alignment
    lab_df = visualize_alignment(
        wav_file=wav_file,
        lab_file=args.lab_file,
        time_unit=args.time_unit,
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        output_file=args.output
    )
    
    # Print timing statistics
    total_duration = lab_df.iloc[-1]['end_sec']
    print(f"Total duration based on .lab file: {total_duration:.2f} seconds")
    
    # Phoneme statistics
    print("\nPhoneme statistics:")
    phoneme_counts = lab_df['label'].value_counts()
    print(f"Unique phonemes: {len(phoneme_counts)}")
    print(phoneme_counts)

if __name__ == "__main__":
    main()