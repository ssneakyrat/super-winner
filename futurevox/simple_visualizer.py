#!/usr/bin/env python3
# simple_visualizer.py

import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to avoid Qt dependency
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path

# Update these paths for your environment
LAB_FILE = "A3 Twinkle Twinkle - Part_1.lab"
WAV_FILE = None  # Will attempt to find automatically
DATA_DIR = "datasets/gin"  # From your config.yaml

def read_lab_file(lab_file):
    """Read a .lab file and return phoneme timings."""
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

def find_wav_file(lab_file, data_dir):
    """Try to find the corresponding WAV file."""
    lab_path = Path(lab_file)
    file_name = lab_path.stem
    
    # Check in data_dir/wav/
    wav_path = Path(data_dir) / "wav" / f"{file_name}.wav"
    if wav_path.exists():
        return str(wav_path)
    
    # Check in same directory
    wav_path = lab_path.parent / f"{file_name}.wav"
    if wav_path.exists():
        return str(wav_path)
    
    # Check by replacing "lab" with "wav" in path
    if "lab" in str(lab_path):
        wav_path = str(lab_path).replace("lab", "wav").replace(".lab", ".wav")
        if Path(wav_path).exists():
            return wav_path
    
    return None

def visualize_alignment():
    global WAV_FILE, LAB_FILE, DATA_DIR
    
    # Find WAV file if not specified
    if WAV_FILE is None:
        WAV_FILE = find_wav_file(LAB_FILE, DATA_DIR)
        if WAV_FILE is None:
            print(f"ERROR: Could not find WAV file for {LAB_FILE}")
            print("Please set WAV_FILE in the script")
            return
    
    print(f"Using WAV file: {WAV_FILE}")
    print(f"Using LAB file: {LAB_FILE}")
    
    # Load audio and extract mel spectrogram
    y, sr = librosa.load(WAV_FILE, sr=128)
    print(f"Audio duration: {librosa.get_duration(y=y, sr=sr):.2f} seconds")
    
    # Extract mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128
    )
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Read lab file
    lab_df = read_lab_file(Path(DATA_DIR) / 'lab' / LAB_FILE)
    
    # Convert lab timings to seconds (assuming samples)
    lab_df['start_sec'] = lab_df['start_time'] / sr
    lab_df['end_sec'] = lab_df['end_time'] / sr
    
    # Create visualization
    plt.figure(figsize=(20, 10))
    
    # Plot spectrogram
    librosa.display.specshow(
        log_mel_spec, sr=sr, hop_length=512, x_axis='time', y_axis='mel'
    )
    plt.colorbar(format='%+2.0f dB')
    
    # Add phoneme boundaries and labels
    ax = plt.gca()
    y_min, y_max = ax.get_ylim()
    
    for i, row in lab_df.iterrows():
        # Add vertical line at boundary
        plt.axvline(x=row['start_sec'], color='r', linestyle='--', alpha=0.7)
        
        # Add label
        label_x = (row['start_sec'] + row['end_sec']) / 2
        plt.text(label_x, y_max*0.9, row['label'], 
                 horizontalalignment='center',
                 verticalalignment='center',
                 fontsize=10,
                 bbox=dict(facecolor='white', alpha=0.7))
    
    # Add final boundary
    plt.axvline(x=lab_df.iloc[-1]['end_sec'], color='r', linestyle='--', alpha=0.7)
    plt.title(f"Alignment: {os.path.basename(WAV_FILE)} with {os.path.basename(LAB_FILE)}")
    plt.tight_layout()
    
    # Save or display
    output_file = f"{Path(LAB_FILE).stem}_alignment.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to: {output_file}")
    plt.show()

if __name__ == "__main__":
    visualize_alignment()