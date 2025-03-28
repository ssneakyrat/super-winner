# preprocess.py
import yaml
import os
import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to avoid Qt dependency
import matplotlib.pyplot as plt
import librosa
import librosa.display
from pathlib import Path
from utils.audio import extract_mel_spectrogram, extract_f0

def load_config(config_path="config/default.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

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
    return data

def plot_mel_spectrogram(mel_spectrogram, sr, hop_length, output_file=None, title="Mel Spectrogram"):
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(
        mel_spectrogram, 
        sr=sr, 
        hop_length=hop_length, 
        x_axis='time', 
        y_axis='mel'
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
        print(f"Plot saved to {output_file}")
    else:
        plt.show()

def main():
    # Load configuration
    config = load_config()
    
    # Read all lab file
    data_raw_dir = config['datasets']['data_raw']

    # create binary file
    os.makedirs(data_raw_dir + '/binary', exist_ok=True)
    singerName = os.path.basename(os.path.normpath(data_raw_dir))
    hf = h5py.File(data_raw_dir + '/binary/' + singerName + '.h5', 'w')
    hf.create_dataset('name', data=singerName)

    # List all files
    index_id = 0
    for root, dirs, files in os.walk(data_raw_dir + '/lab'):
        for file in files:
            file_path = os.path.join(root, file)
            file_name = os.path.splitext(file)[0]
            file_name = os.path.basename(os.path.normpath(file_name))
            wav_path = data_raw_dir + '/wav/' + file_name + '.wav'
            
            # Skip if WAV file doesn't exist
            if not os.path.exists(wav_path):
                print(f"Warning: WAV file not found for {file_name}, skipping")
                continue
            
            # Extract mel spectrogram
            mel_spec, sr = extract_mel_spectrogram(
                wav_path, 
                n_mels=config['audio']['n_mels'], 
                n_fft=config['audio']['n_fft'], 
                hop_length=config['audio']['hop_length']
            )
            
            # Extract F0
            f0, voiced_flag, _ = extract_f0(
                wav_path,
                hop_length=config['audio']['hop_length'],
                fmin=50,  # Typical range for singing voice
                fmax=1000
            )
            
            # Check alignment of mel spectrogram and F0
            if len(f0) != mel_spec.shape[1]:
                print(f"Warning: F0 frames ({len(f0)}) don't match mel spectrogram frames ({mel_spec.shape[1]}) for {file_name}")
                min_frames = min(len(f0), mel_spec.shape[1])
                f0 = f0[:min_frames]
                voiced_flag = voiced_flag[:min_frames]
                mel_spec = mel_spec[:, :min_frames]
            
            # Read phoneme labels from lab file
            phoneme_data = read_lab_file(file_path)
            
            # Create a group for this sample
            sample_group = hf.create_group(f'sample{index_id}')
            
            # Store file information
            sample_group.create_dataset('file_name', data=file_name)
            sample_group.create_dataset('lab_file', data=file)
            
            # Store mel spectrogram
            sample_group.create_dataset('mel_spectrogram', data=mel_spec)
            sample_group.create_dataset('sample_rate', data=sr)
            
            # Store F0 data
            sample_group.create_dataset('f0', data=f0)
            sample_group.create_dataset('voiced_flag', data=voiced_flag)
            
            # Store phoneme data
            phoneme_group = sample_group.create_group('phonemes')
            for i, phoneme in enumerate(phoneme_data):
                phoneme_item = phoneme_group.create_group(f'phoneme{i}')
                phoneme_item.create_dataset('start_time', data=phoneme['start_time'])
                phoneme_item.create_dataset('end_time', data=phoneme['end_time'])
                phoneme_item.create_dataset('label', data=phoneme['label'])
            
            # Store number of phonemes
            sample_group.create_dataset('phoneme_count', data=len(phoneme_data))
            
            print(f"Processed {file_name}: {len(phoneme_data)} phonemes")
            index_id += 1
    
    # Store total number of samples
    hf.create_dataset('sample_count', data=index_id)
    
    print(f"Successfully processed {index_id} samples and stored in {singerName}.h5")
    hf.close()

if __name__ == "__main__":
    main()