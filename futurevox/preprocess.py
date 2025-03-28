# preprocess.py
import yaml
import os
import h5py
import numpy as np

import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to avoid Qt dependency
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from utils.audio import extract_mel_spectrogram
import librosa
import librosa.display
from pathlib import Path

def load_config(config_path="config/default.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

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
            mel_spec, sr = extract_mel_spectrogram(wav_path, n_mels=config['audio']['n_mels'], n_fft=config['audio']['n_fft'], hop_length=config['audio']['hop_length'])
            hf.create_dataset(f'lab{index_id}', data=file)
            hf.create_dataset(f'mel{index_id}', data=mel_spec)
            hf.create_dataset(f'sr{index_id}', data=sr)
            index_id = index_id+1
    
    hf.create_dataset('pair_num', data=index_id)

    #print(f"Found {len(all_lab)} files.")
    #return all_lab
    

if __name__ == "__main__":
    main()