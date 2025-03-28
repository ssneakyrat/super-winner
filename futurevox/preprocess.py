# preprocess.py
import yaml
import os
import h5py
import numpy as np
import librosa
import matplotlib.pyplot as plt

from pathlib import Path

def load_config(config_path="config/default.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def extract_mel_spectrogram(wav_file, n_mels=128, n_fft=2048, hop_length=512):
    # Load the audio file
    y, sr = librosa.load(wav_file, sr=None)
    
    # Extract mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(
        y=y, 
        sr=sr, 
        n_fft=n_fft, 
        hop_length=hop_length, 
        n_mels=n_mels
    )
    
    # Convert to log scale (dB)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    return log_mel_spectrogram, sr

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
    all_lab = []
    all_mel = []
    for root, dirs, files in os.walk(data_raw_dir + '/lab'):
        for file in files:
            file_path = os.path.join(root, file)
            file_name = os.path.splitext(file)[0]
            file_name = os.path.basename(os.path.normpath(file_name))
            wav_path = data_raw_dir + '/wav/' + file_name + '.wav'
            all_mel.append(wav_path)
            all_lab.append(file_path)
    
    hf.create_dataset('lab', data=all_lab)
    hf.create_dataset('mel', data=all_mel)

    # Extract Mel spectrogram
    mel_spec, sr = extract_mel_spectrogram(all_mel[0])
    plot_mel_spectrogram(mel_spec, sr, 512, output_file='test.png' )

    print(f"Found {len(all_lab)} files.")
    return all_lab
    

if __name__ == "__main__":
    main()