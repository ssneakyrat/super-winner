import librosa
import numpy as np

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