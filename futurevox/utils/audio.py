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

def extract_f0(wav_file, hop_length=512, fmin=50, fmax=1000):
    """
    Extract the fundamental frequency (F0) from an audio file.
    
    Args:
        wav_file: Path to the audio file
        hop_length: Hop length for frame-wise analysis
        fmin: Minimum frequency for pitch tracking
        fmax: Maximum frequency for pitch tracking
        
    Returns:
        f0: Fundamental frequency (F0) over time
        voiced_flag: Boolean array indicating whether each frame is voiced
        sr: Sample rate of the audio
    """
    # Load the audio file
    y, sr = librosa.load(wav_file, sr=None)
    
    # Compute the pitch (F0) using pYIN algorithm
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y=y, 
        sr=sr, 
        hop_length=hop_length, 
        fmin=fmin, 
        fmax=fmax
    )
    
    # Replace NaN values (unvoiced frames) with 0
    f0 = np.nan_to_num(f0)
    
    return f0, voiced_flag, sr