"""
Audio utilities for FutureVox.
Includes functions for audio processing and manipulation.
"""

import os
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend (avoid Qt dependency)
import matplotlib.pyplot as plt
import soundfile as sf
import torch
import torch.nn.functional as F
from scipy.io.wavfile import write
from typing import Optional, Union, Tuple

from config.model_config import DataConfig

def clean_mel_spectrogram(mel, replace_nan=True, min_value=-12, max_value=2):
    """Clean a mel spectrogram by removing NaNs and clipping extreme values."""
    if replace_nan:
        mel = np.nan_to_num(mel, nan=min_value, posinf=max_value, neginf=min_value)
    
    # Clip to reasonable range
    mel = np.clip(mel, min_value, max_value)
    
    return mel

def load_audio(
    file_path: str,
    sample_rate: int = 22050,
    mono: bool = True,
    normalize: bool = True
) -> np.ndarray:
    """
    Load audio file.
    
    Args:
        file_path: Path to audio file
        sample_rate: Target sample rate
        mono: Convert to mono
        normalize: Normalize audio
        
    Returns:
        Audio waveform as numpy array
    """
    audio, sr = librosa.load(
        file_path, sr=sample_rate, mono=mono
    )
    
    if normalize:
        audio = audio / np.max(np.abs(audio))
    
    return audio


def save_audio(
    audio: np.ndarray,
    file_path: str,
    sample_rate: int = 22050,
    normalize: bool = True
) -> None:
    """
    Save audio file.
    
    Args:
        audio: Audio waveform
        file_path: Output path
        sample_rate: Sample rate
        normalize: Normalize audio before saving
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    
    # Normalize if requested
    if normalize:
        audio = audio / np.max(np.abs(audio)) * 0.95
    
    # Save audio
    write(file_path, sample_rate, (audio * 32767).astype(np.int16))


def mel_to_audio(
    mel: np.ndarray,
    config: DataConfig,
    griffin_lim_iters: int = 60
) -> np.ndarray:
    """
    Convert mel spectrogram to audio using Griffin-Lim algorithm.
    
    Args:
        mel: Mel spectrogram
        config: Data configuration
        griffin_lim_iters: Number of Griffin-Lim iterations
        
    Returns:
        Audio waveform
    """
    # Create mel basis
    mel_basis = librosa.filters.mel(
        sr=config.sample_rate,
        n_fft=config.fft_size,
        n_mels=config.mel_channels,
        fmin=config.fmin,
        fmax=config.fmax
    )
    
    # Invert mel basis
    mel_inverse = np.linalg.pinv(mel_basis)
    
    # Convert to linear spectrogram
    spec = np.dot(mel_inverse, mel)
    
    # Ensure non-negativity
    spec = np.maximum(spec, 0)
    
    # Griffin-Lim algorithm
    audio = librosa.griffinlim(
        spec,
        n_iter=griffin_lim_iters,
        hop_length=config.hop_size,
        win_length=config.win_size
    )
    
    return audio


def synthesize_f0(
    f0: np.ndarray,
    sample_rate: int = 22050,
    hop_size: int = 256
) -> np.ndarray:
    """
    Synthesize audio from F0 contour (for visualization).
    
    Args:
        f0: F0 contour
        sample_rate: Sample rate
        hop_size: Hop size
        
    Returns:
        Synthesized audio
    """
    # Create time values
    t = np.arange(len(f0)) * hop_size / sample_rate
    
    # Create sin wave
    audio = np.zeros(len(f0) * hop_size)
    
    phase = 0
    for i, f in enumerate(f0):
        if f > 0:  # Only voiced frames
            # Number of samples in this frame
            n_samples = hop_size
            
            # Generate samples
            samples = np.arange(n_samples) / sample_rate
            
            # Phase continuous oscillator
            audio[i*hop_size:(i+1)*hop_size] = np.sin(
                2 * np.pi * f * samples + phase
            )
            
            # Update phase for next frame (for continuity)
            phase = (phase + 2 * np.pi * f * n_samples / sample_rate) % (2 * np.pi)
    
    return audio


def plot_spectrogram(
    spectrogram: np.ndarray,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot spectrogram.
    
    Args:
        spectrogram: Spectrogram to plot
        title: Plot title
        save_path: Path to save the plot
        show: Whether to show the plot
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    
    im = ax.imshow(
        spectrogram,
        aspect="auto",
        origin="lower",
        interpolation="none"
    )
    
    plt.colorbar(im, ax=ax)
    
    if title is not None:
        ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path)
    
    if show:
        plt.show()
    
    return fig


def plot_waveform(
    waveform: np.ndarray,
    sample_rate: int = 22050,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot waveform.
    
    Args:
        waveform: Audio waveform
        sample_rate: Sample rate
        title: Plot title
        save_path: Path to save the plot
        show: Whether to show the plot
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 3))
    
    librosa.display.waveshow(
        waveform,
        sr=sample_rate,
        ax=ax
    )
    
    if title is not None:
        ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path)
    
    if show:
        plt.show()
    
    return fig


def plot_f0(
    f0: np.ndarray,
    hop_size: int = 256,
    sample_rate: int = 22050,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot F0 contour.
    
    Args:
        f0: F0 contour
        hop_size: Hop size
        sample_rate: Sample rate
        title: Plot title
        save_path: Path to save the plot
        show: Whether to show the plot
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 3))
    
    # Create time values (in seconds)
    t = np.arange(len(f0)) * hop_size / sample_rate
    
    # Plot F0
    ax.plot(t, f0)
    
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("F0 (Hz)")
    
    if title is not None:
        ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path)
    
    if show:
        plt.show()
    
    return fig


def dynamic_range_compression(
    x: np.ndarray,
    c: float = 1.0,
    clip_val: float = 1e-5
) -> np.ndarray:
    """
    Dynamic range compression.
    
    Args:
        x: Input signal
        c: Compression factor
        clip_val: Clipping value
        
    Returns:
        Compressed signal
    """
    return np.log(np.maximum(clip_val, x)) * c


def dynamic_range_decompression(
    x: np.ndarray,
    c: float = 1.0
) -> np.ndarray:
    """
    Dynamic range decompression.
    
    Args:
        x: Compressed signal
        c: Compression factor
        
    Returns:
        Decompressed signal
    """
    return np.exp(x / c)


def spectrogram_to_mel(
    spectrogram: np.ndarray,
    config: DataConfig
) -> np.ndarray:
    """
    Convert linear spectrogram to mel spectrogram.
    
    Args:
        spectrogram: Linear spectrogram
        config: Data configuration
        
    Returns:
        Mel spectrogram
    """
    # Create mel basis
    mel_basis = librosa.filters.mel(
        sr=config.sample_rate,
        n_fft=config.fft_size,
        n_mels=config.mel_channels,
        fmin=config.fmin,
        fmax=config.fmax
    )
    
    # Convert to mel scale
    mel = np.dot(mel_basis, spectrogram)
    
    # Apply dynamic range compression
    mel = dynamic_range_compression(mel)
    
    return mel


def mel_to_spectrogram(
    mel: np.ndarray,
    config: DataConfig
) -> np.ndarray:
    """
    Convert mel spectrogram to linear spectrogram.
    
    Args:
        mel: Mel spectrogram
        config: Data configuration
        
    Returns:
        Linear spectrogram
    """
    # Apply dynamic range decompression
    mel = dynamic_range_decompression(mel)
    
    # Create mel basis
    mel_basis = librosa.filters.mel(
        sr=config.sample_rate,
        n_fft=config.fft_size,
        n_mels=config.mel_channels,
        fmin=config.fmin,
        fmax=config.fmax
    )
    
    # Invert mel basis
    mel_inverse = np.linalg.pinv(mel_basis)
    
    # Convert to linear spectrogram
    spectrogram = np.dot(mel_inverse, mel)
    
    # Ensure non-negativity
    spectrogram = np.maximum(spectrogram, 0)
    
    return spectrogram


def audio_to_mel(
    audio: np.ndarray,
    config: DataConfig,
    normalize: bool = True
) -> np.ndarray:
    """
    Convert audio to mel spectrogram.
    
    Args:
        audio: Audio waveform
        config: Data configuration
        normalize: Whether to normalize the mel spectrogram
        
    Returns:
        Mel spectrogram
    """
    # Short-time Fourier transform
    D = librosa.stft(
        audio,
        n_fft=config.fft_size,
        hop_length=config.hop_size,
        win_length=config.win_size
    )
    
    # Magnitude spectrogram
    spectrogram = np.abs(D)
    
    # Convert to mel scale
    mel = spectrogram_to_mel(spectrogram, config)
    
    # Normalize
    if normalize:
        mel = (mel - mel.mean()) / (mel.std() + 1e-8)
    
    return mel


def audio_to_mel_torch(
    audio: torch.Tensor,
    config: DataConfig,
    normalize: bool = True
) -> torch.Tensor:
    """
    Convert audio to mel spectrogram using PyTorch.
    
    Args:
        audio: Audio waveform [B, T]
        config: Data configuration
        normalize: Whether to normalize the mel spectrogram
        
    Returns:
        Mel spectrogram [B, n_mels, T']
    """
    # Check dimensions
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    
    # STFT
    stft = torch.stft(
        audio,
        n_fft=config.fft_size,
        hop_length=config.hop_size,
        win_length=config.win_size,
        window=torch.hann_window(config.win_size).to(audio.device),
        return_complex=True
    )
    
    # Magnitude spectrogram
    spectrogram = torch.abs(stft)
    
    # Create mel filterbank
    mel_basis = librosa.filters.mel(
        sr=config.sample_rate,
        n_fft=config.fft_size,
        n_mels=config.mel_channels,
        fmin=config.fmin,
        fmax=config.fmax
    )
    mel_basis = torch.from_numpy(mel_basis).to(audio.device)
    
    # Convert to mel scale
    mel = torch.matmul(mel_basis, spectrogram)
    
    # Dynamic range compression
    mel = torch.log(torch.clamp(mel, min=1e-5))
    
    # Normalize
    if normalize:
        mel = (mel - torch.mean(mel, dim=(1, 2), keepdim=True)) / (torch.std(mel, dim=(1, 2), keepdim=True) + 1e-8)
    
    return mel


def extract_loudness(
    audio: np.ndarray,
    sample_rate: int = 22050,
    hop_size: int = 256,
    frame_length: int = 1024,
    ref_level: float = 20.0
) -> np.ndarray:
    """
    Extract loudness feature from audio.
    
    Args:
        audio: Audio waveform
        sample_rate: Sample rate
        hop_size: Hop size
        frame_length: Frame length
        ref_level: Reference level in dB
        
    Returns:
        Loudness contour
    """
    # Extract RMS energy
    rms = librosa.feature.rms(
        y=audio,
        frame_length=frame_length,
        hop_length=hop_size
    )[0]
    
    # Convert to dB scale
    loudness = 20 * np.log10(np.maximum(1e-5, rms))
    
    # Normalize
    loudness = (loudness - ref_level) / ref_level
    
    return loudness


def preemphasis(
    audio: np.ndarray,
    coef: float = 0.97
) -> np.ndarray:
    """
    Apply preemphasis filter to audio.
    
    Args:
        audio: Audio waveform
        coef: Preemphasis coefficient
        
    Returns:
        Filtered audio
    """
    return np.append(audio[0], audio[1:] - coef * audio[:-1])


def deemphasis(
    audio: np.ndarray,
    coef: float = 0.97
) -> np.ndarray:
    """
    Apply deemphasis filter to audio.
    
    Args:
        audio: Audio waveform
        coef: Preemphasis coefficient
        
    Returns:
        Filtered audio
    """
    result = np.zeros_like(audio)
    result[0] = audio[0]
    for i in range(1, len(audio)):
        result[i] = audio[i] + coef * result[i-1]
    return result