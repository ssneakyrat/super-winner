import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Rectangle
import torch
import librosa

# Set matplotlib to use Agg backend
matplotlib.use('Agg')

def read_config(config_path="config/default.yaml"):
    """Read the configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_phone_color(phone):
    """Get color for phoneme type."""
    vowels = ['iy', 'ih', 'eh', 'ae', 'aa', 'ah', 'ao', 'uh', 'uw', 'ux', 'er', 'ax', 'ix', 'axr', 'ey', 'ay', 'oy', 'aw', 'ow']
    nasals = ['m', 'n', 'ng', 'em', 'en', 'eng']
    fricatives = ['f', 'v', 'th', 'dh', 's', 'z', 'sh', 'zh', 'hh']
    stops = ['p', 'b', 't', 'd', 'k', 'g']
    affricates = ['ch', 'jh']
    liquids = ['l', 'r', 'el']
    glides = ['w', 'y']
    
    if phone in ['pau', 'sil', 'sp']:
        return '#999999'  # Silence/pause
    elif phone in vowels:
        return '#e74c3c'  # Vowels
    elif phone in nasals:
        return '#3498db'  # Nasals
    elif phone in fricatives:
        return '#2ecc71'  # Fricatives
    elif phone in stops:
        return '#f39c12'  # Stops
    elif phone in affricates:
        return '#9b59b6'  # Affricates
    elif phone in liquids:
        return '#1abc9c'  # Liquids
    elif phone in glides:
        return '#d35400'  # Glides
    else:
        return '#34495e'  # Others

def create_alignment_visualization(sample_id, mel_spec, f0_values, phones, start_times, end_times, config):
    """
    Create a visualization with mel spectrogram, F0, and phoneme alignment for TensorBoard.
    
    Args:
        sample_id: ID of the sample
        mel_spec: Mel spectrogram array of shape [n_mels, T]
        f0_values: F0 values array of shape [T]
        phones: List of phone symbols
        start_times: Array of start times for each phone
        end_times: Array of end times for each phone
        config: Configuration dictionary
        
    Returns:
        fig: Matplotlib figure
    """
    # Create durations array
    durations = end_times - start_times
    
    # Create F0 times array
    hop_length = config['audio']['hop_length']
    sample_rate = config['audio']['sample_rate']
    f0_times = np.arange(len(f0_values)) * hop_length / sample_rate
    
    # Create figure
    fig = plt.figure(figsize=(14, 10), dpi=100)
    
    # First subplot: Mel spectrogram
    ax1 = plt.subplot(3, 1, 1)
    img = librosa.display.specshow(
        mel_spec, 
        x_axis='time', 
        y_axis='mel', 
        sr=sample_rate, 
        hop_length=hop_length,
        fmin=config['audio']['fmin'],
        fmax=config['audio']['fmax']
    )
    ax1.set_title('Mel Spectrogram')
    
    # Second subplot: F0 contour
    ax2 = plt.subplot(3, 1, 2)
    ax2.plot(f0_times, f0_values, 'r-', linewidth=1.5, alpha=0.8)
    ax2.set_title('F0 Contour')
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_xlim(0, f0_times[-1] if len(f0_times) > 0 else 0)
    ax2.set_ylim(70, 400)  # Typical F0 range
    ax2.grid(True, alpha=0.3)
    
    # Third subplot: Phoneme alignment
    ax3 = plt.subplot(3, 1, 3)
    ax3.set_title('Phoneme Alignment')
    ax3.set_xlabel('Time (s)')
    ax3.set_xlim(0, f0_times[-1] if len(f0_times) > 0 else 0)
    ax3.set_ylim(0, 1)
    ax3.set_yticks([])
    
    # Add phoneme segments
    for i, (phone, start, end, duration) in enumerate(zip(phones, start_times, end_times, durations)):
        rect = Rectangle(
            (start, 0), 
            duration, 
            1, 
            facecolor=get_phone_color(phone), 
            edgecolor='black', 
            alpha=0.7
        )
        ax3.add_patch(rect)
        
        # Add phoneme text
        text_x = start + duration / 2
        ax3.text(
            text_x, 
            0.5, 
            phone, 
            horizontalalignment='center', 
            verticalalignment='center', 
            fontweight='bold',
            fontsize=9
        )
        
        # Add vertical alignment lines across all plots
        if i > 0:
            ax1.axvline(x=start, color='gray', linestyle='--', alpha=0.4)
            ax2.axvline(x=start, color='gray', linestyle='--', alpha=0.4)
    
    # Add sample ID
    plt.figtext(0.02, 0.01, f"Sample ID: {sample_id}", fontsize=8)
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    return fig

def create_checkpoint_dir(config):
    """Create checkpoint directory if it doesn't exist."""
    checkpoint_dir = config['training']['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir