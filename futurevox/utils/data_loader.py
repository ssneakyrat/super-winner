# utils/data_loader.py
import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to avoid Qt dependency
import matplotlib.pyplot as plt

def load_sample_from_h5(h5_file_path, sample_index):
    """
    Load a single sample from an HDF5 file.
    
    Args:
        h5_file_path: Path to the HDF5 file
        sample_index: Index of the sample to load
        
    Returns:
        A dictionary containing:
        - file_name: Original file name
        - mel_spectrogram: The mel spectrogram array
        - sample_rate: Audio sample rate
        - phonemes: DataFrame of phoneme timings and labels
    """
    with h5py.File(h5_file_path, 'r') as hf:
        # Check if the sample exists
        sample_key = f'sample{sample_index}'
        if sample_key not in hf:
            raise ValueError(f"Sample index {sample_index} not found in {h5_file_path}")
        
        # Get the sample group
        sample = hf[sample_key]
        
        # Extract file info
        file_name = sample['file_name'][()]
        if isinstance(file_name, bytes):
            file_name = file_name.decode('utf-8')
        
        # Extract mel spectrogram and sample rate
        mel_spec = sample['mel_spectrogram'][()]
        sr = sample['sample_rate'][()]
        
        # Extract phoneme data
        phoneme_data = []
        phoneme_count = sample['phoneme_count'][()]
        
        for i in range(phoneme_count):
            phoneme = sample['phonemes'][f'phoneme{i}']
            
            # Get label, converting from bytes to string if needed
            label = phoneme['label'][()]
            if isinstance(label, bytes):
                label = label.decode('utf-8')
                
            phoneme_data.append({
                'start_time': phoneme['start_time'][()],
                'end_time': phoneme['end_time'][()],
                'label': label
            })
        
        # Create a DataFrame for the phonemes
        phonemes_df = pd.DataFrame(phoneme_data)
        
        return {
            'file_name': file_name,
            'mel_spectrogram': mel_spec,
            'sample_rate': sr,
            'phonemes': phonemes_df
        }

def list_samples_in_h5(h5_file_path):
    """
    List all samples in an HDF5 file.
    
    Args:
        h5_file_path: Path to the HDF5 file
        
    Returns:
        A DataFrame with sample indices and file names
    """
    samples = []
    
    with h5py.File(h5_file_path, 'r') as hf:
        sample_count = hf['sample_count'][()]
        
        for i in range(sample_count):
            sample_key = f'sample{i}'
            
            # Get file name
            file_name = hf[sample_key]['file_name'][()]
            if isinstance(file_name, bytes):
                file_name = file_name.decode('utf-8')
                
            # Get phoneme count
            phoneme_count = hf[sample_key]['phoneme_count'][()]
            
            samples.append({
                'index': i,
                'file_name': file_name,
                'phoneme_count': phoneme_count
            })
    
    return pd.DataFrame(samples)

def extract_aligned_features(h5_file_path, sample_index, feature_type='mel_frame_to_phoneme'):
    """
    Extract aligned features from the HDF5 file.
    
    Args:
        h5_file_path: Path to the HDF5 file
        sample_index: Index of the sample to process
        feature_type: Type of alignment to extract
            - 'mel_frame_to_phoneme': Map each mel frame to a phoneme
            
    Returns:
        Depends on feature_type:
        - For 'mel_frame_to_phoneme': (mel_spec, phoneme_indices)
            where phoneme_indices maps each frame to the corresponding phoneme index
    """
    # Load the sample
    sample = load_sample_from_h5(h5_file_path, sample_index)
    mel_spec = sample['mel_spectrogram']
    sr = sample['sample_rate']
    phonemes = sample['phonemes']
    
    if feature_type == 'mel_frame_to_phoneme':
        # Get hop length from config or use default
        hop_length = 256  # Default, should match your extraction settings
        
        # Calculate time for each mel frame
        n_frames = mel_spec.shape[1]
        frame_times = np.arange(n_frames) * hop_length / sr
        
        # Map each frame to a phoneme
        phoneme_indices = np.full(n_frames, -1)  # Default to -1 (no phoneme)
        
        for i, phoneme in phonemes.iterrows():
            start_sec = phoneme['start_time'] / sr
            end_sec = phoneme['end_time'] / sr
            
            # Find frames that fall within this phoneme's time range
            mask = (frame_times >= start_sec) & (frame_times < end_sec)
            phoneme_indices[mask] = i
        
        return mel_spec, phoneme_indices, phonemes
    
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")

def visualize_sample_from_h5(h5_file_path, sample_index):
    """
    Visualize a sample from the HDF5 file, showing mel spectrogram and phoneme boundaries.
    """
    # Load the aligned features
    mel_spec, phoneme_indices, phonemes_df = extract_aligned_features(
        h5_file_path, sample_index, feature_type='mel_frame_to_phoneme'
    )
    
    # Also get the raw sample data for additional information
    sample = load_sample_from_h5(h5_file_path, sample_index)
    sr = sample['sample_rate']
    
    # Create the visualization
    plt.figure(figsize=(20, 10))
    
    # Plot the mel spectrogram
    librosa.display.specshow(
        mel_spec, 
        sr=sr, 
        hop_length=256,  # Should match your extraction settings
        x_axis='time', 
        y_axis='mel'
    )
    plt.colorbar(format='%+2.0f dB')
    
    # Get axis limits
    ax = plt.gca()
    y_min, y_max = ax.get_ylim()
    
    # Add phoneme boundaries and labels
    for i, phoneme in phonemes_df.iterrows():
        # Convert time to seconds
        start_sec = phoneme['start_time'] / sr
        end_sec = phoneme['end_time'] / sr
        
        # Add vertical line at boundary
        plt.axvline(x=start_sec, color='r', linestyle='--', alpha=0.7)
        
        # Add label text
        label_x = (start_sec + end_sec) / 2
        plt.text(label_x, y_max*0.9, phoneme['label'], 
                 horizontalalignment='center', 
                 verticalalignment='center',
                 fontsize=10, 
                 bbox=dict(facecolor='white', alpha=0.7))
    
    # Add final boundary
    plt.axvline(x=phonemes_df.iloc[-1]['end_time'] / sr, color='r', linestyle='--', alpha=0.7)
    
    plt.title(f"Sample: {sample['file_name']}")
    plt.tight_layout()
    plt.show()