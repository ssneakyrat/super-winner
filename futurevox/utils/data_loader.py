# utils/data_loader.py
import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to avoid Qt dependency
import matplotlib.pyplot as plt
import librosa
import librosa.display

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
        - f0: Fundamental frequency (if available)
        - voiced_flag: Boolean array indicating voiced frames (if available)
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
        
        # Extract F0 data if available
        f0 = None
        voiced_flag = None
        if 'f0' in sample:
            f0 = sample['f0'][()]
            if 'voiced_flag' in sample:
                voiced_flag = sample['voiced_flag'][()]
        
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
            'phonemes': phonemes_df,
            'f0': f0,
            'voiced_flag': voiced_flag
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
            
            # Check if F0 is available
            has_f0 = 'f0' in hf[sample_key]
            
            samples.append({
                'index': i,
                'file_name': file_name,
                'phoneme_count': phoneme_count,
                'has_f0': has_f0
            })
    
    return pd.DataFrame(samples)