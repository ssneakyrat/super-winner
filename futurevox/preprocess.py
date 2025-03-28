import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
import h5py
import yaml
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # This must be done before importing pyplot
from matplotlib.patches import Rectangle

def read_config(config_path="futurevox/config/default.yaml"):
    """Read the configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def read_lab_file(lab_path):
    """Read a label file and return the phoneme segments."""
    phonemes = []
    with open(lab_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                start, end, phone = int(parts[0]), int(parts[1]), parts[2]
                phonemes.append({'start': start, 'end': end, 'phone': phone})
    return phonemes

def extract_f0(audio, sample_rate, min_f0=70, max_f0=400, frame_length=1024, hop_length=256):
    """
    Extract fundamental frequency (F0) using librosa's pyin algorithm.
    Returns times and f0 values.
    """
    # Use PYIN algorithm for F0 extraction
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio,
        fmin=min_f0,
        fmax=max_f0,
        sr=sample_rate,
        frame_length=frame_length,
        hop_length=hop_length
    )
    
    # Get corresponding time values
    times = librosa.times_like(f0, sr=sample_rate, hop_length=hop_length)
    
    return times, f0

def extract_mel_spectrogram(audio, config):
    """Extract mel spectrogram based on config parameters."""
    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio,
        sr=config['audio']['sample_rate'],
        n_fft=config['audio']['n_fft'],
        hop_length=config['audio']['hop_length'],
        n_mels=config['audio']['n_mels'],
        fmin=config['audio']['fmin'],
        fmax=config['audio']['fmax']
    )
    # Convert to log scale (dB)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return log_mel_spectrogram

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

def phonemes_to_frames(phonemes, lab_sample_rate, hop_length, sample_rate, scaling_factor=227.13):
    """Convert phoneme timings to frame indices for the mel spectrogram."""
    phoneme_frames = []
    for p in phonemes:
        # Convert from lab time units to seconds
        start_time = p['start'] / lab_sample_rate / scaling_factor
        end_time = p['end'] / lab_sample_rate / scaling_factor
        
        # Convert from seconds to frame indices
        start_frame = int(start_time * sample_rate / hop_length)
        end_frame = int(end_time * sample_rate / hop_length)
        
        duration = end_time - start_time
        
        phoneme_frames.append({
            'start_frame': start_frame,
            'end_frame': end_frame,
            'start_time': start_time,
            'end_time': end_time,
            'duration': duration,
            'phone': p['phone']
        })
    return phoneme_frames

def create_visualization(sample_id, h5_path, output_path, config):
    """Create visualization with mel spectrogram, F0, and phoneme alignment from the HDF5 file."""
    with h5py.File(h5_path, 'r') as f:
        # Get the sample group
        sample_group = f[sample_id]
        
        # Extract data
        mel_spec = sample_group['features']['mel_spectrogram'][:]
        f0_times = sample_group['features']['f0_times'][:]
        f0_values = sample_group['features']['f0_values'][:]
        
        # Get phoneme data
        phones_bytes = sample_group['phonemes']['phones'][:]
        phones = [p.decode('utf-8') for p in phones_bytes]
        start_times = sample_group['phonemes']['start_times'][:]
        end_times = sample_group['phonemes']['end_times'][:]
        durations = sample_group['phonemes']['durations'][:]
        
        # Recreate phoneme frames for visualization
        phoneme_frames = []
        for i, phone in enumerate(phones):
            phoneme_frames.append({
                'phone': phone,
                'start_time': start_times[i],
                'end_time': end_times[i],
                'duration': durations[i]
            })

    plt.figure(figsize=(14, 10), dpi=100)
    
    # First subplot: Mel spectrogram
    ax1 = plt.subplot(3, 1, 1)
    img = librosa.display.specshow(
        mel_spec, 
        x_axis='time', 
        y_axis='mel', 
        sr=config['audio']['sample_rate'], 
        hop_length=config['audio']['hop_length'],
        fmin=config['audio']['fmin'],
        fmax=config['audio']['fmax']
    )
    #plt.colorbar(img, format='%+2.0f dB')
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
    for i, p in enumerate(phoneme_frames):
        rect = Rectangle(
            (p['start_time'], 0), 
            p['duration'], 
            1, 
            facecolor=get_phone_color(p['phone']), 
            edgecolor='black', 
            alpha=0.7
        )
        ax3.add_patch(rect)
        
        # Add phoneme text
        text_x = p['start_time'] + p['duration'] / 2
        ax3.text(
            text_x, 
            0.5, 
            p['phone'], 
            horizontalalignment='center', 
            verticalalignment='center', 
            fontweight='bold',
            fontsize=9
        )
        
        # Add vertical alignment lines across all plots
        if i > 0:
            ax1.axvline(x=p['start_time'], color='gray', linestyle='--', alpha=0.4)
            ax2.axvline(x=p['start_time'], color='gray', linestyle='--', alpha=0.4)
    
    # Add phoneme duration table
    table_text = "Phoneme durations (seconds):\n"
    table_text += "------------------------\n"
    table_text += "Phone | Start  | End    | Duration\n"
    table_text += "------+--------+--------+----------\n"
    
    for p in phoneme_frames:
        table_text += f"{p['phone']:<6}| {p['start_time']:.3f} | {p['end_time']:.3f} | {p['duration']:.3f}\n"
    
    plt.figtext(0.5, 0.01, table_text, fontsize=7, family='monospace')
    
    # Add sample ID
    plt.figtext(0.02, 0.01, f"Sample ID: {sample_id}", fontsize=8)
    
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")
    
    return plt.gcf()

def process_files(config_path="futurevox/config/default.yaml", lab_sample_rate=44100, scaling_factor=227.13):
    """Process WAV and LAB files, extract features, and save to a single HDF5 file."""
    # Read config
    config = read_config(config_path)
    
    # Get paths
    data_raw_path = config['datasets']['data_raw']
    wav_dir = os.path.join(data_raw_path, "wav")
    lab_dir = os.path.join(data_raw_path, "lab")
    binary_dir = os.path.join(data_raw_path, "binary")
    
    # Create binary directory if it doesn't exist
    os.makedirs(binary_dir, exist_ok=True)
    
    # Define the path for the single HDF5 file
    h5_path = os.path.join(binary_dir, "dataset.h5")
    
    # Get list of WAV files
    wav_files = [f for f in os.listdir(wav_dir) if f.endswith('.wav')]
    
    if not wav_files:
        print(f"No WAV files found in {wav_dir}")
        return None
    
    # Create the HDF5 file
    with h5py.File(h5_path, 'w') as f:
        # Store metadata and configuration
        metadata_group = f.create_group('metadata')
        for key, value in config['audio'].items():
            metadata_group.attrs[key] = value
        metadata_group.attrs['lab_sample_rate'] = lab_sample_rate
        metadata_group.attrs['scaling_factor'] = scaling_factor
        
        # Store file list
        file_list = np.array([os.path.splitext(wav)[0] for wav in wav_files], dtype='S100')
        metadata_group.create_dataset('file_list', data=file_list)
        
        # Process each WAV file
        sample_id_for_visualization = None
        
        for i, wav_file in enumerate(wav_files):
            # Get corresponding lab file
            base_name = os.path.splitext(wav_file)[0]
            lab_file = base_name + '.lab'
            lab_path = os.path.join(lab_dir, lab_file)
            
            # Skip if lab file doesn't exist
            if not os.path.exists(lab_path):
                print(f"Warning: No matching lab file for {wav_file}")
                continue
                
            wav_path = os.path.join(wav_dir, wav_file)
            
            try:
                print(f"Processing {wav_file} ({i+1}/{len(wav_files)})")
                
                # Store the first valid sample ID for visualization
                if sample_id_for_visualization is None:
                    sample_id_for_visualization = base_name
                
                # Create a group for this sample
                sample_group = f.create_group(base_name)
                
                # Load audio
                audio, sample_rate = librosa.load(wav_path, sr=config['audio']['sample_rate'])
                
                # Extract features
                mel_spec = extract_mel_spectrogram(audio, config)
                f0_times, f0_values = extract_f0(
                    audio, 
                    sample_rate, 
                    frame_length=config['audio']['n_fft'], 
                    hop_length=config['audio']['hop_length']
                )
                
                # Load phonemes
                phonemes = read_lab_file(lab_path)
                
                # Convert phoneme timings to frame indices
                phoneme_frames = phonemes_to_frames(
                    phonemes, 
                    lab_sample_rate, 
                    config['audio']['hop_length'],
                    sample_rate,
                    scaling_factor
                )
                
                # Create subgroups
                audio_group = sample_group.create_group('audio')
                feature_group = sample_group.create_group('features')
                phoneme_group = sample_group.create_group('phonemes')
                
                # Store audio data
                audio_group.create_dataset('waveform', data=audio)
                
                # Store features
                feature_group.create_dataset('mel_spectrogram', data=mel_spec)
                feature_group.create_dataset('f0_times', data=f0_times)
                feature_group.create_dataset('f0_values', data=f0_values)
                
                # Store phoneme data
                phones = np.array([p['phone'] for p in phoneme_frames], dtype='S10')
                start_frames = np.array([p['start_frame'] for p in phoneme_frames])
                end_frames = np.array([p['end_frame'] for p in phoneme_frames])
                start_times = np.array([p['start_time'] for p in phoneme_frames])
                end_times = np.array([p['end_time'] for p in phoneme_frames])
                durations = np.array([p['duration'] for p in phoneme_frames])
                
                phoneme_group.create_dataset('phones', data=phones)
                phoneme_group.create_dataset('start_frames', data=start_frames)
                phoneme_group.create_dataset('end_frames', data=end_frames)
                phoneme_group.create_dataset('start_times', data=start_times)
                phoneme_group.create_dataset('end_times', data=end_times)
                phoneme_group.create_dataset('durations', data=durations)
                
                # Store sample metadata
                sample_group.attrs['filename'] = wav_file
                sample_group.attrs['duration'] = len(audio) / sample_rate
                sample_group.attrs['num_frames'] = mel_spec.shape[1]
                sample_group.attrs['num_phonemes'] = len(phones)
                
            except Exception as e:
                print(f"Error processing {wav_file}: {str(e)}")
    
    print(f"All samples processed and saved to {h5_path}")
    return h5_path, sample_id_for_visualization

def validate_dataset(h5_path):
    """Validate the entire dataset by checking sample integrity."""
    print(f"Validating dataset: {h5_path}")
    
    with h5py.File(h5_path, 'r') as f:
        # Get dataset metadata
        metadata = f['metadata']
        file_list_bytes = metadata['file_list'][:]
        file_list = [name.decode('utf-8') for name in file_list_bytes]
        
        print(f"\nFound {len(file_list)} samples in the dataset")
        print(f"Audio configuration: {dict(metadata.attrs.items())}")
        
        # Validate each sample
        valid_samples = 0
        issues = 0
        
        print("\nSample validation:")
        for i, sample_id in enumerate(file_list):
            if sample_id not in f:
                print(f"  - Warning: Sample '{sample_id}' in file list but not in dataset")
                issues += 1
                continue
            
            sample = f[sample_id]
            
            # Check required groups and datasets
            required_groups = ['audio', 'features', 'phonemes']
            missing_groups = [g for g in required_groups if g not in sample]
            
            if missing_groups:
                print(f"  - Warning: Sample '{sample_id}' missing groups: {missing_groups}")
                issues += 1
                continue
            
            # Check mel spectrogram and F0 alignment
            mel_spec = sample['features']['mel_spectrogram'][:]
            f0_values = sample['features']['f0_values'][:]
            
            if len(f0_values) != mel_spec.shape[1]:
                print(f"  - Warning: Sample '{sample_id}' has misaligned F0 ({len(f0_values)}) and mel spectrogram ({mel_spec.shape[1]})")
                issues += 1
            
            # Check phoneme frame ranges
            phoneme_end_frames = sample['phonemes']['end_frames'][:]
            if len(phoneme_end_frames) > 0 and max(phoneme_end_frames) > mel_spec.shape[1]:
                print(f"  - Warning: Sample '{sample_id}' has phoneme frames exceeding spectrogram length")
                issues += 1
            
            valid_samples += 1
        
        # Print summary statistics
        print(f"\nValidation Summary:")
        print(f"  - Valid samples: {valid_samples}/{len(file_list)} ({valid_samples/len(file_list)*100:.1f}%)")
        print(f"  - Issues found: {issues}")
        
        # Calculate dataset statistics
        print("\nDataset Statistics:")
        total_duration = 0
        total_phonemes = 0
        phoneme_counts = {}
        
        for sample_id in file_list:
            if sample_id in f:
                sample = f[sample_id]
                total_duration += sample.attrs.get('duration', 0)
                
                if 'phonemes' in sample and 'phones' in sample['phonemes']:
                    phones = sample['phonemes']['phones'][:]
                    total_phonemes += len(phones)
                    
                    # Count phoneme occurrences
                    for phone_bytes in phones:
                        phone = phone_bytes.decode('utf-8')
                        phoneme_counts[phone] = phoneme_counts.get(phone, 0) + 1
        
        print(f"  - Total audio duration: {total_duration:.2f} seconds")
        print(f"  - Total phonemes: {total_phonemes}")
        print(f"  - Average phonemes per second: {total_phonemes/total_duration:.2f}")
        
        # Print top 10 most common phonemes
        top_phonemes = sorted(phoneme_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        print("\nTop 10 most common phonemes:")
        for phone, count in top_phonemes:
            print(f"  - {phone}: {count} occurrences ({count/total_phonemes*100:.1f}%)")
        
        return valid_samples, issues

def main():
    print("Starting audio data processing...")
    
    config_path = "config/default.yaml"
    config = read_config(config_path)
    data_raw_path = config['datasets']['data_raw']
    binary_dir = os.path.join(data_raw_path, "binary")
    
    # Process files and create single HDF5 file
    h5_path, sample_id = process_files(config_path)
    
    if h5_path and sample_id:
        # Validate the dataset
        validate_dataset(h5_path)
        
        # Create visualization for the first sample
        vis_output_path = os.path.join(binary_dir, f"{sample_id}_visualization.png")
        create_visualization(sample_id, h5_path, vis_output_path, config)
    
    print("Processing complete!")

if __name__ == '__main__':
    main()