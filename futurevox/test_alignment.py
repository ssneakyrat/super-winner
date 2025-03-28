import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
import argparse
import matplotlib
matplotlib.use('Agg')  # This must be done before importing pyplot
from matplotlib.patches import Rectangle
from pathlib import Path

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

def extract_f0(audio, sample_rate, min_f0=70, max_f0=400, frame_length=1024, hop_length=256):
    """
    Extract fundamental frequency (F0) using librosa's pyin algorithm.
    Returns times and f0 values.
    """
    # Use PYIN algorithm for F0 extraction (more reliable than YIN)
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

def create_visualization(wav_path, lab_path, output_path, min_f0=70, max_f0=400, lab_sample_rate=44100, manual_scaling=None):
    """Create a visualization of waveform, F0 and phoneme alignment."""
    # Load audio
    audio, sample_rate = librosa.load(wav_path, sr=None)
    audio_duration = librosa.get_duration(y=audio, sr=sample_rate)
    
    # Load phonemes
    phonemes = read_lab_file(lab_path)
    
    # Calculate raw label duration
    raw_label_duration = phonemes[-1]['end'] / lab_sample_rate
    
    # Calculate scaling factor
    if manual_scaling:
        scaling_factor = manual_scaling
    else:
        scaling_factor = raw_label_duration / audio_duration
    
    # Print scaling information
    print(f"Audio duration: {audio_duration:.2f} seconds")
    print(f"Raw label duration: {raw_label_duration:.2f} seconds")
    print(f"Scaling factor: {scaling_factor:.2f}")
    
    # Convert phoneme timings to seconds
    phonemes_in_seconds = []
    for p in phonemes:
        start_sec = p['start'] / lab_sample_rate / scaling_factor
        end_sec = p['end'] / lab_sample_rate / scaling_factor
        duration_sec = end_sec - start_sec
        phonemes_in_seconds.append({
            'start': start_sec,
            'end': end_sec,
            'duration': duration_sec,
            'phone': p['phone']
        })
    
    # Extract F0
    f0_times, f0_values = extract_f0(audio, sample_rate, min_f0, max_f0)
    
    # Create visualization figure with higher resolution for better clarity
    plt.figure(figsize=(14, 10), dpi=100)
    
    # First subplot: Waveform
    ax1 = plt.subplot(3, 1, 1)
    librosa.display.waveshow(audio, sr=sample_rate, alpha=0.8)
    ax1.set_title('Waveform')
    ax1.set_xlim(0, audio_duration)
    
    # Second subplot: F0 contour
    ax2 = plt.subplot(3, 1, 2)
    ax2.plot(f0_times, f0_values, 'r-', linewidth=1.5, alpha=0.8)
    ax2.set_title('F0 Contour')
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_xlim(0, audio_duration)
    ax2.set_ylim(min_f0, max_f0)
    ax2.grid(True, alpha=0.3)
    
    # Third subplot: Phoneme alignment
    ax3 = plt.subplot(3, 1, 3)
    ax3.set_title('Phoneme Alignment')
    ax3.set_xlabel('Time (s)')
    ax3.set_xlim(0, audio_duration)
    ax3.set_ylim(0, 1)
    ax3.set_yticks([])
    
    # Add phoneme segments
    for i, p in enumerate(phonemes_in_seconds):
        rect = Rectangle((p['start'], 0), p['duration'], 1, 
                        facecolor=get_phone_color(p['phone']), 
                        edgecolor='black', alpha=0.7)
        ax3.add_patch(rect)
        
        # Add phoneme text
        text_x = p['start'] + p['duration'] / 2
        ax3.text(text_x, 0.5, p['phone'], 
                horizontalalignment='center', 
                verticalalignment='center', 
                fontweight='bold',
                fontsize=9)
        
        # Add vertical alignment lines across all plots
        if i > 0:
            ax1.axvline(x=p['start'], color='gray', linestyle='--', alpha=0.4)
            ax2.axvline(x=p['start'], color='gray', linestyle='--', alpha=0.4)
    
    # Add a phoneme duration table
    table_text = "Phoneme durations (seconds):\n"
    table_text += "------------------------\n"
    table_text += "Phone | Start  | End    | Duration\n"
    table_text += "------+--------+--------+----------\n"
    
    for p in phonemes_in_seconds:
        table_text += f"{p['phone']:<6}| {p['start']:.3f} | {p['end']:.3f} | {p['duration']:.3f}\n"
    
    # Add metadata and table
    plt.figtext(0.02, 0.01, 
               f"Audio: {os.path.basename(wav_path)}\n"
               f"Label: {os.path.basename(lab_path)}\n"
               f"Duration: {audio_duration:.2f}s\n"
               f"Label SR: {lab_sample_rate}Hz\n"
               f"Scaling: {scaling_factor:.2f}x", 
               fontsize=8)
    
    plt.figtext(0.5, 0.01, table_text, fontsize=7, family='monospace')
    
    # Tight layout with adjusted bottom margin to fit the table
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    
    # Save the figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")
    
    return plt.gcf()

def process_directory(wav_dir, lab_dir, output_dir, min_f0=70, max_f0=400, lab_sample_rate=44100, manual_scaling=None):
    """Process all matching WAV/LAB files in directories."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of WAV files
    wav_files = [f for f in os.listdir(wav_dir) if f.endswith('.wav')]
    
    for wav_file in wav_files:
        # Get corresponding lab file
        base_name = os.path.splitext(wav_file)[0]
        lab_file = base_name + '.lab'
        lab_path = os.path.join(lab_dir, lab_file)
        
        # Skip if lab file doesn't exist
        if not os.path.exists(lab_path):
            print(f"Warning: No matching lab file for {wav_file}")
            continue
            
        wav_path = os.path.join(wav_dir, wav_file)
        output_path = os.path.join(output_dir, base_name + '_alignment.png')
        
        try:
            create_visualization(
                wav_path, 
                lab_path, 
                output_path, 
                min_f0=min_f0, 
                max_f0=max_f0, 
                lab_sample_rate=lab_sample_rate, 
                manual_scaling=manual_scaling
            )
            print(f"Processed {wav_file}")
        except Exception as e:
            print(f"Error processing {wav_file}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='F0 and Phoneme Alignment Visualization')
    parser.add_argument('--wav', type=str, help='Path to WAV file or basename if using default directories')
    parser.add_argument('--lab', type=str, help='Path to LAB file or basename if using default directories')
    parser.add_argument('--output', type=str, default='f0_phoneme_alignment.png', help='Output image path')
    parser.add_argument('--min_f0', type=int, default=70, help='Minimum F0 value')
    parser.add_argument('--max_f0', type=int, default=400, help='Maximum F0 value')
    parser.add_argument('--lab_sr', type=int, default=44100, help='Label file sample rate')
    parser.add_argument('--scaling', type=float, default=227.13, help='Manual scaling factor (default: 227.13)')
    parser.add_argument('--batch', action='store_true', help='Process all files in the directories')
    args = parser.parse_args()
    
    # Handle default directories
    wav_dir = os.path.join("datasets", "gin", "wav")
    lab_dir = os.path.join("datasets", "gin", "lab")
    output_dir = "output"
    
    if args.batch:
        process_directory(
            wav_dir,
            lab_dir,
            output_dir,
            min_f0=args.min_f0, 
            max_f0=args.max_f0, 
            lab_sample_rate=args.lab_sr, 
            manual_scaling=args.scaling
        )
    else:
        # For single file processing
        wav_path = args.wav
        lab_path = args.lab
        
        # If only base names are provided, use default directories
        if wav_path and not os.path.dirname(wav_path):
            wav_path = os.path.join(wav_dir, wav_path)
        if lab_path and not os.path.dirname(lab_path):
            lab_path = os.path.join(lab_dir, lab_path)
        
        # Handle case where no arguments are provided
        if not wav_path or not lab_path:
            print("Using example: a little love - Part_1")
            wav_path = os.path.join(wav_dir, "a little love - Part_1.wav")
            lab_path = os.path.join(lab_dir, "a little love - Part_1.lab")
        
        # Ensure the output directory exists
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        create_visualization(
            wav_path, 
            lab_path, 
            args.output, 
            min_f0=args.min_f0, 
            max_f0=args.max_f0, 
            lab_sample_rate=args.lab_sr, 
            manual_scaling=args.scaling
        )

if __name__ == '__main__':
    main()