import os
import argparse
import yaml
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from pathlib import Path
import json
import h5py
import time
import soundfile as sf
from tqdm import tqdm
import pandas as pd

from models.singer_modules.futurevox_singer import FutureVoxSinger
from data.singer_dataset import SingingVoxDataset


def load_model(checkpoint_path, config):
    """
    Load the model from a checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        config: Configuration dictionary
        
    Returns:
        model: Loaded model
    """
    print(f"Loading model from {checkpoint_path}")
    
    # Get phoneme and singer counts from config or defaults
    num_phonemes = config.get('model', {}).get('phoneme_encoder', {}).get('num_phonemes', 100)
    num_singers = config.get('model', {}).get('variance_adaptor', {}).get('num_singers', 10)
    
    # Create model
    model = FutureVoxSinger(
        config=config,
        num_phonemes=num_phonemes,
        num_singers=num_singers
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        # Lightning checkpoint
        state_dict = checkpoint['state_dict']
        # Remove 'model.' prefix if needed
        if all(k.startswith('model.') for k in state_dict):
            state_dict = {k[6:]: v for k, v in state_dict.items()}
    else:
        # Regular state_dict
        state_dict = checkpoint
    
    # Load weights
    model.load_state_dict(state_dict)
    
    return model


def synthesize_from_phonemes(model, phoneme_data, config, output_dir, device='cuda'):
    """
    Synthesize audio from phoneme data.
    
    Args:
        model: FutureVoxSinger model
        phoneme_data: Dictionary with phoneme information
        config: Configuration dictionary
        output_dir: Output directory
        device: Device to use for inference
        
    Returns:
        output_path: Path to the synthesized audio file
    """
    # Move model to device and set to evaluation mode
    model = model.to(device)
    model.eval()
    
    # Extract data from phoneme_data
    phone_indices = torch.tensor(phoneme_data['phone_indices'], dtype=torch.long).unsqueeze(0).to(device)
    note_indices = torch.tensor(phoneme_data['note_indices'], dtype=torch.long).unsqueeze(0).to(device) if 'note_indices' in phoneme_data else None
    rhythm_info = torch.tensor(phoneme_data['rhythm_info'], dtype=torch.float).unsqueeze(0).to(device) if 'rhythm_info' in phoneme_data else None
    singer_id = torch.tensor([phoneme_data.get('singer_id', 0)], dtype=torch.long).to(device)
    
    # Get inference parameters
    tempo_factor = phoneme_data.get('tempo_factor', config['inference']['tempo_factor'])
    
    # Reference mel spectrogram for style transfer
    ref_mel = None
    if 'ref_mel_path' in phoneme_data and os.path.exists(phoneme_data['ref_mel_path']):
        # Load reference audio
        ref_audio, sr = librosa.load(phoneme_data['ref_mel_path'], sr=config['audio']['sample_rate'])
        
        # Convert to mel spectrogram
        ref_mel = librosa.feature.melspectrogram(
            y=ref_audio,
            sr=sr,
            n_fft=config['audio']['n_fft'],
            hop_length=config['audio']['hop_length'],
            win_length=config['audio']['win_length'],
            n_mels=config['audio']['n_mels'],
            fmin=config['audio']['fmin'],
            fmax=config['audio']['fmax']
        )
        
        # Convert to log scale
        ref_mel = librosa.power_to_db(ref_mel, ref=np.max)
        
        # Convert to tensor
        ref_mel = torch.tensor(ref_mel, dtype=torch.float).unsqueeze(0).to(device)
    
    # Generate output
    with torch.no_grad():
        start_time = time.time()
        outputs = model.inference(
            phone_indices=phone_indices,
            note_indices=note_indices,
            rhythm_info=rhythm_info,
            singer_id=singer_id,
            tempo_factor=tempo_factor,
            ref_mel=ref_mel
        )
        inference_time = time.time() - start_time
        
        # Get generated audio
        waveform = outputs['waveform'].squeeze().cpu().numpy()
        
        # Get mel spectrogram
        mel_spectrogram = outputs['mel_postnet'].squeeze().cpu().numpy()
        
        # Get F0 contour
        f0_contour = outputs['f0_contour'].squeeze().cpu().numpy() if 'f0_contour' in outputs else None
    
    # Create output filename
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_basename = f"futurevox_singer_{timestamp}"
    
    # Save audio file
    output_path = os.path.join(output_dir, f"{output_basename}.wav")
    sf.write(output_path, waveform, config['audio']['sample_rate'])
    
    # Save mel spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        mel_spectrogram,
        y_axis='mel',
        x_axis='time',
        sr=config['audio']['sample_rate'],
        hop_length=config['audio']['hop_length'],
        fmin=config['audio']['fmin'],
        fmax=config['audio']['fmax']
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title('Generated Mel Spectrogram')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{output_basename}_mel.png"))
    plt.close()
    
    # Save F0 contour if available
    if f0_contour is not None:
        plt.figure(figsize=(10, 4))
        times = np.arange(len(f0_contour)) * config['audio']['hop_length'] / config['audio']['sample_rate']
        plt.plot(times, f0_contour, color='blue', alpha=0.8)
        plt.title('F0 Contour')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, f"{output_basename}_f0.png"))
        plt.close()
    
    # Save synthesis information
    synthesis_info = {
        'timestamp': timestamp,
        'phoneme_data': phoneme_data,
        'inference_time': inference_time,
        'audio_duration': len(waveform) / config['audio']['sample_rate'],
        'output_path': output_path,
        'model_checkpoint': args.checkpoint,
        'tempo_factor': tempo_factor,
        'realtime_factor': len(waveform) / config['audio']['sample_rate'] / inference_time
    }
    
    with open(os.path.join(output_dir, f"{output_basename}_info.json"), 'w') as f:
        json.dump(synthesis_info, f, indent=2)
    
    # Print synthesis information
    print(f"Generated audio saved to: {output_path}")
    print(f"Audio duration: {len(waveform) / config['audio']['sample_rate']:.2f} seconds")
    print(f"Inference time: {inference_time:.2f} seconds")
    print(f"Real-time factor: {synthesis_info['realtime_factor']:.2f}x")
    
    return output_path


def batch_synthesize_from_dataset(model, dataset, config, output_dir, num_samples=5, device='cuda'):
    """
    Synthesize audio from multiple samples in a dataset.
    
    Args:
        model: FutureVoxSinger model
        dataset: Dataset containing samples
        config: Configuration dictionary
        output_dir: Output directory
        num_samples: Number of samples to synthesize
        device: Device to use for inference
        
    Returns:
        output_paths: List of paths to the synthesized audio files
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Select random samples
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    # Performance metrics
    synthesis_times = []
    rtf_values = []
    output_paths = []
    
    # Synthesize each sample
    for idx in tqdm(indices, desc="Synthesizing samples"):
        # Get sample
        sample = dataset[idx]
        
        # Create phoneme data
        phoneme_data = {
            'phone_indices': sample['phone_indices'].numpy().tolist(),
            'note_indices': sample['note_indices'].numpy().tolist() if 'note_indices' in sample else None,
            'rhythm_info': sample['rhythm_info'].numpy().tolist() if 'rhythm_info' in sample else None,
            'singer_id': sample['singer_id'].item() if 'singer_id' in sample else 0,
            'tempo_factor': 1.0
        }
        
        # Synthesize
        start_time = time.time()
        output_path = synthesize_from_phonemes(
            model=model,
            phoneme_data=phoneme_data,
            config=config,
            output_dir=os.path.join(output_dir, f"sample_{idx}"),
            device=device
        )
        synthesis_time = time.time() - start_time
        
        # Get ground truth audio for comparison
        gt_audio = sample['audio'].numpy()
        gt_path = os.path.join(output_dir, f"sample_{idx}", "ground_truth.wav")
        sf.write(gt_path, gt_audio, config['audio']['sample_rate'])
        
        # Get audio duration
        audio_duration = len(gt_audio) / config['audio']['sample_rate']
        
        # Calculate real-time factor
        rtf = audio_duration / synthesis_time
        
        # Store metrics
        synthesis_times.append(synthesis_time)
        rtf_values.append(rtf)
        output_paths.append(output_path)
        
        # Print per-sample info
        print(f"Sample {idx}: RTF = {rtf:.2f}x, Duration = {audio_duration:.2f}s, Synthesis time = {synthesis_time:.2f}s")
    
    # Calculate overall statistics
    avg_rtf = np.mean(rtf_values)
    avg_time = np.mean(synthesis_times)
    
    # Save statistics to file
    stats = {
        'avg_rtf': float(avg_rtf),
        'avg_synthesis_time': float(avg_time),
        'rtf_values': [float(x) for x in rtf_values],
        'synthesis_times': [float(x) for x in synthesis_times],
        'model_checkpoint': args.checkpoint,
        'num_samples': len(indices)
    }
    
    with open(os.path.join(output_dir, "synthesis_stats.json"), 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Print overall statistics
    print(f"Average real-time factor: {avg_rtf:.2f}x")
    print(f"Average synthesis time: {avg_time:.2f} seconds")
    
    return output_paths


def evaluate_model(model, dataset, config, output_dir, num_samples=20, device='cuda'):
    """
    Evaluate model quantitatively and generate evaluation metrics.
    
    Args:
        model: FutureVoxSinger model
        dataset: Dataset containing samples
        config: Configuration dictionary
        output_dir: Output directory
        num_samples: Number of samples to evaluate
        device: Device to use for inference
        
    Returns:
        metrics: Dictionary with evaluation metrics
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Select random samples
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    # Initialize metrics
    metrics = {
        'mel_l1_error': [],
        'mel_l2_error': [],
        'f0_rmse': [],
        'f0_corr': [],
        'real_time_factor': [],
        'synthesis_time': []
    }
    
    # Move model to device and set to evaluation mode
    model = model.to(device)
    model.eval()
    
    # Evaluate each sample
    for idx in tqdm(indices, desc="Evaluating samples"):
        # Get sample
        sample = dataset[idx]
        
        # Prepare input tensors
        phone_indices = sample['phone_indices'].unsqueeze(0).to(device)
        
        note_indices = None
        if 'note_indices' in sample:
            note_indices = sample['note_indices'].unsqueeze(0).to(device)
            
        rhythm_info = None
        if 'rhythm_info' in sample:
            rhythm_info = sample['rhythm_info'].unsqueeze(0).to(device)
            
        singer_id = None
        if 'singer_id' in sample:
            singer_id = sample['singer_id'].to(device)
        
        # Ground truth data
        gt_mel = sample['mel_spectrogram'].unsqueeze(0).to(device)
        gt_f0 = sample['f0_values'].to(device) if 'f0_values' in sample else None
        
        # Generate output
        with torch.no_grad():
            start_time = time.time()
            outputs = model.inference(
                phone_indices=phone_indices,
                note_indices=note_indices,
                rhythm_info=rhythm_info,
                singer_id=singer_id
            )
            synthesis_time = time.time() - start_time
            
            # Get generated mel spectrogram
            pred_mel = outputs['mel_postnet']
            
            # Get F0 contour
            pred_f0 = outputs['f0_contour'] if 'f0_contour' in outputs else None
        
        # Calculate metrics
        
        # Mel spectrogram errors (using only valid frames)
        valid_len = min(gt_mel.size(2), pred_mel.size(2))
        gt_mel_valid = gt_mel[:, :, :valid_len]
        pred_mel_valid = pred_mel[:, :, :valid_len]
        
        mel_l1 = torch.mean(torch.abs(gt_mel_valid - pred_mel_valid)).item()
        mel_l2 = torch.mean(torch.pow(gt_mel_valid - pred_mel_valid, 2)).item()
        
        metrics['mel_l1_error'].append(mel_l1)
        metrics['mel_l2_error'].append(mel_l2)
        
        # F0 metrics
        if gt_f0 is not None and pred_f0 is not None:
            valid_len = min(gt_f0.size(0), pred_f0.size(1))
            gt_f0_valid = gt_f0[:valid_len].cpu().numpy()
            pred_f0_valid = pred_f0[0, :valid_len].cpu().numpy()
            
            # Remove zeros and NaNs
            valid_indices = (gt_f0_valid > 0) & (pred_f0_valid > 0)
            if np.any(valid_indices):
                gt_f0_valid = gt_f0_valid[valid_indices]
                pred_f0_valid = pred_f0_valid[valid_indices]
                
                # RMSE in cents
                f0_rmse = np.sqrt(np.mean(np.power(1200 * np.log2(pred_f0_valid / gt_f0_valid), 2)))
                
                # Correlation
                f0_corr = np.corrcoef(gt_f0_valid, pred_f0_valid)[0, 1]
                
                metrics['f0_rmse'].append(f0_rmse)
                metrics['f0_corr'].append(f0_corr)
        
        # Performance metrics
        audio_duration = outputs['waveform'].size(2) / config['audio']['sample_rate']
        rtf = audio_duration / synthesis_time
        
        metrics['real_time_factor'].append(rtf)
        metrics['synthesis_time'].append(synthesis_time)
    
    # Calculate average metrics
    avg_metrics = {k: np.mean(v) for k, v in metrics.items() if len(v) > 0}
    std_metrics = {f"{k}_std": np.std(v) for k, v in metrics.items() if len(v) > 0}
    
    # Combine metrics
    all_metrics = {**avg_metrics, **std_metrics}
    
    # Save metrics to file
    with open(os.path.join(output_dir, "evaluation_metrics.json"), 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    # Print metrics
    print("\nEvaluation Metrics:")
    for k, v in all_metrics.items():
        print(f"{k}: {v:.4f}")
    
    # Create detailed metrics table
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(os.path.join(output_dir, "detailed_metrics.csv"), index=False)
    
    return all_metrics


def main(args):
    """Main inference function."""
    # Read configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update config with command line arguments
    if args.output_dir:
        config['inference']['output_dir'] = args.output_dir
        
    # Create output directory
    output_dir = config['inference']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # Save a copy of the config file
    config_copy_path = os.path.join(output_dir, "inference_config.yaml")
    with open(config_copy_path, 'w') as f:
        yaml.dump(config, f)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.checkpoint, config)
    
    # Choose inference mode
    if args.phoneme_file:
        # Load phoneme data
        with open(args.phoneme_file, 'r') as f:
            phoneme_data = json.load(f)
            
        # Synthesize from phoneme data
        synthesize_from_phonemes(model, phoneme_data, config, output_dir, device)
    
    elif args.dataset:
        # Load dataset
        dataset = SingingVoxDataset(args.dataset, config, split=args.split)
        
        if args.evaluate:
            # Evaluate model
            evaluate_model(model, dataset, config, 
                          os.path.join(output_dir, "evaluation"),
                          num_samples=args.num_samples, device=device)
        else:
            # Batch synthesize from dataset
            batch_synthesize_from_dataset(model, dataset, config,
                                         os.path.join(output_dir, "synthesis"),
                                         num_samples=args.num_samples, device=device)
    
    else:
        print("No input specified. Please provide either a phoneme file or a dataset.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FutureVox-Singer Inference')
    parser.add_argument('--config', type=str, default='config/singer_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--phoneme_file', type=str, default=None,
                        help='Path to phoneme data file (JSON)')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Path to HDF5 dataset file')
    parser.add_argument('--split', type=str, default='test',
                        help='Dataset split to use (train, val, test)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for generated files')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to synthesize/evaluate from dataset')
    parser.add_argument('--evaluate', action='store_true',
                        help='Run evaluation instead of synthesis')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU inference')
    args = parser.parse_args()
    
    # Run inference
    main(args)