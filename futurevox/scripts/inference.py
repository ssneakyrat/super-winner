"""
Inference script for FutureVox.
Generates singing voice from text input.
"""

import os
import argparse
import json
import torch
import numpy as np
from typing import Dict, List, Optional, Union, Tuple

from config.model_config import FutureVoxConfig
from training.lightning_model import FutureVoxLightning
from data.preprocessing import text_to_phonemes, phoneme_to_sequence
from utils.audio import save_audio, plot_spectrogram, plot_f0, plot_waveform


def load_phoneme_dict(path: str) -> Dict[str, int]:
    """
    Load phoneme dictionary.
    
    Args:
        path: Path to dictionary file
        
    Returns:
        Phoneme to ID mapping
    """
    with open(path, "r") as f:
        phoneme_dict = json.load(f)
    return phoneme_dict


def inference(
    model: FutureVoxLightning,
    phonemes: List[int],
    temperature: float = 0.667,
    max_length: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Run inference with the model.
    
    Args:
        model: FutureVox model
        phonemes: Phoneme token IDs
        temperature: Sampling temperature
        max_length: Maximum phoneme length
        
    Returns:
        Dictionary of outputs
    """
    # Convert to tensor
    phonemes_tensor = torch.tensor(phonemes, dtype=torch.long)
    
    # Truncate if needed
    if max_length is not None and len(phonemes) > max_length:
        phonemes_tensor = phonemes_tensor[:max_length]
    
    # Add batch dimension
    phonemes_batch = phonemes_tensor.unsqueeze(0).to(model.device)
    phoneme_lengths = torch.tensor([len(phonemes_tensor)], dtype=torch.long).to(model.device)
    
    # Run inference
    with torch.no_grad():
        outputs, _ = model.model(
            phonemes=phonemes_batch,
            phoneme_lengths=phoneme_lengths,
            temperature=temperature
        )
    
    # Convert to numpy
    result = {
        "mel_pred": outputs["mel_pred"][0].cpu().numpy(),  # [T, M]
        "f0_pred": outputs["f0_pred"][0].cpu().numpy(),    # [T]
        "waveform": outputs["waveform"][0, 0].cpu().numpy(), # [T]
        "durations": outputs["durations_pred"][0].cpu().numpy() # [L]
    }
    
    return result


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate singing voice with FutureVox")
    
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint"
    )
    
    parser.add_argument(
        "--text", type=str, default=None,
        help="Text input"
    )
    
    parser.add_argument(
        "--phonemes", type=str, default=None,
        help="Phoneme input (space-separated)"
    )
    
    parser.add_argument(
        "--phoneme_dict", type=str, required=True,
        help="Path to phoneme dictionary"
    )
    
    parser.add_argument(
        "--output_dir", type=str, default="./outputs",
        help="Path to output directory"
    )
    
    parser.add_argument(
        "--temperature", type=float, default=0.667,
        help="Sampling temperature"
    )
    
    parser.add_argument(
        "--language", type=str, default="en",
        help="Language code for text-to-phoneme conversion"
    )
    
    parser.add_argument(
        "--output_name", type=str, default="output",
        help="Output file name (without extension)"
    )
    
    parser.add_argument(
        "--plot", action="store_true",
        help="Plot spectrograms and F0 contours"
    )
    
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to run inference on"
    )
    
    return parser.parse_args()


def main():
    """Main inference function."""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load configuration
    config = FutureVoxConfig.from_yaml(args.config)
    
    # Load model
    model = FutureVoxLightning.load_from_checkpoint(
        args.checkpoint,
        config=config
    )
    model.eval()
    model = model.to(args.device)
    
    # Prepare phoneme input
    phoneme_dict = load_phoneme_dict(args.phoneme_dict)
    
    if args.phonemes is not None:
        # Use provided phonemes
        phonemes = args.phonemes
    elif args.text is not None:
        # Convert text to phonemes
        phonemes = text_to_phonemes(args.text, language=args.language)
        print(f"Converted text to phonemes: {phonemes}")
    else:
        raise ValueError("Either --text or --phonemes must be provided")
    
    # Convert phonemes to token IDs
    phoneme_ids = phoneme_to_sequence(phonemes, phoneme_dict)
    
    # Run inference
    print(f"Running inference with temperature {args.temperature}...")
    outputs = inference(
        model,
        phoneme_ids,
        temperature=args.temperature
    )
    
    # Save outputs
    output_base = os.path.join(args.output_dir, args.output_name)
    
    # Save audio
    save_audio(
        outputs["waveform"],
        f"{output_base}.wav",
        sample_rate=config.data.sample_rate
    )
    print(f"Saved audio to {output_base}.wav")
    
    # Save spectrograms and F0 contours
    if args.plot:
        # Plot and save mel spectrogram
        fig = plot_spectrogram(
            outputs["mel_pred"].T,
            title="Predicted Mel Spectrogram",
            show=False
        )
        fig.savefig(f"{output_base}_mel.png")
        print(f"Saved mel spectrogram to {output_base}_mel.png")
        
        # Plot and save F0 contour
        fig = plot_f0(
            outputs["f0_pred"],
            hop_size=config.data.hop_size,
            sample_rate=config.data.sample_rate,
            title="Predicted F0 Contour",
            show=False
        )
        fig.savefig(f"{output_base}_f0.png")
        print(f"Saved F0 contour to {output_base}_f0.png")
        
        # Plot and save waveform
        fig = plot_waveform(
            outputs["waveform"],
            sample_rate=config.data.sample_rate,
            title="Generated Waveform",
            show=False
        )
        fig.savefig(f"{output_base}_waveform.png")
        print(f"Saved waveform to {output_base}_waveform.png")
    
    # Save metadata
    metadata = {
        "text": args.text if args.text else None,
        "phonemes": phonemes,
        "temperature": args.temperature,
        "duration_sum": int(outputs["durations"].sum()),
        "audio_length": len(outputs["waveform"]) / config.data.sample_rate
    }
    
    with open(f"{output_base}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Generation complete! Audio length: {metadata['audio_length']:.2f} seconds")


if __name__ == "__main__":
    main()