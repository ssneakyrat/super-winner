"""
Dataset module for LightSinger.
Provides implementation for SVS dataset loading from .lab files.
"""

import os
import json
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import librosa
import pyworld as pw
from typing import Dict, List, Tuple, Optional, Union
import glob
import logging

from config.model_config import DataConfig

logger = logging.getLogger(__name__)

class LightSingerDataset(Dataset):
    """Dataset for LightSinger that loads .lab files and corresponding audio files."""
    
    def __init__(
        self,
        data_dir: str,
        config: DataConfig,
        split: str = "train",
        limit_dataset_size: Optional[int] = None
    ):
        """
        Initialize dataset.
        
        Args:
            data_dir: Directory containing the dataset
            config: Data configuration
            split: Dataset split ("train", "val", or "test")
            limit_dataset_size: Limit dataset to first N items (for debugging)
        """
        super().__init__()
        self.data_dir = data_dir
        self.config = config
        self.split = split
        
        # Load phoneme dictionary if it exists
        self.phoneme_dict = self._load_phoneme_dict()
        
        # Load metadata
        self.metadata = []
        self._load_metadata()
        
        # Limit dataset size if needed
        if limit_dataset_size is not None:
            self.metadata = self.metadata[:limit_dataset_size]
            
        logger.info(f"Loaded {len(self.metadata)} samples for {split} split")
    
    def _load_phoneme_dict(self) -> Dict[str, int]:
        """
        Load phoneme dictionary from file or create a new one.
        
        Returns:
            Dict mapping phoneme to ID
        """
        phoneme_dict_path = os.path.join(
            self.data_dir, self.config.phoneme_dict_file
        )
        
        if os.path.exists(phoneme_dict_path):
            with open(phoneme_dict_path, "r") as f:
                phoneme_dict = json.load(f)
            logger.info(f"Loaded phoneme dictionary with {len(phoneme_dict)} phonemes")
            return phoneme_dict
        else:
            logger.warning(f"No phoneme dictionary found at {phoneme_dict_path}. Will create one dynamically.")
            return {}
    
    def _load_metadata(self) -> None:
        """
        Load dataset metadata by scanning for .lab files.
        """
        # Set up split file path
        if self.split == "train":
            split_file = os.path.join(self.data_dir, self.config.train_file)
        elif self.split == "val":
            split_file = os.path.join(self.data_dir, self.config.val_file)
        elif self.split == "test":
            split_file = os.path.join(self.data_dir, self.config.test_file)
        else:
            raise ValueError(f"Invalid split: {self.split}")
        
        # Load file list if available
        if os.path.exists(split_file):
            with open(split_file, "r") as f:
                file_list = [line.strip() for line in f.readlines()]
            
            # Process each file in the list
            for file_path in file_list:
                lab_path = os.path.join(self.data_dir, file_path)
                wav_path = lab_path.replace(".lab", ".wav")
                
                if os.path.exists(lab_path) and os.path.exists(wav_path):
                    self.metadata.append({
                        "id": os.path.basename(file_path).replace(".lab", ""),
                        "lab_path": lab_path,
                        "wav_path": wav_path
                    })
        else:
            # Scan for .lab files directly if no split file exists
            logger.warning(f"No split file found at {split_file}. Scanning directory for .lab files.")
            lab_files = glob.glob(os.path.join(self.data_dir, "**/*.lab"), recursive=True)
            
            for lab_path in lab_files:
                wav_path = lab_path.replace(".lab", ".wav")
                
                if os.path.exists(wav_path):
                    self.metadata.append({
                        "id": os.path.basename(lab_path).replace(".lab", ""),
                        "lab_path": lab_path,
                        "wav_path": wav_path
                    })
    
    def _parse_lab_file(self, lab_path: str) -> Tuple[List[str], np.ndarray]:
        """
        Parse .lab file to extract phonemes and durations.
        
        Args:
            lab_path: Path to .lab file
            
        Returns:
            Tuple of (phonemes, durations)
        """
        phonemes = []
        start_times = []
        end_times = []
        
        with open(lab_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    start_time = int(parts[0])
                    end_time = int(parts[1])
                    phoneme = parts[2]
                    
                    phonemes.append(phoneme)
                    start_times.append(start_time)
                    end_times.append(end_time)
        
        # Convert times to frame-level durations
        hop_size = self.config.hop_size
        sample_rate = self.config.sample_rate
        
        # UPDATED: Check if values are in HTK format (100ns units)
        # Compare the last end_time with reasonable audio length 
        max_time = end_times[-1]
        expected_max_samples = 30 * sample_rate  # Expect max 30 seconds
        
        # If times appear to be in HTK format (100ns units)
        time_scale = 1
        if max_time > expected_max_samples * 10:  # If unreasonably large
            time_scale = sample_rate / 10000000  # Convert from HTK 100ns units to samples
        
        # Convert from time to frame-level durations with appropriate scaling
        durations = []
        for start, end in zip(start_times, end_times):
            # Apply scaling if needed
            start_sample = int(start * time_scale)
            end_sample = int(end * time_scale)
            
            # Convert from sample indices to frame indices
            start_frame = start_sample // hop_size
            end_frame = end_sample // hop_size
            duration = end_frame - start_frame
            durations.append(duration)
        
        durations = np.array(durations)
        
        # Add proportional scaling to match mel length
        # This can be added if needed
        
        # Update phoneme dictionary if needed
        for phoneme in phonemes:
            if phoneme not in self.phoneme_dict:
                self.phoneme_dict[phoneme] = len(self.phoneme_dict) + 1  # 0 is reserved for padding
        
        return phonemes, durations
    
    def _extract_f0(self, audio: np.ndarray, durations: np.ndarray) -> np.ndarray:
        """
        Extract F0 contour from audio using WORLD vocoder.
        
        Args:
            audio: Audio waveform
            durations: Phoneme durations in frames
            
        Returns:
            F0 contour
        """
        # Extract raw F0
        sample_rate = self.config.sample_rate
        hop_size = self.config.hop_size
        
        # Convert audio to float64 for pyworld
        audio = audio.astype(np.float64)

        # Use WORLD vocoder for F0 extraction
        frame_period = hop_size / sample_rate * 1000  # Convert to ms
        
        # WORLD F0 extraction
        _f0, t = pw.harvest(audio, sample_rate, frame_period=frame_period)
        f0 = pw.stonemask(audio, _f0, t, sample_rate)
        
        # Make sure F0 length matches expected length from durations
        expected_length = int(np.sum(durations))
        if len(f0) > expected_length:
            f0 = f0[:expected_length]
        elif len(f0) < expected_length:
            # Pad with zeros if needed
            f0 = np.pad(f0, (0, expected_length - len(f0)))
        
        return f0
    
    def _load_audio(self, wav_path: str) -> np.ndarray:
        """
        Load audio file.
        
        Args:
            wav_path: Path to audio file
            
        Returns:
            Audio waveform
        """
        audio, sr = librosa.load(wav_path, sr=self.config.sample_rate, mono=True)
        return audio
    
    def _compute_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute mel spectrogram from audio.
        
        Args:
            audio: Audio waveform
            
        Returns:
            Mel spectrogram
        """
        # STFT
        stft = librosa.stft(
            audio,
            n_fft=self.config.fft_size,
            hop_length=self.config.hop_size,
            win_length=self.config.win_size
        )
        
        # Convert to magnitude
        magnitude = np.abs(stft)
        
        # Convert to mel scale
        mel_basis = librosa.filters.mel(
            sr=self.config.sample_rate,
            n_fft=self.config.fft_size,
            n_mels=self.config.mel_channels,
            fmin=self.config.fmin,
            fmax=self.config.fmax
        )
        
        mel = np.dot(mel_basis, magnitude)
        
        # Apply log transform
        mel = np.log(np.maximum(mel, 1e-5))
        
        return mel
    
    def _phonemes_to_ids(self, phonemes: List[str]) -> List[int]:
        """
        Convert phoneme strings to token IDs.
        
        Args:
            phonemes: List of phoneme strings
            
        Returns:
            List of phoneme token IDs
        """
        return [self.phoneme_dict.get(p, 0) for p in phonemes]  # 0 is padding/unknown
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single dataset item.
        
        Args:
            idx: Item index
            
        Returns:
            Dictionary of tensors
        """
        item = self.metadata[idx]
        
        # Parse .lab file
        phonemes, durations = self._parse_lab_file(item["lab_path"])
        
        # Convert phonemes to IDs
        phoneme_ids = self._phonemes_to_ids(phonemes)
        
        # Load audio
        audio = self._load_audio(item["wav_path"])
        
        # Extract F0
        f0 = self._extract_f0(audio, durations)
        
        # Compute mel spectrogram
        mel = self._compute_mel_spectrogram(audio)
        
        # Convert to tensors
        phoneme_ids = torch.tensor(phoneme_ids, dtype=torch.long)
        durations = torch.tensor(durations, dtype=torch.long)
        f0 = torch.tensor(f0, dtype=torch.float)
        mel = torch.tensor(mel, dtype=torch.float)
        
        # Return item
        return {
            "id": item["id"],
            "phonemes": phoneme_ids,
            "durations": durations,
            "f0": f0,
            "mel": mel,
            "text": " ".join(phonemes)  # Original phoneme text
        }
    
    def save_phoneme_dict(self) -> None:
        """Save phoneme dictionary to file."""
        phoneme_dict_path = os.path.join(
            self.data_dir, self.config.phoneme_dict_file
        )
        
        with open(phoneme_dict_path, "w") as f:
            json.dump(self.phoneme_dict, f, indent=2)
        
        logger.info(f"Saved phoneme dictionary with {len(self.phoneme_dict)} phonemes to {phoneme_dict_path}")


class LightSingerDataModule:
    """PyTorch Lightning DataModule for LightSinger."""
    
    def __init__(
        self,
        data_dir: str,
        config: DataConfig,
        batch_size: int = 16,
        num_workers: int = 4,
        limit_dataset_size: Optional[int] = None
    ):
        """
        Initialize data module.
        
        Args:
            data_dir: Directory containing the dataset
            config: Data configuration
            batch_size: Batch size
            num_workers: Number of workers for data loading
            limit_dataset_size: Limit dataset to first N items (for debugging)
        """
        self.data_dir = data_dir
        self.config = config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.limit_dataset_size = limit_dataset_size
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self, stage: Optional[str] = None) -> None:
        """
        Set up datasets.
        
        Args:
            stage: Current stage ("fit", "validate", "test")
        """
        # Create datasets for training, validation, and testing
        if stage == "fit" or stage is None:
            self.train_dataset = LightSingerDataset(
                self.data_dir,
                self.config,
                split="train",
                limit_dataset_size=self.limit_dataset_size
            )
            self.val_dataset = LightSingerDataset(
                self.data_dir,
                self.config,
                split="val",
                limit_dataset_size=self.limit_dataset_size
            )
            
            # Save phoneme dictionary after processing training dataset
            self.train_dataset.save_phoneme_dict()
        
        if stage == "test" or stage is None:
            self.test_dataset = LightSingerDataset(
                self.data_dir,
                self.config,
                split="test",
                limit_dataset_size=self.limit_dataset_size
            )
    
    def train_dataloader(self) -> DataLoader:
        """Return training data loader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=self._collate_fn
        )
    
    def val_dataloader(self) -> DataLoader:
        """Return validation data loader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=self._collate_fn
        )
    
    def test_dataloader(self) -> DataLoader:
        """Return test data loader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate function for batch creation.
        
        Args:
            batch: List of items
            
        Returns:
            Dict[str, torch.Tensor]: Batched items
        """
        # Get max lengths
        max_phoneme_len = max(len(item["phonemes"]) for item in batch)
        max_mel_len = max(item["mel"].shape[1] for item in batch)
        max_f0_len = max(len(item["f0"]) for item in batch)
        
        # Prepare batched tensors
        phonemes_batch = []
        durations_batch = []
        f0_batch = []
        mel_batch = []
        ids = []
        phoneme_lens = []
        mel_lens = []
        texts = []
        
        for item in batch:
            ids.append(item["id"])
            texts.append(item["text"])
            
            # Pad phonemes
            phoneme_len = len(item["phonemes"])
            phoneme_lens.append(phoneme_len)
            padded_phonemes = F.pad(
                item["phonemes"],
                (0, max_phoneme_len - phoneme_len),
                value=0  # Padding value
            )
            phonemes_batch.append(padded_phonemes)
            
            # Pad durations
            padded_durations = F.pad(
                item["durations"],
                (0, max_phoneme_len - len(item["durations"])),
                value=0
            )
            durations_batch.append(padded_durations)
            
            # Pad F0
            f0_len = len(item["f0"])
            padded_f0 = F.pad(
                item["f0"],
                (0, max_f0_len - f0_len),
                value=0
            )
            f0_batch.append(padded_f0)
            
            # Pad mel spectrogram
            mel_len = item["mel"].shape[1]
            mel_lens.append(mel_len)
            padded_mel = F.pad(
                item["mel"],
                (0, max_mel_len - mel_len),
                value=0
            )
            mel_batch.append(padded_mel)
        
        # Stack tensors
        phonemes_batch = torch.stack(phonemes_batch)
        durations_batch = torch.stack(durations_batch)
        f0_batch = torch.stack(f0_batch)
        mel_batch = torch.stack(mel_batch)
        phoneme_lens = torch.tensor(phoneme_lens)
        mel_lens = torch.tensor(mel_lens)
        
        return {
            "id": ids,
            "phonemes": phonemes_batch,
            "phoneme_lengths": phoneme_lens,
            "durations": durations_batch,
            "f0": f0_batch,
            "mel": mel_batch,
            "mel_lengths": mel_lens,
            "text": texts
        }