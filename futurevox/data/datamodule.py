import os
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np

class FutureVoxDataset(Dataset):
    """Dataset class for FutureVox that loads data from H5 files."""
    
    def __init__(self, h5_file_path):
        """
        Initialize the dataset.
        
        Args:
            h5_file_path: Path to the H5 file containing the processed data
        """
        self.h5_file_path = h5_file_path
        
        # Get number of samples
        with h5py.File(self.h5_file_path, 'r') as hf:
            self.sample_count = hf['sample_count'][()]
    
    def __len__(self):
        return self.sample_count
    
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Dictionary with mel spectrogram and other features
        """
        with h5py.File(self.h5_file_path, 'r') as hf:
            sample_key = f'sample{idx}'
            
            # Get mel spectrogram
            mel_spec = hf[sample_key]['mel_spectrogram'][()]
            
            # Get F0 data if available
            f0 = None
            if 'f0' in hf[sample_key]:
                f0 = hf[sample_key]['f0'][()]
            
            # Convert to PyTorch tensors
            mel_spec_tensor = torch.FloatTensor(mel_spec)
            
            # Create sample dict
            sample = {
                'mel_spectrogram': mel_spec_tensor,
                'sample_idx': idx,  # Include the sample index for reference
            }
            
            if f0 is not None:
                sample['f0'] = torch.FloatTensor(f0)
            
            return sample


def collate_variable_length_sequences(batch):
    """
    Custom collate function for batching variable-length sequences.
    
    This function does the following:
    1. Finds the longest mel spectrogram in the batch
    2. Pads all other spectrograms to the same length
    3. Creates a batch with uniform-sized tensors
    4. Adds a mask to indicate which parts are padded
    
    Args:
        batch: List of dictionaries containing 'mel_spectrogram' and other features
        
    Returns:
        Dictionary with batched and padded tensors
    """
    # Extract all spectrograms and keys
    all_specs = [sample['mel_spectrogram'] for sample in batch]
    batch_size = len(all_specs)
    
    # Find max length
    n_mels = all_specs[0].shape[0]  # Number of mel bands should be the same for all
    max_length = max(spec.shape[1] for spec in all_specs)
    
    # Create padded tensor
    padded_specs = torch.zeros(batch_size, n_mels, max_length, dtype=all_specs[0].dtype)
    
    # Create a mask tensor (1 for actual data, 0 for padding)
    lengths = torch.zeros(batch_size, dtype=torch.long)
    masks = torch.zeros(batch_size, max_length, dtype=torch.bool)
    
    # Fill in the data
    for i, spec in enumerate(all_specs):
        spec_length = spec.shape[1]
        padded_specs[i, :, :spec_length] = spec
        lengths[i] = spec_length
        masks[i, :spec_length] = 1
    
    # Create the batch with padded values
    result = {
        'mel_spectrogram': padded_specs,
        'lengths': lengths,  # Store original lengths
        'masks': masks,      # Store masks for attention/loss calculations
        'sample_idx': torch.tensor([sample['sample_idx'] for sample in batch]),
    }
    
    # Add other features if they exist and can be batched
    if 'f0' in batch[0] and batch[0]['f0'] is not None:
        all_f0 = [sample['f0'] for sample in batch]
        padded_f0 = torch.zeros(batch_size, max_length, dtype=all_f0[0].dtype)
        for i, f0 in enumerate(all_f0):
            f0_length = min(f0.shape[0], max_length)  # Ensure f0 doesn't exceed max_length
            padded_f0[i, :f0_length] = f0[:f0_length]
        result['f0'] = padded_f0
    
    return result


class FutureVoxDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for FutureVox."""
    
    def __init__(
        self,
        data_dir,
        batch_size=16,
        num_workers=4,
        train_val_split=0.9
    ):
        """
        Initialize the DataModule.
        
        Args:
            data_dir: Directory containing the preprocessed data
            batch_size: Batch size for training and validation
            num_workers: Number of workers for data loading
            train_val_split: Proportion of data to use for training vs validation
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_split = train_val_split
        
        # Find H5 files in binary directory
        self.h5_file_path = None
        binary_dir = os.path.join(data_dir, 'binary')
        if os.path.exists(binary_dir):
            h5_files = [f for f in os.listdir(binary_dir) if f.endswith('.h5')]
            if h5_files:
                self.h5_file_path = os.path.join(binary_dir, h5_files[0])
    
    def setup(self, stage=None):
        """
        Set up the dataset for the given stage.
        
        Args:
            stage: 'fit', 'validate', 'test', or 'predict'
        """
        if self.h5_file_path is None:
            raise ValueError(f"No H5 file found in {os.path.join(self.data_dir, 'binary')}")
        
        if stage == 'fit' or stage is None:
            dataset = FutureVoxDataset(self.h5_file_path)
            
            # Split dataset into train and validation
            train_size = int(len(dataset) * self.train_val_split)
            val_size = len(dataset) - train_size
            
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size]
            )
    
    def train_dataloader(self):
        """Return the training dataloader with custom collate function."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_variable_length_sequences  # Use custom collate function
        )
    
    def val_dataloader(self):
        """Return the validation dataloader with custom collate function."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_variable_length_sequences  # Use custom collate function
        )