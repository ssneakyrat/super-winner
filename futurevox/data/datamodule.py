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
        """Return the training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self):
        """Return the validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )