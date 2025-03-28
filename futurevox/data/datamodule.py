import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import h5py

from futurevox.dataset import FutureVoxDataset


class FutureVoxDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for FutureVox."""
    
    def __init__(self, config, h5_path=None):
        """
        Initialize the data module.
        
        Args:
            config: Configuration dictionary
            h5_path: Path to HDF5 file (if None, constructed from config)
        """
        super().__init__()
        self.config = config
        
        if h5_path is None:
            data_raw_path = config['datasets']['data_raw']
            self.h5_path = os.path.join(data_raw_path, "binary", "dataset.h5")
        else:
            self.h5_path = h5_path
            
        self.batch_size = config['training']['batch_size']
        self.num_workers = config['training']['num_workers']
        
        self.train_dataset = None
        self.val_dataset = None
    
    def prepare_data(self):
        """Check that the dataset exists."""
        if not os.path.exists(self.h5_path):
            raise FileNotFoundError(
                f"Dataset file not found at {self.h5_path}. "
                f"Run preprocess.py first to create the dataset."
            )
    
    def setup(self, stage=None):
        """Set up the datasets."""
        if stage == 'fit' or stage is None:
            self.train_dataset = FutureVoxDataset(self.h5_path, self.config, split='train')
            self.val_dataset = FutureVoxDataset(self.h5_path, self.config, split='val')
    
    def train_dataloader(self):
        """Return the training data loader."""
        return self.train_dataset.get_dataloader(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True
        )
    
    def val_dataloader(self):
        """Return the validation data loader."""
        return self.val_dataset.get_dataloader(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )
    
    def get_num_phonemes(self):
        """Count the unique phonemes in the dataset."""
        with h5py.File(self.h5_path, 'r') as f:
            # Get file list
            file_list = f['metadata']['file_list'][:]
            file_ids = [name.decode('utf-8') for name in file_list]
            
            # Collect all phonemes
            all_phonemes = set()
            for sample_id in file_ids:
                if sample_id in f:
                    sample = f[sample_id]
                    if 'phonemes' in sample and 'phones' in sample['phonemes']:
                        phones_bytes = sample['phonemes']['phones'][:]
                        phones = [p.decode('utf-8') for p in phones_bytes]
                        all_phonemes.update(phones)
            
            return len(all_phonemes)