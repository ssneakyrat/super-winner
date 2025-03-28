import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import yaml
import os

class SingingVoxDataset(Dataset):
    """Dataset for loading preprocessed singing voice data from HDF5 file."""
    
    def __init__(self, h5_path, config, split='train'):
        """
        Initialize the dataset.
        
        Args:
            h5_path (str): Path to the HDF5 file containing processed data
            config (dict): Configuration dictionary
            split (str): Data split ('train', 'val', or 'test')
        """
        self.config = config
        self.h5_path = h5_path
        self.split = split
        
        # Open HDF5 file temporarily to get file list
        with h5py.File(h5_path, 'r') as h5_file:
            # Get file list
            file_list = h5_file['metadata']['file_list'][:]
            self.file_ids = [name.decode('utf-8') for name in file_list]
            
            # Split files into train, validation, and test (80/10/10 split)
            if split == 'train':
                self.file_ids = self.file_ids[:int(len(self.file_ids) * 0.8)]
            elif split == 'val':
                self.file_ids = self.file_ids[int(len(self.file_ids) * 0.8):int(len(self.file_ids) * 0.9)]
            else:  # test
                self.file_ids = self.file_ids[int(len(self.file_ids) * 0.9):]
                
        print(f"Loaded {len(self.file_ids)} samples for {split}")
    
    def __len__(self):
        return len(self.file_ids)
    
    def __getitem__(self, idx):
        """Get a single sample from the dataset."""
        sample_id = self.file_ids[idx]
        
        # Open the file for each access
        with h5py.File(self.h5_path, 'r') as h5_file:
            sample_group = h5_file[sample_id]
            
            # Get mel spectrogram
            mel_spec = sample_group['features']['mel_spectrogram'][:]
            mel_spec = torch.from_numpy(mel_spec).float()
            
            # Get F0 values
            f0_values = sample_group['features']['f0_values'][:]
            # Replace NaN values with zeros
            f0_values = np.nan_to_num(f0_values)
            f0_values = torch.from_numpy(f0_values).float()
            
            # Get phoneme data
            phones_bytes = sample_group['phonemes']['phones'][:]
            phones = [p.decode('utf-8') for p in phones_bytes]
            start_times = sample_group['phonemes']['start_times'][:]
            end_times = sample_group['phonemes']['end_times'][:]
            durations = sample_group['phonemes']['durations'][:]
            
            # Create phone indices tensor 
            phone_set = set(phones)
            phone_to_idx = {phone: i for i, phone in enumerate(sorted(phone_set))}
            phone_indices = torch.tensor([phone_to_idx.get(phone, 0) for phone in phones], dtype=torch.long)
            
            # Create a tensor with start and end frames for each phoneme
            start_frames = sample_group['phonemes']['start_frames'][:]
            end_frames = sample_group['phonemes']['end_frames'][:]
            frames = torch.tensor(np.stack([start_frames, end_frames], axis=1), dtype=torch.long)
            
            # Audio waveform
            audio = sample_group['audio']['waveform'][:]
            audio = torch.from_numpy(audio).float()
            
            # SINGING-SPECIFIC DATA
            
            # Get musical note information if available
            if 'notes' in sample_group:
                note_values = sample_group['notes']['note_values'][:]
                note_indices = torch.from_numpy(note_values).long()
            else:
                # Create dummy note indices if not available
                note_indices = torch.zeros_like(phone_indices)
            
            # Get energy values if available
            if 'energy' in sample_group['features']:
                energy = sample_group['features']['energy'][:]
                energy = torch.from_numpy(energy).float()
            else:
                # Calculate energy from mel spectrogram if not available
                energy = torch.norm(mel_spec, dim=0)
                
            # Get singer ID if available
            if 'singer_id' in sample_group.attrs:
                singer_id = sample_group.attrs['singer_id']
            else:
                singer_id = 0  # Default singer ID
            
            # Get vibrato information if available
            if 'vibrato' in sample_group['features']:
                vibrato_rate = sample_group['features']['vibrato']['rate'][:]
                vibrato_extent = sample_group['features']['vibrato']['extent'][:]
                
                vibrato_rate = torch.from_numpy(vibrato_rate).float()
                vibrato_extent = torch.from_numpy(vibrato_extent).float()
            else:
                # Create dummy vibrato tensors if not available
                vibrato_rate = torch.zeros_like(f0_values)
                vibrato_extent = torch.zeros_like(f0_values)
                
            # Get rhythm information if available
            if 'rhythm' in sample_group:
                rhythm_info = sample_group['rhythm']['beats'][:]
                rhythm_info = torch.from_numpy(rhythm_info).float()
            else:
                # Create dummy rhythm tensor if not available
                rhythm_info = torch.zeros_like(phone_indices).float()
        
        # Return everything in a dictionary
        return {
            'sample_id': sample_id,
            'mel_spectrogram': mel_spec,
            'f0_values': f0_values,
            'phones': phones,
            'phone_indices': phone_indices,
            'frames': frames,
            'start_times': torch.from_numpy(start_times).float(),
            'end_times': torch.from_numpy(end_times).float(),
            'durations': torch.from_numpy(durations).float(),
            'audio': audio,
            'note_indices': note_indices,
            'energy': energy,
            'singer_id': torch.tensor([singer_id], dtype=torch.long),
            'vibrato_rate': vibrato_rate,
            'vibrato_extent': vibrato_extent,
            'rhythm_info': rhythm_info
        }
    
    def collate_fn(self, batch):
        """
        Custom collate function for variable length sequences.
        
        Args:
            batch: List of samples from __getitem__
            
        Returns:
            Batched tensors with appropriate padding
        """
        # Get max lengths
        max_mel_length = max(x['mel_spectrogram'].shape[1] for x in batch)
        max_phone_length = max(len(x['phones']) for x in batch)
        max_f0_length = max(len(x['f0_values']) for x in batch)
        max_audio_length = max(len(x['audio']) for x in batch)
        
        # Initialize batched tensors
        batch_size = len(batch)
        mel_dim = batch[0]['mel_spectrogram'].shape[0]
        
        batched_mel_spectrograms = torch.zeros(batch_size, mel_dim, max_mel_length)
        batched_f0_values = torch.zeros(batch_size, max_f0_length)
        batched_phone_indices = torch.zeros(batch_size, max_phone_length, dtype=torch.long)
        batched_note_indices = torch.zeros(batch_size, max_phone_length, dtype=torch.long)
        batched_frames = torch.zeros(batch_size, max_phone_length, 2, dtype=torch.long)
        batched_audio = torch.zeros(batch_size, max_audio_length)
        batched_energy = torch.zeros(batch_size, max_f0_length)
        batched_vibrato_rate = torch.zeros(batch_size, max_f0_length)
        batched_vibrato_extent = torch.zeros(batch_size, max_f0_length)
        batched_rhythm_info = torch.zeros(batch_size, max_phone_length)
        
        # Masks for variable length sequences
        mel_masks = torch.ones(batch_size, max_mel_length, dtype=torch.bool)
        phone_masks = torch.ones(batch_size, max_phone_length, dtype=torch.bool)
        
        # Sample IDs and original data
        sample_ids = []
        phones_list = []
        start_times_list = []
        end_times_list = []
        durations_list = []
        singer_ids = []
        
        # Fill batched tensors
        for i, sample in enumerate(batch):
            sample_ids.append(sample['sample_id'])
            phones_list.append(sample['phones'])
            start_times_list.append(sample['start_times'])
            end_times_list.append(sample['end_times'])
            durations_list.append(sample['durations'])
            singer_ids.append(sample['singer_id'])
            
            # Get sequence lengths
            mel_length = sample['mel_spectrogram'].shape[1]
            phone_length = len(sample['phones'])
            f0_length = len(sample['f0_values'])
            audio_length = len(sample['audio'])
            
            # Copy data to batched tensors
            batched_mel_spectrograms[i, :, :mel_length] = sample['mel_spectrogram']
            batched_f0_values[i, :f0_length] = sample['f0_values']
            batched_phone_indices[i, :phone_length] = sample['phone_indices']
            batched_note_indices[i, :phone_length] = sample['note_indices']
            batched_frames[i, :phone_length] = sample['frames']
            batched_audio[i, :audio_length] = sample['audio']
            batched_energy[i, :f0_length] = sample['energy'][:f0_length]
            batched_vibrato_rate[i, :f0_length] = sample['vibrato_rate'][:f0_length]
            batched_vibrato_extent[i, :f0_length] = sample['vibrato_extent'][:f0_length]
            batched_rhythm_info[i, :phone_length] = sample['rhythm_info'][:phone_length]
            
            # Fill masks (False = valid data, True = padding)
            mel_masks[i, :mel_length] = False
            phone_masks[i, :phone_length] = False
        
        # Stack singer IDs
        batched_singer_ids = torch.cat(singer_ids, dim=0)
        
        return {
            'sample_ids': sample_ids,
            'mel_spectrograms': batched_mel_spectrograms,
            'f0_values': batched_f0_values,
            'phone_indices': batched_phone_indices,
            'note_indices': batched_note_indices,
            'frames': batched_frames,
            'phones': phones_list,
            'start_times': start_times_list,
            'end_times': end_times_list,
            'durations': durations_list,
            'audio': batched_audio,
            'mel_masks': mel_masks,
            'phone_masks': phone_masks,
            'energy': batched_energy,
            'singer_ids': batched_singer_ids,
            'vibrato_rate': batched_vibrato_rate,
            'vibrato_extent': batched_vibrato_extent,
            'rhythm_info': batched_rhythm_info
        }
    
    def get_dataloader(self, batch_size, num_workers=4, shuffle=True):
        """Create a DataLoader for this dataset."""
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )


class SingingVoxDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for SingingVox."""
    
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
            self.h5_path = os.path.join(data_raw_path, "binary", "singing_dataset.h5")
        else:
            self.h5_path = h5_path
            
        self.batch_size = config['training']['batch_size']
        self.num_workers = config['training']['num_workers']
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
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
            self.train_dataset = SingingVoxDataset(self.h5_path, self.config, split='train')
            self.val_dataset = SingingVoxDataset(self.h5_path, self.config, split='val')
        
        if stage == 'test' or stage is None:
            self.test_dataset = SingingVoxDataset(self.h5_path, self.config, split='test')
    
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
    
    def test_dataloader(self):
        """Return the test data loader."""
        return self.test_dataset.get_dataloader(
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
    
    def get_num_singers(self):
        """Count the unique singers in the dataset."""
        with h5py.File(self.h5_path, 'r') as f:
            # Get file list
            file_list = f['metadata']['file_list'][:]
            file_ids = [name.decode('utf-8') for name in file_list]
            
            # Collect all singer IDs
            singer_ids = set()
            for sample_id in file_ids:
                if sample_id in f:
                    sample = f[sample_id]
                    if 'singer_id' in sample.attrs:
                        singer_ids.add(sample.attrs['singer_id'])
            
            # If no singer IDs found, return default value
            if not singer_ids:
                return 1
                
            return max(singer_ids) + 1  # +1 to account for 0-indexing