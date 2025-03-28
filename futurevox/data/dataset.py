import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import yaml


class FutureVoxDataset(Dataset):
    """Dataset for loading preprocessed FutureVox data from HDF5 file."""
    
    def __init__(self, h5_path, config, split='train'):
        """
        Initialize the dataset.
        
        Args:
            h5_path (str): Path to the HDF5 file containing processed data
            config (dict): Configuration dictionary
            split (str): Data split ('train' or 'val')
        """
        self.config = config
        self.h5_path = h5_path
        self.split = split
        
        # Open HDF5 file in read mode
        self.h5_file = h5py.File(h5_path, 'r')
        
        # Get file list
        file_list = self.h5_file['metadata']['file_list'][:]
        self.file_ids = [name.decode('utf-8') for name in file_list]
        
        # Split files into train and validation (80/20 split)
        if split == 'train':
            self.file_ids = self.file_ids[:int(len(self.file_ids) * 0.8)]
        else:  # validation
            self.file_ids = self.file_ids[int(len(self.file_ids) * 0.8):]
            
        print(f"Loaded {len(self.file_ids)} samples for {split}")
    
    def __len__(self):
        return len(self.file_ids)
    
    def __getitem__(self, idx):
        """Get a single sample from the dataset."""
        sample_id = self.file_ids[idx]
        sample_group = self.h5_file[sample_id]
        
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
        
        # Create phone indices tensor (will use a simple mapping for now)
        # In a real implementation, you would use a proper phoneme dictionary
        phone_set = set(phones)
        phone_to_idx = {phone: i for i, phone in enumerate(sorted(phone_set))}
        phone_indices = torch.tensor([phone_to_idx.get(phone, 0) for phone in phones], dtype=torch.long)
        
        # Create a tensor with start and end frames for each phoneme
        start_frames = sample_group['phonemes']['start_frames'][:]
        end_frames = sample_group['phonemes']['end_frames'][:]
        frames = torch.tensor(np.stack([start_frames, end_frames], axis=1), dtype=torch.long)
        
        # Audio waveform (optional)
        audio = sample_group['audio']['waveform'][:]
        audio = torch.from_numpy(audio).float()
        
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
            'audio': audio
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
        batched_frames = torch.zeros(batch_size, max_phone_length, 2, dtype=torch.long)
        batched_audio = torch.zeros(batch_size, max_audio_length)
        
        # Masks for variable length sequences
        mel_masks = torch.zeros(batch_size, max_mel_length, dtype=torch.bool)
        phone_masks = torch.zeros(batch_size, max_phone_length, dtype=torch.bool)
        
        # Sample IDs and original data
        sample_ids = []
        phones_list = []
        start_times_list = []
        end_times_list = []
        durations_list = []
        
        # Fill batched tensors
        for i, sample in enumerate(batch):
            sample_ids.append(sample['sample_id'])
            phones_list.append(sample['phones'])
            start_times_list.append(sample['start_times'])
            end_times_list.append(sample['end_times'])
            durations_list.append(sample['durations'])
            
            # Get sequence lengths
            mel_length = sample['mel_spectrogram'].shape[1]
            phone_length = len(sample['phones'])
            f0_length = len(sample['f0_values'])
            audio_length = len(sample['audio'])
            
            # Copy data to batched tensors
            batched_mel_spectrograms[i, :, :mel_length] = sample['mel_spectrogram']
            batched_f0_values[i, :f0_length] = sample['f0_values']
            batched_phone_indices[i, :phone_length] = sample['phone_indices']
            batched_frames[i, :phone_length] = sample['frames']
            batched_audio[i, :audio_length] = sample['audio']
            
            # Fill masks
            mel_masks[i, mel_length:] = True  # True indicates padding
            phone_masks[i, phone_length:] = True
        
        return {
            'sample_ids': sample_ids,
            'mel_spectrograms': batched_mel_spectrograms,
            'f0_values': batched_f0_values,
            'phone_indices': batched_phone_indices,
            'frames': batched_frames,
            'phones': phones_list,
            'start_times': start_times_list,
            'end_times': end_times_list,
            'durations': durations_list,
            'audio': batched_audio,
            'mel_masks': mel_masks,
            'phone_masks': phone_masks
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