import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for server environments
import matplotlib.pyplot as plt
import numpy as np
import io
from PIL import Image
import torchvision.transforms as transforms
import h5py
import librosa
import librosa.display

class FutureVoxBaseModel(pl.LightningModule):
    """
    Base model for FutureVox using PyTorch Lightning.
    This implementation includes TensorBoard visualization.
    """
    
    def __init__(
        self,
        n_mels=80,
        hidden_dim=256,
        learning_rate=1e-4,
        h5_file_path=None,
        hop_length=256,
        sample_rate=22050
    ):
        """
        Initialize the model.
        
        Args:
            n_mels: Number of mel bands in the spectrogram
            hidden_dim: Hidden dimension size
            learning_rate: Learning rate for optimization
            h5_file_path: Path to the HDF5 file for ground truth data visualization
            hop_length: Hop length for the audio processing
            sample_rate: Audio sample rate
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Store parameters for visualization
        self.h5_file_path = h5_file_path
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        
        # Simple encoder (mel spectrogram -> hidden features)
        self.encoder = nn.Sequential(
            nn.Conv1d(n_mels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Simple decoder (hidden features -> output)
        self.decoder = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, n_mels, kernel_size=3, padding=1)
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input mel spectrogram tensor, shape [batch_size, n_mels, time]
            
        Returns:
            Output tensor (reconstructed mel spectrogram)
        """
        # Encode
        hidden = self.encoder(x)
        
        # Decode
        output = self.decoder(hidden)
        
        return output
    
    def training_step(self, batch, batch_idx):
        """
        Training step with masked loss for variable-length sequences.
        
        Args:
            batch: Batch of data from the dataloader
            batch_idx: Index of the batch
            
        Returns:
            Loss value
        """
        # Get mel spectrogram and mask
        mel_spec = batch['mel_spectrogram']
        masks = batch['masks']
        
        # Forward pass (simple autoencoder)
        output = self(mel_spec)
        
        # Calculate masked reconstruction loss
        # First calculate MSE for all positions
        mse = F.mse_loss(output, mel_spec, reduction='none')
        
        # Sum across mel dimension to get [batch_size, time]
        mse = mse.sum(dim=1)
        
        # Apply mask to ignore padded regions
        masked_mse = mse * masks.float()
        
        # Sum over time dim and divide by number of actual (non-padded) frames
        loss = masked_mse.sum() / masks.float().sum()
        
        # Log loss
        self.log('train_loss', loss, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step with TensorBoard logging for ground truth visualization.
        
        Args:
            batch: Batch of data from the dataloader
            batch_idx: Index of the batch
        """
        # Get mel spectrogram and mask
        mel_spec = batch['mel_spectrogram']
        masks = batch['masks']
        
        # Forward pass
        output = self(mel_spec)
        
        # Calculate masked reconstruction loss
        mse = F.mse_loss(output, mel_spec, reduction='none')
        mse = mse.sum(dim=1)
        masked_mse = mse * masks.float()
        loss = masked_mse.sum() / masks.float().sum()
        
        # Log loss
        self.log('val_loss', loss, prog_bar=True)
        
        # Log visualizations (only for the first batch to avoid excessive logging)
        if batch_idx == 0 and hasattr(self, 'h5_file_path') and self.h5_file_path is not None:
            self._log_visualizations(batch)
    
    def configure_optimizers(self):
        """
        Configure optimizers.
        
        Returns:
            Optimizer
        """
        # Explicit conversion to float to avoid string-to-float comparison errors
        learning_rate = float(self.hparams.learning_rate)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        return optimizer
    
    def _fig_to_tensor(self, fig):
        """Convert matplotlib figure to tensor for TensorBoard logging."""
        # Save figure to a memory buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=120)
        buf.seek(0)
        
        # Convert to PIL Image and then to tensor
        img = Image.open(buf)
        transform = transforms.ToTensor()
        img_tensor = transform(img)
        
        plt.close(fig)  # Close the figure to avoid memory leaks
        return img_tensor
    
    def _log_visualizations(self, batch):
        """
        Log ground truth visualizations to TensorBoard.
        Modified to handle padded batches.
        
        Args:
            batch: Batch of data from the dataloader
        """
        # Get data from batch
        mel_spec = batch['mel_spectrogram']
        sample_indices = batch['sample_idx']
        masks = batch['masks']
        lengths = batch['lengths']
        
        # Log for a limited number of samples
        max_samples = min(4, len(sample_indices))
        
        for i in range(max_samples):
            # Get sample index
            sample_idx = sample_indices[i].item()
            length = lengths[i].item()
            
            # Get non-padded mel spectrogram
            actual_mel = mel_spec[i, :, :length]
            
            # Log mel spectrogram
            self._log_mel_spectrogram(actual_mel, sample_idx)
            
            # Log F0, phonemes, and durations from the H5 file
            self._log_sample_data(sample_idx, actual_mel)
    
    def _log_mel_spectrogram(self, mel_spec, sample_idx):
        """Log mel spectrogram to TensorBoard."""
        # Convert to numpy for visualization
        mel_np = mel_spec.cpu().numpy()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 4))
        img = librosa.display.specshow(
            mel_np, 
            x_axis='time', 
            y_axis='mel', 
            sr=self.sample_rate,
            hop_length=self.hop_length,
            ax=ax
        )
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set_title(f'Mel Spectrogram - Sample {sample_idx}')
        
        # Convert figure to tensor and log to TensorBoard
        mel_img = self._fig_to_tensor(fig)
        self.logger.experiment.add_image(f'val/mel_spectrogram/{sample_idx}', mel_img, self.global_step)
    
    def _log_f0_and_durations(self, f0, voiced_flag, phoneme_data, sr, sample_idx):
        """
        Log F0 (fundamental frequency) and phoneme durations in a single combined visualization.
        
        Args:
            f0: Fundamental frequency array
            voiced_flag: Boolean array indicating voiced frames (if available)
            phoneme_data: List of dictionaries with phoneme information
            sr: Sample rate
            sample_idx: Index of the sample
        """
        # Create figure with two subplots that share x-axis
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, 
                                        gridspec_kw={'height_ratios': [2, 1]})
        
        # Plot F0 in the top subplot
        times = np.arange(len(f0)) * self.hop_length / sr
        max_time = times[-1] if len(times) > 0 else 0
        
        if voiced_flag is not None:
            # Plot only voiced frames
            voiced_times = times[voiced_flag]
            voiced_f0 = f0[voiced_flag]
            ax1.scatter(voiced_times, voiced_f0, s=10, c='b', alpha=0.8, label='F0 (voiced)')
        else:
            # Plot all F0 values
            ax1.plot(times, f0, 'b-', alpha=0.7, label='F0')
        
        # Customize F0 plot
        ax1.set_ylabel('Frequency (Hz)')
        ax1.set_title(f'Combined F0 and Phoneme Durations - Sample {sample_idx}')
        ax1.grid(alpha=0.3)
        ax1.legend()
        
        # Plot phoneme boundaries on F0 plot
        y_min, y_max = ax1.get_ylim()
        
        # Add phoneme boundaries and labels to F0 plot
        for phoneme in phoneme_data:
            start_time = phoneme['start_time'] / sr
            end_time = phoneme['end_time'] / sr
            label = phoneme['label']
            
            # Add vertical line at boundary
            ax1.axvline(x=start_time, color='r', linestyle='--', alpha=0.5)
            
            # Add label
            label_x = (start_time + end_time) / 2
            ax1.text(label_x, y_max*0.9, label, 
                    horizontalalignment='center', 
                    verticalalignment='top',
                    fontsize=8, 
                    bbox=dict(facecolor='white', alpha=0.7))
        
        # Add final boundary
        if phoneme_data:  # Check if there are any phonemes
            ax1.axvline(x=phoneme_data[-1]['end_time'] / sr, color='r', linestyle='--', alpha=0.5)
        
        # Calculate durations in frames
        durations = []
        labels = []
        time_positions = []  # Store time positions for x-axis alignment
        
        for phoneme in phoneme_data:
            # Convert from sample indices to frames
            start_frame = phoneme['start_time'] // self.hop_length
            end_frame = phoneme['end_time'] // self.hop_length
            duration = max(1, end_frame - start_frame)  # Ensure positive duration
            
            # Calculate time positions for bar chart
            start_time = phoneme['start_time'] / sr
            time_positions.append(start_time)
            
            durations.append(duration)
            labels.append(phoneme['label'])
        
        # Create bar chart with proper x-axis alignment
        if durations:  # Check if there are any durations
            bar_width = []
            for i in range(len(time_positions)):
                if i < len(time_positions) - 1:
                    # Width is the distance to the next time position
                    width = time_positions[i+1] - time_positions[i]
                else:
                    # For the last element, use the same width as previous
                    # or a fraction of the total time
                    width = max_time * 0.05 if i == 0 else (time_positions[i] - time_positions[i-1])
                bar_width.append(width)
            
            bars = ax2.bar(time_positions, durations, width=bar_width, alpha=0.7)
            
            # Add labels
            for i, (x, v) in enumerate(zip(time_positions, durations)):
                ax2.text(x + bar_width[i]/2, v + 0.5, str(v), ha='center', fontsize=8)
        
        # Customize durations plot
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Duration (frames)')
        ax2.set_xlim(0, max_time)
        ax2.grid(axis='y', alpha=0.3)
        
        # Add a title to clarify this is the duration plot
        ax2.set_title('Phoneme Durations (frames)', fontsize=10)
        
        plt.tight_layout()
        
        # Convert figure to tensor and log to TensorBoard
        combined_img = self._fig_to_tensor(fig)
        self.logger.experiment.add_image(
            f'val/f0_and_durations/{sample_idx}', 
            combined_img, 
            self.global_step
        )
        
    def _log_sample_data(self, sample_idx, mel_spec):
        """
        Log sample data (F0, phonemes, durations) to TensorBoard.
        
        Args:
            sample_idx: Index of the sample in the H5 file
            mel_spec: Mel spectrogram tensor for the sample
        """
        with h5py.File(self.h5_file_path, 'r') as hf:
            sample_key = f'sample{sample_idx}'
            
            # Get sample rate if available
            sr = self.sample_rate
            if 'sample_rate' in hf[sample_key]:
                sr = hf[sample_key]['sample_rate'][()]
            
            # Collect F0 data if available
            f0 = None 
            voiced_flag = None
            if 'f0' in hf[sample_key]:
                f0 = hf[sample_key]['f0'][()]
                if 'voiced_flag' in hf[sample_key]:
                    voiced_flag = hf[sample_key]['voiced_flag'][()]
            
            # Collect phoneme data if available
            phoneme_data = []
            if 'phonemes' in hf[sample_key] and 'phoneme_count' in hf[sample_key]:
                phoneme_count = hf[sample_key]['phoneme_count'][()]
                
                for i in range(phoneme_count):
                    phoneme_key = f'phoneme{i}'
                    if phoneme_key in hf[sample_key]['phonemes']:
                        phoneme = hf[sample_key]['phonemes'][phoneme_key]
                        label = phoneme['label'][()]
                        if isinstance(label, bytes):
                            label = label.decode('utf-8')
                        
                        phoneme_data.append({
                            'start_time': phoneme['start_time'][()],
                            'end_time': phoneme['end_time'][()],
                            'label': label
                        })
        
        # Log combined F0 and durations if both are available
        if f0 is not None and phoneme_data:
            self._log_f0_and_durations(f0, voiced_flag, phoneme_data, sr, sample_idx)
            
            # Still log phoneme alignment as before
            self._log_phonemes(phoneme_data, mel_spec.cpu().numpy(), sr, sample_idx)
        else:
            # Log individually if one is missing
            if f0 is not None:
                self._log_f0(f0, voiced_flag, sample_idx, sr)
            
            if phoneme_data:
                # Log phoneme alignment
                self._log_phonemes(phoneme_data, mel_spec.cpu().numpy(), sr, sample_idx)
                
                # Calculate and log durations
                self._log_durations(phoneme_data, sr, sample_idx)
    
    def _log_f0(self, f0, voiced_flag, sample_idx, sr):
        """Log F0 (fundamental frequency) to TensorBoard."""
        fig, ax = plt.subplots(figsize=(12, 4))
        
        # Calculate time axis
        times = np.arange(len(f0)) * self.hop_length / sr
        
        if voiced_flag is not None:
            # Plot only voiced frames
            voiced_times = times[voiced_flag]
            voiced_f0 = f0[voiced_flag]
            ax.scatter(voiced_times, voiced_f0, s=10, c='b', alpha=0.8, label='F0 (voiced)')
            
            # Label axes
            ax.set_ylabel('Frequency (Hz)')
            ax.set_xlabel('Time (s)')
            ax.set_title(f'Fundamental Frequency (F0) - Sample {sample_idx}')
            ax.grid(alpha=0.3)
            ax.legend()
        else:
            # Plot all F0 values
            ax.plot(times, f0, 'b-', alpha=0.7, label='F0')
            
            # Label axes
            ax.set_ylabel('Frequency (Hz)')
            ax.set_xlabel('Time (s)')
            ax.set_title(f'Fundamental Frequency (F0) - Sample {sample_idx}')
            ax.grid(alpha=0.3)
            ax.legend()
        
        # Convert figure to tensor and log to TensorBoard
        f0_img = self._fig_to_tensor(fig)
        self.logger.experiment.add_image(f'val/f0/{sample_idx}', f0_img, self.global_step)
    
    def _log_phonemes(self, phoneme_data, mel_spec, sr, sample_idx):
        """Log phoneme alignment to TensorBoard."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot the mel spectrogram
        img = librosa.display.specshow(
            mel_spec, 
            sr=sr, 
            hop_length=self.hop_length, 
            x_axis='time', 
            y_axis='mel', 
            ax=ax
        )
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        
        # Get y-axis limits for placing text
        y_min, y_max = ax.get_ylim()
        
        # Add phoneme boundaries and labels
        for phoneme in phoneme_data:
            start_time = phoneme['start_time'] / sr
            end_time = phoneme['end_time'] / sr
            label = phoneme['label']
            
            # Add vertical line at boundary
            ax.axvline(x=start_time, color='r', linestyle='--', alpha=0.7)
            
            # Add label text
            label_x = (start_time + end_time) / 2
            ax.text(label_x, y_max*0.9, label, 
                    horizontalalignment='center', 
                    verticalalignment='center',
                    fontsize=10, 
                    bbox=dict(facecolor='white', alpha=0.7))
        
        # Add final boundary
        ax.axvline(x=phoneme_data[-1]['end_time'] / sr, color='r', linestyle='--', alpha=0.7)
        
        ax.set_title(f'Phoneme Alignment - Sample {sample_idx}')
        
        # Convert figure to tensor and log to TensorBoard
        phoneme_img = self._fig_to_tensor(fig)
        self.logger.experiment.add_image(f'val/phonemes/{sample_idx}', phoneme_img, self.global_step)
    
    def _log_durations(self, phoneme_data, sr, sample_idx):
        """Log phoneme durations to TensorBoard."""
        # Calculate durations in frames
        durations = []
        labels = []
        
        for phoneme in phoneme_data:
            # Convert from sample indices to frames
            start_frame = phoneme['start_time'] // self.hop_length
            end_frame = phoneme['end_time'] // self.hop_length
            duration = max(1, end_frame - start_frame)  # Ensure positive duration
            durations.append(duration)
            labels.append(phoneme['label'])
        
        # Create figure
        fig, ax = plt.subplots(figsize=(max(8, len(durations) * 0.3), 6))
        bars = ax.bar(range(len(durations)), durations)
        
        # Add labels to the x-axis
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        
        # Add values on top of bars
        for i, v in enumerate(durations):
            ax.text(i, v + 0.5, str(v), ha='center')
        
        ax.set_xlabel('Phoneme')
        ax.set_ylabel('Duration (frames)')
        ax.set_title(f'Phoneme Durations - Sample {sample_idx}')
        plt.tight_layout()
        
        # Convert figure to tensor and log to TensorBoard
        durations_img = self._fig_to_tensor(fig)
        self.logger.experiment.add_image(f'val/durations/{sample_idx}', durations_img, self.global_step)
        
        # Also log as scalar values for comparison
        for i, (label, duration) in enumerate(zip(labels, durations)):
            self.logger.experiment.add_scalar(
                f'val/duration_frames/{sample_idx}/{label}_{i}', 
                duration, 
                self.global_step
            )