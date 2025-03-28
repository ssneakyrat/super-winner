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
        Validation step with selective logging - only logs sample data.
        
        Args:
            batch: Batch of data from the dataloader
            batch_idx: Index of the batch
        """
        # Skip loss calculation and logging
        
        # Only log sample data for the first batch to avoid excessive logging
        if batch_idx == 0 and hasattr(self, 'h5_file_path') and self.h5_file_path is not None:
            # Get data from batch
            mel_spec = batch['mel_spectrogram']
            sample_indices = batch['sample_idx']
            lengths = batch['lengths']
            
            # Log for a limited number of samples
            max_samples = min(4, len(sample_indices))
            
            for i in range(max_samples):
                # Get sample index
                sample_idx = sample_indices[i].item()
                length = lengths[i].item()
                
                # Get non-padded mel spectrogram
                actual_mel = mel_spec[i, :, :length]
                
                # Log only sample data (F0, phonemes, durations, and combined visualization)
                self._log_sample_data(sample_idx, actual_mel)

    
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