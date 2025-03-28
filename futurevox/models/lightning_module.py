import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np
import io
from PIL import Image
import torch.nn.functional as F

from models.model import FutureVoxModel
from utils.utils import create_alignment_visualization


class FutureVoxLightningModule(pl.LightningModule):
    """PyTorch Lightning module for FutureVox."""
    
    def __init__(self, config, num_phonemes=100):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        # Initialize model
        self.model = FutureVoxModel(config, num_phonemes)
        
        # Store configuration
        self.learning_rate = float(config['training']['learning_rate'])
        
        # Define loss functions
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
    
    def forward(self, mel_spectrograms, phone_indices, frames=None, mel_masks=None, phone_masks=None):
        """Forward pass through model."""
        return self.model(mel_spectrograms, phone_indices, frames, mel_masks, phone_masks)
    
    def training_step(self, batch, batch_idx):
        """
        Empty training step as required.
        In a real implementation, this would compute losses and update the model.
        """
        # Get model outputs
        outputs = self(
            batch['mel_spectrograms'],
            batch['phone_indices'],
            batch['frames'],
            batch['mel_masks'],
            batch['phone_masks']
        )
        
        # For now, just return a dummy loss
        dummy_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Log loss
        self.log('train_loss', dummy_loss, prog_bar=True)
        
        return dummy_loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step with visualization."""
        # Get model outputs
        outputs = self(
            batch['mel_spectrograms'],
            batch['phone_indices'],
            batch['frames'],
            batch['mel_masks'],
            batch['phone_masks']
        )
        
        # Calculate dummy validation loss
        dummy_val_loss = torch.tensor(0.0, device=self.device)
        self.log('val_loss', dummy_val_loss, prog_bar=True)
        
        # Visualize alignment for first sample in batch
        if batch_idx == 0:
            sample_idx = 0
            
            # Extract data for visualization
            sample_id = batch['sample_ids'][sample_idx]
            mel_spec = batch['mel_spectrograms'][sample_idx].cpu().numpy()
            f0_values = batch['f0_values'][sample_idx].cpu().numpy()
            phones = batch['phones'][sample_idx]
            start_times = batch['start_times'][sample_idx].cpu().numpy()
            end_times = batch['end_times'][sample_idx].cpu().numpy()
            
            # Create visualization figure
            fig = create_alignment_visualization(
                sample_id, 
                mel_spec, 
                f0_values, 
                phones, 
                start_times, 
                end_times, 
                self.config
            )
            
            # Convert figure to image and log to TensorBoard
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            
            # Open image with PIL to convert to RGB
            image = Image.open(buf).convert('RGB')
            image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1)  # HWC -> CHW
            
            # Log to TensorBoard
            self.logger.experiment.add_image(
                f'Alignment/{sample_id}', 
                image_tensor, 
                self.global_step
            )
            
            plt.close(fig)
        
        return dummy_val_loss
    
    def configure_optimizers(self):
        """Configure the optimizer."""
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        
        # Optional: Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5,
            verbose=True
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }