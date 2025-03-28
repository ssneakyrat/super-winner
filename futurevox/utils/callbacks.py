import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import torch
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image
import h5py
from pathlib import Path

from utils.utils import create_alignment_visualization


class AlignmentVisualizationCallback(Callback):
    """
    Callback to visualize phoneme alignments during training.
    """
    
    def __init__(self, h5_path, config, num_samples=4):
        """
        Initialize callback.
        
        Args:
            h5_path: Path to the HDF5 file
            config: Configuration dictionary
            num_samples: Number of samples to visualize
        """
        super().__init__()
        self.h5_path = h5_path
        self.config = config
        self.num_samples = num_samples
        
        # Load a fixed set of sample IDs for visualization
        with h5py.File(h5_path, 'r') as f:
            file_list = f['metadata']['file_list'][:]
            file_ids = [name.decode('utf-8') for name in file_list]
            
            # Select a few samples for visualization
            self.vis_sample_ids = file_ids[:num_samples]
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Visualize alignments at the end of each validation epoch."""
        # Open HDF5 file
        with h5py.File(self.h5_path, 'r') as f:
            for i, sample_id in enumerate(self.vis_sample_ids):
                # Get sample data
                sample_group = f[sample_id]
                
                # Extract features
                mel_spec = sample_group['features']['mel_spectrogram'][:]
                f0_values = sample_group['features']['f0_values'][:]
                
                # Replace NaN values in F0
                f0_values = np.nan_to_num(f0_values)
                
                # Get phoneme data
                phones_bytes = sample_group['phonemes']['phones'][:]
                phones = [p.decode('utf-8') for p in phones_bytes]
                start_times = sample_group['phonemes']['start_times'][:]
                end_times = sample_group['phonemes']['end_times'][:]
                
                # Create visualization
                fig = create_alignment_visualization(
                    sample_id, 
                    mel_spec, 
                    f0_values, 
                    phones, 
                    start_times, 
                    end_times, 
                    self.config
                )
                
                # Convert figure to image
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                
                # Open image with PIL
                image = Image.open(buf).convert('RGB')
                image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1)  # HWC -> CHW
                
                # Log to TensorBoard
                trainer.logger.experiment.add_image(
                    f'Epoch_Alignment/{sample_id}', 
                    image_tensor, 
                    trainer.current_epoch
                )
                
                plt.close(fig)


class ModelCheckpointCallback(pl.callbacks.ModelCheckpoint):
    """Extended model checkpoint callback with custom naming."""
    
    def __init__(self, config):
        """Initialize with configuration."""
        checkpoint_dir = Path(config['training']['checkpoint_dir'])
        checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
        super().__init__(
            dirpath=str(checkpoint_dir),
            filename='{epoch:02d}-{val_loss:.4f}',
            save_top_k=3,
            monitor='val_loss',
            mode='min',
            save_last=True,
            verbose=True
        )


class LoggingCallback(Callback):
    """Callback for additional logging during training."""
    
    def __init__(self):
        super().__init__()
    
    def on_train_epoch_start(self, trainer, pl_module):
        """Log when training epoch starts."""
        epoch = trainer.current_epoch
        lr = trainer.optimizers[0].param_groups[0]['lr']
        pl_module.logger.experiment.add_scalar('learning_rate', lr, epoch)
        
    def on_validation_epoch_end(self, trainer, pl_module):
        """Log validation metrics at the end of each epoch."""
        # Here we would log more detailed validation metrics
        # For now, it's just a placeholder as we have an empty training step
        pass