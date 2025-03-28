import os
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image
import librosa
import librosa.display

from models.model import FutureVoxSinger


class FutureVoxSingerLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for FutureVox-Singer training and evaluation.
    """
    
    def __init__(self, config, num_phonemes=100):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters()
        self.config = config
        
        # Initialize model
        self.model = FutureVoxSinger(
            config=config,
            num_phonemes=num_phonemes,
            num_notes=config.get('model', {}).get('num_notes', 128),
            num_singers=config.get('model', {}).get('num_singers', 10)
        )
        
        # Store configuration
        self.learning_rate = float(config['training']['learning_rate'])
        self.progressive_training = config.get('training', {}).get('progressive', True)
        
        # Track current training phase
        self.current_phase = 'phoneme_encoder'  # Start with phoneme encoder
        
        # For progressive training
        self.phase_epochs = config.get('training', {}).get('phase_epochs', {
            'phoneme_encoder': 10,
            'variance_adaptor': 10,
            'acoustic_decoder': 20,
            'vocoder': 10,
            'all': 50
        })
    
    def forward(self, batch, phase='all'):
        """Forward pass through model."""
        return self.model(batch, phase=phase)
    
    def training_step(self, batch, batch_idx):
        # Determine current training phase based on epoch
        if self.progressive_training:
            epoch = self.current_epoch
            
            if epoch < self.phase_epochs['phoneme_encoder']:
                phase = 'phoneme_encoder'
            elif epoch < (self.phase_epochs['phoneme_encoder'] + self.phase_epochs['variance_adaptor']):
                phase = 'variance_adaptor'
            elif epoch < (self.phase_epochs['phoneme_encoder'] + self.phase_epochs['variance_adaptor'] + 
                        self.phase_epochs['acoustic_decoder']):
                phase = 'acoustic_decoder'
            elif epoch < (self.phase_epochs['phoneme_encoder'] + self.phase_epochs['variance_adaptor'] + 
                        self.phase_epochs['acoustic_decoder'] + self.phase_epochs['vocoder']):
                phase = 'vocoder'
            else:
                phase = 'all'
                
            # Only update phase if it's changing
            if phase != self.current_phase:
                print(f"Transitioning from {self.current_phase} to {phase} phase")
                self.current_phase = phase
        else:
            phase = 'all'
        
        # Forward pass with the appropriate phase
        output_dict = self.model(batch, phase=phase)
        
        # Calculate losses
        loss_dict, total_loss = self.model.calculate_losses(output_dict, batch)
        
        # Get optimizers
        opt = self.optimizers()
        
        # Zero gradients and backward pass
        opt.zero_grad()
        self.manual_backward(total_loss)
        
        # Apply gradient clipping
        gradient_clip_val = self.config['training'].get('grad_clip_val', 1.0)
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=gradient_clip_val)
        
        # Step optimizer
        opt.step()
        
        # Log losses
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                self.log(f'train/{key}', value.item(), prog_bar=True)
            else:
                self.log(f'train/{key}', value, prog_bar=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step with visualization.
        
        Args:
            batch: Input batch from dataloader
            batch_idx: Index of the batch
            
        Returns:
            val_loss: Validation loss
        """
        # Forward pass through model
        output_dict = self.model(batch, phase=self.current_phase)
        
        # Calculate losses
        loss_dict, total_loss = self.model.calculate_losses(output_dict, batch)
        
        # Log validation losses
        for key, value in loss_dict.items():
            self.log(f'val/{key}', value, prog_bar=True)
        
        # Generate visualizations for a few samples
        if batch_idx == 0:
            # Create visualizations and log to TensorBoard
            self._log_visualizations(output_dict, batch)
        
        return total_loss
    
    def configure_optimizers(self):
        """
        Configure optimizers with support for vocoder discriminator.
        
        Returns:
            optimizers: List of optimizers
            lr_schedulers: List of learning rate schedulers
        """
        # Parameters for the main model
        model_params = [p for name, p in self.model.named_parameters() 
                       if not name.startswith('vocoder.mpd') and not name.startswith('vocoder.msd')]
        
        # Parameters for the vocoder discriminator
        disc_params = [p for name, p in self.model.named_parameters() 
                     if name.startswith('vocoder.mpd') or name.startswith('vocoder.msd')]
        
        # Create optimizers
        model_optimizer = optim.AdamW(
            model_params, 
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01
        )
        
        # Create second optimizer for discriminator if needed
        optimizers = [model_optimizer]
        if self.model.train_vocoder and len(disc_params) > 0:
            disc_optimizer = optim.AdamW(
                disc_params,
                lr=self.learning_rate,
                betas=(0.9, 0.999),
                weight_decay=0.01
            )
            optimizers.append(disc_optimizer)
        
        # Learning rate schedulers
        schedulers = []
        for optimizer in optimizers:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )
            
            schedulers.append({
                'scheduler': scheduler,
                'monitor': 'val/total_loss',
                'interval': 'epoch',
                'frequency': 1
            })
        
        return optimizers, schedulers
    
    def _log_visualizations(self, output_dict, batch, max_samples=2):
        """
        Create and log visualizations to TensorBoard.
        
        Args:
            output_dict: Dictionary with model outputs
            batch: Dictionary with input batch
            max_samples: Maximum number of samples to visualize
        """
        # Sample indices to visualize (up to max_samples)
        batch_size = batch['phone_indices'].size(0)
        sample_indices = range(min(batch_size, max_samples))
        
        # 1. Visualize phoneme encoder (attention maps)
        if 'phoneme_attention_weights' in output_dict and output_dict['phoneme_attention_weights']:
            self._log_attention_maps(output_dict, batch, sample_indices)
        
        # 2. Visualize variance predictions
        if 'f0_contour' in output_dict and 'f0_values' in batch:
            self._log_f0_predictions(output_dict, batch, sample_indices)
            
        if 'durations' in output_dict and 'durations' in batch:
            self._log_duration_predictions(output_dict, batch, sample_indices)
        
        # 3. Visualize mel spectrograms
        if 'mel_postnet' in output_dict and 'mel_spectrograms' in batch:
            self._log_mel_spectrograms(output_dict, batch, sample_indices)
        
        # 4. Visualize waveforms and spectrograms
        if 'waveform_pred' in output_dict and 'audio' in batch:
            self._log_waveforms(output_dict, batch, sample_indices)
    
    def _log_attention_maps(self, output_dict, batch, sample_indices):
        """
        Log attention maps visualization to TensorBoard.
        """
        # This is a placeholder for attention visualization
        # Would use matplotlib to create heatmaps of attention weights
        pass
    
    def _log_f0_predictions(self, output_dict, batch, sample_indices):
        """
        Log F0 prediction visualization to TensorBoard.
        """
        f0_pred = output_dict['f0_contour']
        f0_target = batch['f0_values']
        
        for i in sample_indices:
            # Get lengths (non-padded frames)
            mel_mask = batch['mel_masks'][i] if 'mel_masks' in batch else None
            if mel_mask is not None:
                valid_len = torch.sum(~mel_mask).item()
            else:
                valid_len = f0_pred.size(1)
            
            # Create figure for F0 visualization
            fig, ax = plt.subplots(figsize=(10, 4))
            
            # Get data for plotting
            f0_pred_i = f0_pred[i, :valid_len].cpu().numpy()
            f0_target_i = f0_target[i, :valid_len].cpu().numpy()
            
            # Plot F0 curves
            times = np.arange(valid_len) * self.config['audio']['hop_length'] / self.config['audio']['sample_rate']
            ax.plot(times, f0_pred_i, label='Predicted F0', color='blue', alpha=0.8)
            ax.plot(times, f0_target_i, label='Ground Truth F0', color='red', alpha=0.8)
            
            # Add labels and title
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Frequency (Hz)')
            ax.set_title(f'F0 Prediction Sample {i}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Convert figure to image and log to TensorBoard
            self._log_figure_to_tensorboard(fig, f'F0_prediction/sample_{i}')
    
    def _log_duration_predictions(self, output_dict, batch, sample_indices):
        """
        Log duration prediction visualization to TensorBoard.
        """
        durations_pred = output_dict['durations']
        durations_target = batch['durations']
        
        for i in sample_indices:
            # Get lengths (non-padded phones)
            phone_mask = batch['phone_masks'][i] if 'phone_masks' in batch else None
            if phone_mask is not None:
                valid_len = torch.sum(~phone_mask).item()
            else:
                valid_len = durations_pred.size(1)
            
            # Create figure for duration visualization
            fig, ax = plt.subplots(figsize=(10, 4))
            
            # Get data for plotting
            dur_pred_i = durations_pred[i, :valid_len].cpu().numpy()
            dur_target_i = durations_target[i, :valid_len].cpu().numpy()
            
            # Plot durations as bar charts
            x = np.arange(valid_len)
            width = 0.35
            
            ax.bar(x - width/2, dur_target_i, width, label='Ground Truth', alpha=0.7)
            ax.bar(x + width/2, dur_pred_i, width, label='Predicted', alpha=0.7)
            
            # Add labels and title
            ax.set_xlabel('Phoneme Index')
            ax.set_ylabel('Duration (frames)')
            ax.set_title(f'Duration Prediction Sample {i}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Convert figure to image and log to TensorBoard
            self._log_figure_to_tensorboard(fig, f'Duration_prediction/sample_{i}')
    
    def _log_mel_spectrograms(self, output_dict, batch, sample_indices):
        """
        Log mel spectrogram visualization to TensorBoard.
        """
        mel_pred = output_dict['mel_postnet']
        mel_target = batch['mel_spectrograms']
        
        for i in sample_indices:
            # Get lengths (non-padded frames)
            mel_mask = batch['mel_masks'][i] if 'mel_masks' in batch else None
            if mel_mask is not None:
                valid_len = torch.sum(~mel_mask).item()
            else:
                valid_len = mel_pred.size(2)
            
            # Create figure for mel spectrogram visualization
            fig, axs = plt.subplots(2, 1, figsize=(10, 8))
            
            # Get data for plotting
            mel_pred_i = mel_pred[i, :, :valid_len].cpu().numpy()
            mel_target_i = mel_target[i, :, :valid_len].cpu().numpy()
            
            # Plot spectrograms
            librosa.display.specshow(
                mel_target_i,
                y_axis='mel',
                x_axis='time',
                sr=self.config['audio']['sample_rate'],
                hop_length=self.config['audio']['hop_length'],
                ax=axs[0]
            )
            axs[0].set_title('Ground Truth Mel Spectrogram')
            
            librosa.display.specshow(
                mel_pred_i,
                y_axis='mel',
                x_axis='time',
                sr=self.config['audio']['sample_rate'],
                hop_length=self.config['audio']['hop_length'],
                ax=axs[1]
            )
            axs[1].set_title('Predicted Mel Spectrogram')
            
            # Add colorbar
            fig.colorbar(axs[0].images[0], ax=axs[0], format='%+2.0f dB')
            fig.colorbar(axs[1].images[0], ax=axs[1], format='%+2.0f dB')
            
            plt.tight_layout()
            
            # Convert figure to image and log to TensorBoard
            self._log_figure_to_tensorboard(fig, f'Mel_spectrogram/sample_{i}')
    
    def _log_waveforms(self, output_dict, batch, sample_indices):
        """
        Log waveform visualization to TensorBoard.
        """
        waveform_pred = output_dict['waveform_pred']
        waveform_target = batch['audio']
        
        for i in sample_indices:
            # Create figure for waveform visualization
            fig, axs = plt.subplots(2, 1, figsize=(10, 6))
            
            # Get data for plotting (trim to minimum length)
            wav_pred_i = waveform_pred[i, 0].cpu().numpy()
            wav_target_i = waveform_target[i, 0].cpu().numpy()
            min_len = min(len(wav_pred_i), len(wav_target_i))
            
            # Calculate time axis
            times = np.arange(min_len) / self.sample_rate
            
            # Plot waveforms
            axs[0].plot(times, wav_target_i[:min_len], color='blue', alpha=0.8)
            axs[0].set_title('Ground Truth Waveform')
            axs[0].set_ylim([-1, 1])
            
            axs[1].plot(times, wav_pred_i[:min_len], color='red', alpha=0.8)
            axs[1].set_title('Generated Waveform')
            axs[1].set_ylim([-1, 1])
            
            # Add labels
            axs[1].set_xlabel('Time (s)')
            for ax in axs:
                ax.set_ylabel('Amplitude')
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Convert figure to image and log to TensorBoard
            self._log_figure_to_tensorboard(fig, f'Waveform/sample_{i}')
            
            # Also log audio to TensorBoard (both predicted and ground truth)
            self.logger.experiment.add_audio(
                f'Audio/sample_{i}_pred',
                wav_pred_i.reshape(1, -1),
                self.global_step,
                sample_rate=self.sample_rate
            )
            
            self.logger.experiment.add_audio(
                f'Audio/sample_{i}_target',
                wav_target_i.reshape(1, -1),
                self.global_step,
                sample_rate=self.sample_rate
            )
    
    def _log_figure_to_tensorboard(self, fig, tag):
        """
        Convert matplotlib figure to image and log to TensorBoard.
        
        Args:
            fig: Matplotlib figure
            tag: Tag for TensorBoard visualization
        """
        # Convert figure to image
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        
        # Open image with PIL and convert to tensor
        image = Image.open(buf).convert('RGB')
        image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1)  # HWC -> CHW
        
        # Log to TensorBoard
        self.logger.experiment.add_image(
            tag,
            image_tensor,
            self.global_step
        )
        
        plt.close(fig)