"""
PyTorch Lightning implementation for FutureVox.
"""

import os
import torch
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import io

from config.model_config import FutureVoxConfig
from model.futurevox import FutureVox
from utils.logging import plot_spectrogram_to_numpy, plot_alignment_to_numpy


class FutureVoxLightning(pl.LightningModule):
    """PyTorch Lightning module for FutureVox."""
    
    def __init__(self, config_path=None, config=None, n_vocab=100):
        """
        Initialize lightning module.
        
        Args:
            config_path: Path to configuration file
            config: Configuration object (if config_path is None)
            n_vocab: Vocabulary size for phoneme tokens
        """
        super().__init__()
        
        # Load configuration
        if config is not None:
            self.config = config
        elif config_path is not None:
            self.config = FutureVoxConfig.from_yaml(config_path)
        else:
            raise ValueError("Either config_path or config must be provided")
        
        # Save hyperparameters
        self.save_hyperparameters(ignore=["config"])
        
        # Initialize model
        self.model = FutureVox(self.config, n_vocab)
    
    def forward(self, **kwargs):
        """Forward pass."""
        return self.model(**kwargs)
    
    def configure_optimizers(self):
        """Configure optimizers and schedulers."""
        optimizer = AdamW(
            self.parameters(),
            lr=self.config.training.learning_rate,
            betas=tuple(self.config.training.betas),
            weight_decay=self.config.training.optimizer.weight_decay
        )
        
        scheduler = ExponentialLR(
            optimizer,
            gamma=self.config.training.scheduler.gamma
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch"
            }
        }
    
    def training_step(self, batch, batch_idx):
        """
        Training step.
        
        Args:
            batch: Input batch
            batch_idx: Batch index
            
        Returns:
            Loss dictionary
        """
        # Unpack batch
        phonemes = batch["phonemes"]
        phoneme_lengths = batch["phoneme_lengths"]
        durations = batch["durations"]
        f0 = batch["f0"]
        mel = batch["mel"]
        mel_lengths = batch["mel_lengths"]
        
        # Forward pass
        outputs, losses = self.model(
            phonemes=phonemes,
            phoneme_lengths=phoneme_lengths,
            durations=durations,
            f0=f0,
            mel=mel,
            mel_lengths=mel_lengths,
            temperature=1.0
        )
        
        # Calculate total loss
        total_loss = (
            self.config.training.loss_weights.duration * losses["duration_loss"] +
            self.config.training.loss_weights.f0 * losses["f0_loss"] +
            self.config.training.loss_weights.mel * losses["mel_loss"] +
            0.1 * losses["kl_loss"]  # Small weight for KL loss
        )
        
        # Log losses
        self.log("train/duration_loss", losses["duration_loss"], prog_bar=True)
        self.log("train/f0_loss", losses["f0_loss"], prog_bar=True)
        self.log("train/mel_loss", losses["mel_loss"], prog_bar=True)
        self.log("train/kl_loss", losses["kl_loss"])
        self.log("train/total_loss", total_loss, prog_bar=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step.
        
        Args:
            batch: Input batch
            batch_idx: Batch index
            
        Returns:
            Loss dictionary
        """
        # Unpack batch
        phonemes = batch["phonemes"]
        phoneme_lengths = batch["phoneme_lengths"]
        durations = batch["durations"]
        f0 = batch["f0"]
        mel = batch["mel"]
        mel_lengths = batch["mel_lengths"]
        
        # Forward pass with ground truth inputs
        gt_outputs, gt_losses = self.model(
            phonemes=phonemes,
            phoneme_lengths=phoneme_lengths,
            durations=durations,
            f0=f0,
            mel=mel,
            mel_lengths=mel_lengths,
            temperature=1.0
        )
        
        # Forward pass with predicted inputs (inference mode)
        pred_outputs, _ = self.model(
            phonemes=phonemes,
            phoneme_lengths=phoneme_lengths,
            temperature=0.667  # Lower temperature for inference
        )
        
        # Calculate total loss
        total_loss = (
            self.config.training.loss_weights.duration * gt_losses["duration_loss"] +
            self.config.training.loss_weights.f0 * gt_losses["f0_loss"] +
            self.config.training.loss_weights.mel * gt_losses["mel_loss"] +
            0.1 * gt_losses["kl_loss"]  # Small weight for KL loss
        )
        
        # Log losses
        self.log("val/duration_loss", gt_losses["duration_loss"])
        self.log("val/f0_loss", gt_losses["f0_loss"])
        self.log("val/mel_loss", gt_losses["mel_loss"])
        self.log("val/kl_loss", gt_losses["kl_loss"])
        self.log("val/total_loss", total_loss)
        
        # Log spectrograms and audio for first few samples
        if batch_idx == 0:
            n_samples = min(self.config.logging.audio_samples, phonemes.size(0))
            
            for i in range(n_samples):
                # Get lengths
                mel_len = mel_lengths[i].item()
                phoneme_len = phoneme_lengths[i].item()
                
                # Prepare ground truth mel
                gt_mel = mel[i, :, :mel_len].transpose(0, 1).detach().cpu().numpy()
                
                # Prepare predicted mel (with ground truth durations)
                gt_pred_mel = gt_outputs["mel_pred"][i, :mel_len].detach().cpu().numpy()
                
                # Prepare predicted mel (with predicted durations)
                pred_mel_len = pred_outputs["mel_pred"].shape[1]
                pred_mel = pred_outputs["mel_pred"][i].detach().cpu().numpy()
                
                # Prepare ground truth waveform
                gt_waveform = None  # Not available in this implementation
                
                # Prepare predicted waveform (with ground truth durations)
                gt_pred_waveform = gt_outputs["waveform"][i, 0].detach().cpu().numpy()
                
                # Prepare predicted waveform (with predicted durations)
                pred_waveform = pred_outputs["waveform"][i, 0].detach().cpu().numpy()
                
                # Prepare duration alignment
                gt_duration = durations[i, :phoneme_len].detach().cpu().numpy()
                pred_duration = pred_outputs["durations_pred"][i, :phoneme_len].detach().cpu().numpy()
                
                # Log spectrograms
                self.logger.experiment.add_figure(
                    f"val/mel_spectrograms_{i}",
                    self._plot_spectrograms(gt_mel, gt_pred_mel, pred_mel),
                    global_step=self.global_step
                )
                
                # Log alignment
                self.logger.experiment.add_figure(
                    f"val/duration_alignment_{i}",
                    self._plot_alignment(gt_duration, pred_duration),
                    global_step=self.global_step
                )
                
                # Log audio
                sample_rate = self.config.data.sample_rate
                
                if gt_waveform is not None:
                    self.logger.experiment.add_audio(
                        f"val/audio_gt_{i}",
                        gt_waveform / np.max(np.abs(gt_waveform)),
                        global_step=self.global_step,
                        sample_rate=sample_rate
                    )
                
                self.logger.experiment.add_audio(
                    f"val/audio_gt_pred_{i}",
                    gt_pred_waveform / np.max(np.abs(gt_pred_waveform)),
                    global_step=self.global_step,
                    sample_rate=sample_rate
                )
                
                self.logger.experiment.add_audio(
                    f"val/audio_pred_{i}",
                    pred_waveform / np.max(np.abs(pred_waveform)),
                    global_step=self.global_step,
                    sample_rate=sample_rate
                )
        
        return total_loss
    
    def _plot_spectrograms(self, gt_mel, gt_pred_mel, pred_mel):
        """
        Plot ground truth and predicted spectrograms.
        
        Args:
            gt_mel: Ground truth mel-spectrogram
            gt_pred_mel: Predicted mel-spectrogram (with ground truth durations)
            pred_mel: Predicted mel-spectrogram (with predicted durations)
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(3, 1, figsize=(10, 8))
        
        im = axes[0].imshow(gt_mel.T, aspect='auto', origin='lower')
        axes[0].set_title('Ground Truth Mel-Spectrogram')
        fig.colorbar(im, ax=axes[0])
        
        im = axes[1].imshow(gt_pred_mel.T, aspect='auto', origin='lower')
        axes[1].set_title('Predicted Mel-Spectrogram (GT Durations)')
        fig.colorbar(im, ax=axes[1])
        
        im = axes[2].imshow(pred_mel.T, aspect='auto', origin='lower')
        axes[2].set_title('Predicted Mel-Spectrogram (Pred Durations)')
        fig.colorbar(im, ax=axes[2])
        
        plt.tight_layout()
        
        return fig
    
    def _plot_alignment(self, gt_duration, pred_duration):
        """
        Plot duration alignment.
        
        Args:
            gt_duration: Ground truth durations
            pred_duration: Predicted durations
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 1, figsize=(10, 6))
        
        axes[0].bar(range(len(gt_duration)), gt_duration)
        axes[0].set_title('Ground Truth Durations')
        axes[0].set_ylabel('Frames')
        
        axes[1].bar(range(len(pred_duration)), pred_duration)
        axes[1].set_title('Predicted Durations')
        axes[1].set_xlabel('Phoneme Index')
        axes[1].set_ylabel('Frames')
        
        plt.tight_layout()
        
        return fig