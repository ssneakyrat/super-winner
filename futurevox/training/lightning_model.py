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
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend (avoid Qt dependency)
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import io
from utils.logging import prepare_audio_for_tensorboard
from config.model_config import FutureVoxConfig
from model.futurevox import FutureVox
from utils.logging import plot_spectrogram_to_numpy, plot_alignment_to_numpy

def safe_normalize(waveform, eps=1e-8):
    """Safely normalize waveform to [-1, 1] range."""
    peak = np.max(np.abs(waveform))
    if peak > eps:
        return waveform / peak
    return waveform

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
        Training step with added duration validation.
        
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
        
        # Check for zero durations and fix if needed
        if torch.sum(durations) == 0:
            print(f"Warning: Batch {batch_idx} has all zero durations. Setting to default values.")
            # Set to default value of 1 frame per phoneme
            durations = torch.ones_like(durations)
            
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
            0.1 * losses["kl_loss"].mean()  # Apply mean to reduce to scalar
        )
        
        # Check for NaN values
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            self.log("train/nan_detected", 1.0, prog_bar=True, batch_size=phonemes.size(0))
            # Use a small constant loss to keep training going
            total_loss = torch.tensor(1.0, device=self.device, requires_grad=True)
        else:
            self.log("train/nan_detected", 0.0, prog_bar=False, batch_size=phonemes.size(0))
            
        # Log losses - ensure KL loss is a scalar by taking the mean
        self.log("train/duration_loss", losses["duration_loss"], prog_bar=True, batch_size=phonemes.size(0))
        self.log("train/f0_loss", losses["f0_loss"], prog_bar=True)
        self.log("train/mel_loss", losses["mel_loss"], prog_bar=True)
        self.log("train/kl_loss", losses["kl_loss"].mean(), prog_bar=False)  # Apply mean to reduce to scalar
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
            0.1 * gt_losses["kl_loss"].mean()  # Small weight for KL loss - apply mean to reduce to scalar
        )
        
        # Log losses - ensure KL loss is a scalar by taking the mean
        self.log("val/duration_loss", gt_losses["duration_loss"])
        self.log("val/f0_loss", gt_losses["f0_loss"])
        self.log("val/mel_loss", gt_losses["mel_loss"])
        self.log("val/kl_loss", gt_losses["kl_loss"].mean())  # Apply mean to reduce to scalar
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
                
                if batch_idx == 0:
                    n_samples = min(self.config.logging.audio_samples, phonemes.size(0))
                    
                    for i in range(n_samples):
                        # Get lengths
                        phoneme_len = phoneme_lengths[i].item()
                        
                        # Get ground truth and predicted durations
                        gt_dur = durations[i, :phoneme_len].detach().cpu().numpy()
                        pred_dur = pred_outputs["durations_pred"][i, :phoneme_len].detach().cpu().numpy()
                        
                        # Log statistics
                        gt_sum = gt_dur.sum()
                        pred_sum = pred_dur.sum()
                        
                        # Enhanced combined duration plot
                        combined_fig = self._plot_combined_durations(gt_dur, pred_dur)
                        self.logger.experiment.add_figure(
                            f"val/duration_combined_{i}",
                            combined_fig,
                            global_step=self.global_step
                        )
                        
                        # Print debug info
                        if i == 0:  # Only print for first sample to avoid spam
                            print(f"\nDuration Debug - Sample {i}:")
                            print(f"GT Durations - Sum: {gt_sum}, Mean: {gt_dur.mean():.2f}")
                            print(f"Pred Durations - Sum: {pred_sum}, Mean: {pred_dur.mean():.2f}")
                            if phoneme_len <= 10:
                                print(f"GT Durations: {gt_dur}")
                                print(f"Pred Durations: {pred_dur}\n")
                            else:
                                print(f"GT Durations (first 10): {gt_dur[:10]}")
                                print(f"Pred Durations (first 10): {pred_dur[:10]}\n")
                                
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
                        prepare_audio_for_tensorboard(gt_waveform),
                        global_step=self.global_step,
                        sample_rate=sample_rate
                    )
                
                self.logger.experiment.add_audio(
                    f"val/audio_gt_pred_{i}",
                    prepare_audio_for_tensorboard(gt_pred_waveform),
                    global_step=self.global_step,
                    sample_rate=sample_rate
                )
                
                self.logger.experiment.add_audio(
                    f"val/audio_pred_{i}",
                    prepare_audio_for_tensorboard(pred_waveform),
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
        Plot duration alignment with improved visualization.
        
        Args:
            gt_duration: Ground truth durations
            pred_duration: Predicted durations
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 1, figsize=(10, 6))
        
        # Make sure we have non-empty data
        if np.sum(gt_duration) == 0 and np.sum(pred_duration) == 0:
            # If both are zeros, create a simple message plot
            for ax in axes:
                ax.text(0.5, 0.5, "No duration data available yet", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=14)
                ax.set_xticks([])
                ax.set_yticks([])
            plt.tight_layout()
            return fig
        
        # Set same y-axis limit for both plots
        max_duration = max(np.max(gt_duration) if len(gt_duration) > 0 else 0, 
                        np.max(pred_duration) if len(pred_duration) > 0 else 0)
        if max_duration > 0:
            max_duration = max_duration * 1.2  # Add 20% margin
        else:
            max_duration = 10  # Default if all zeros
        
        # Plot ground truth
        axes[0].bar(range(len(gt_duration)), gt_duration, color='blue', alpha=0.7)
        axes[0].set_title('Ground Truth Durations')
        axes[0].set_ylabel('Frames')
        axes[0].set_ylim(0, max_duration)
        axes[0].grid(axis='y', linestyle='--', alpha=0.7)
        
        # Plot predicted durations
        axes[1].bar(range(len(pred_duration)), pred_duration, color='red', alpha=0.7)
        axes[1].set_title('Predicted Durations')
        axes[1].set_xlabel('Phoneme Index')
        axes[1].set_ylabel('Frames')
        axes[1].set_ylim(0, max_duration)
        axes[1].grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add sum of frames and avg per phoneme as text
        gt_sum = np.sum(gt_duration)
        pred_sum = np.sum(pred_duration)
        gt_avg = gt_sum / len(gt_duration) if len(gt_duration) > 0 else 0
        pred_avg = pred_sum / len(pred_duration) if len(pred_duration) > 0 else 0
        
        axes[0].text(0.02, 0.92, f'Sum: {gt_sum:.1f} frames, Avg: {gt_avg:.1f}/phoneme', 
                    transform=axes[0].transAxes, fontsize=9)
        axes[1].text(0.02, 0.92, f'Sum: {pred_sum:.1f} frames, Avg: {pred_avg:.1f}/phoneme', 
                    transform=axes[1].transAxes, fontsize=9)
        
        plt.tight_layout()
        
        return fig
    
    # Add this new method to FutureVoxLightning class
    def _plot_combined_durations(self, gt_duration, pred_duration):
        """
        Plot ground truth and predicted durations on the same axes for direct comparison.
        
        Args:
            gt_duration: Ground truth durations
            pred_duration: Predicted durations
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 5))
        
        # Use minimal length to avoid index errors
        min_len = min(len(gt_duration), len(pred_duration))
        x = np.arange(min_len)
        
        # Set width of the bars
        width = 0.4
        
        # Plot bars side by side
        rects1 = ax.bar(x - width/2, gt_duration[:min_len], width, label='Ground Truth', alpha=0.7, color='blue')
        rects2 = ax.bar(x + width/2, pred_duration[:min_len], width, label='Predicted', alpha=0.7, color='red')
        
        # Add some text for labels, title and custom x-axis tick labels
        ax.set_xlabel('Phoneme Index')
        ax.set_ylabel('Duration (frames)')
        ax.set_title('Ground Truth vs Predicted Durations')
        if min_len <= 20:  # Only show ticks for reasonable number of phonemes
            ax.set_xticks(x)
        ax.legend()
        
        # Calculate difference metrics
        abs_diff = np.abs(gt_duration[:min_len] - pred_duration[:min_len])
        mean_abs_error = np.mean(abs_diff)
        
        # Add difference information
        ax.text(0.02, 0.92, f'Mean Absolute Error: {mean_abs_error:.2f} frames', 
                transform=ax.transAxes, fontsize=10)
        
        plt.tight_layout()
        
        return fig