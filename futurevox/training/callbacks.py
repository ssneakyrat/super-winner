"""
Custom callbacks for PyTorch Lightning.
"""

import os
import time
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from typing import Dict, List, Optional, Any

from config.model_config import FutureVoxConfig
from utils.logging import plot_spectrogram_to_numpy, plot_alignment_to_numpy
from utils.audio import save_audio


class AudioLoggerCallback(Callback):
    """
    Callback for logging audio samples and visualizations during training.
    """
    
    def __init__(
        self,
        output_dir: str,
        log_steps: int = 1000,
        n_samples: int = 4,
        sample_rate: int = 22050
    ):
        """
        Initialize audio logger callback.
        
        Args:
            output_dir: Directory to save audio samples
            log_steps: Log every N steps
            n_samples: Number of samples to log
            sample_rate: Audio sample rate
        """
        super().__init__()
        self.output_dir = output_dir
        self.log_steps = log_steps
        self.n_samples = n_samples
        self.sample_rate = sample_rate
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Dict[str, Any],
        batch: Dict[str, torch.Tensor],
        batch_idx: int
    ) -> None:
        """
        Log audio samples at the end of training batch.
        
        Args:
            trainer: PyTorch Lightning trainer
            pl_module: PyTorch Lightning module
            outputs: Batch outputs
            batch: Input batch
            batch_idx: Batch index
        """
        # Check if it's time to log
        if trainer.global_step % self.log_steps != 0:
            return
        
        # Get model and config
        model = pl_module.model
        config = pl_module.config
        
        # Unpack batch
        phonemes = batch["phonemes"]
        phoneme_lengths = batch["phoneme_lengths"]
        durations = batch["durations"]
        f0 = batch["f0"]
        mel = batch["mel"]
        mel_lengths = batch["mel_lengths"]
        
        # Limit number of samples
        n_samples = min(self.n_samples, phonemes.size(0))
        
        # Generate with ground truth durations
        gt_outputs, _ = model(
            phonemes=phonemes[:n_samples],
            phoneme_lengths=phoneme_lengths[:n_samples],
            durations=durations[:n_samples],
            f0=f0[:n_samples],
            mel=mel[:n_samples],
            mel_lengths=mel_lengths[:n_samples],
            temperature=1.0
        )
        
        # Generate with predicted durations
        pred_outputs, _ = model(
            phonemes=phonemes[:n_samples],
            phoneme_lengths=phoneme_lengths[:n_samples],
            temperature=0.667  # Lower temperature for inference
        )
        
        # Create output directory for this step
        step_dir = os.path.join(self.output_dir, f"step_{trainer.global_step}")
        os.makedirs(step_dir, exist_ok=True)
        
        # Process each sample
        for i in range(n_samples):
            # Get lengths
            mel_len = mel_lengths[i].item()
            phoneme_len = phoneme_lengths[i].item()
            
            # Ground truth mel
            gt_mel = mel[i, :, :mel_len].cpu().numpy()
            
            # Predicted mel (with ground truth durations)
            gt_pred_mel = gt_outputs["mel_pred"][i, :mel_len].detach().cpu().numpy()
            
            # Predicted mel (with predicted durations)
            pred_mel = pred_outputs["mel_pred"][i].detach().cpu().numpy()
            
            # Ground truth waveform (not available in this implementation)
            
            # Predicted waveform (with ground truth durations)
            gt_pred_waveform = gt_outputs["waveform"][i, 0].detach().cpu().numpy()
            
            # Predicted waveform (with predicted durations)
            pred_waveform = pred_outputs["waveform"][i, 0].detach().cpu().numpy()
            
            # Save audio files
            save_audio(
                gt_pred_waveform,
                os.path.join(step_dir, f"sample_{i}_gt_pred.wav"),
                sample_rate=self.sample_rate
            )
            
            save_audio(
                pred_waveform,
                os.path.join(step_dir, f"sample_{i}_pred.wav"),
                sample_rate=self.sample_rate
            )
            
            # Log to TensorBoard
            if trainer.logger:
                # Log spectrograms
                trainer.logger.experiment.add_image(
                    f"train/mel_gt_{i}",
                    plot_spectrogram_to_numpy(gt_mel.T),
                    global_step=trainer.global_step,
                    dataformats="HWC"
                )
                
                trainer.logger.experiment.add_image(
                    f"train/mel_gt_pred_{i}",
                    plot_spectrogram_to_numpy(gt_pred_mel.T),
                    global_step=trainer.global_step,
                    dataformats="HWC"
                )
                
                trainer.logger.experiment.add_image(
                    f"train/mel_pred_{i}",
                    plot_spectrogram_to_numpy(pred_mel.T),
                    global_step=trainer.global_step,
                    dataformats="HWC"
                )
                
                # Log audio
                trainer.logger.experiment.add_audio(
                    f"train/audio_gt_pred_{i}",
                    gt_pred_waveform / np.max(np.abs(gt_pred_waveform)),
                    global_step=trainer.global_step,
                    sample_rate=self.sample_rate
                )
                
                trainer.logger.experiment.add_audio(
                    f"train/audio_pred_{i}",
                    pred_waveform / np.max(np.abs(pred_waveform)),
                    global_step=trainer.global_step,
                    sample_rate=self.sample_rate
                )


class TimingCallback(Callback):
    """
    Callback for logging timing information during training.
    """
    
    def __init__(self, log_freq: int = 1):
        """
        Initialize timing callback.
        
        Args:
            log_freq: Log frequency (in steps)
        """
        super().__init__()
        self.log_freq = log_freq
        self.start_time = None
        self.start_step = 0
        self.batch_start_time = None
    
    def on_train_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule
    ) -> None:
        """Start training timer."""
        self.start_time = time.time()
        self.start_step = trainer.global_step
    
    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Dict[str, torch.Tensor],
        batch_idx: int
    ) -> None:
        """Start batch timer."""
        self.batch_start_time = time.time()
    
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Dict[str, Any],
        batch: Dict[str, torch.Tensor],
        batch_idx: int
    ) -> None:
        """Log timing information."""
        # Calculate time per step
        if self.batch_start_time is not None:
            step_time = time.time() - self.batch_start_time
            
            # Log step time
            pl_module.log("timing/step_time", step_time, on_step=True, on_epoch=False)
            
            # Calculate and log average step time
            if trainer.global_step % self.log_freq == 0 and trainer.global_step > self.start_step:
                elapsed_time = time.time() - self.start_time
                steps = trainer.global_step - self.start_step
                avg_step_time = elapsed_time / steps
                steps_per_second = steps / elapsed_time
                
                pl_module.log("timing/avg_step_time", avg_step_time, on_step=True, on_epoch=False)
                pl_module.log("timing/steps_per_second", steps_per_second, on_step=True, on_epoch=False)
                
                # Print timing information
                trainer.logger.info(
                    f"Step {trainer.global_step}: "
                    f"{avg_step_time:.4f} s/step, "
                    f"{steps_per_second:.4f} steps/s"
                )


class BatchSizeSchedulerCallback(Callback):
    """
    Callback for dynamic batch size scheduling during training.
    Increases batch size over time to speed up training.
    """
    
    def __init__(
        self,
        initial_batch_size: int,
        max_batch_size: int,
        increase_every: int = 5000,
        increase_factor: float = 1.5
    ):
        """
        Initialize batch size scheduler.
        
        Args:
            initial_batch_size: Initial batch size
            max_batch_size: Maximum batch size
            increase_every: Increase batch size every N steps
            increase_factor: Factor to increase batch size by
        """
        super().__init__()
        self.initial_batch_size = initial_batch_size
        self.max_batch_size = max_batch_size
        self.increase_every = increase_every
        self.increase_factor = increase_factor
        
        self.current_batch_size = initial_batch_size
    
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Dict[str, Any],
        batch: Dict[str, torch.Tensor],
        batch_idx: int
    ) -> None:
        """Update batch size."""
        # Check if it's time to increase batch size
        if (
            trainer.global_step > 0 and
            trainer.global_step % self.increase_every == 0 and
            self.current_batch_size < self.max_batch_size
        ):
            # Calculate new batch size
            new_batch_size = min(
                int(self.current_batch_size * self.increase_factor),
                self.max_batch_size
            )
            
            # Update batch size if changed
            if new_batch_size > self.current_batch_size:
                self.current_batch_size = new_batch_size
                
                # Update data loaders
                if hasattr(trainer, "datamodule") and trainer.datamodule is not None:
                    trainer.datamodule.batch_size = new_batch_size
                    
                    # Reload data loaders
                    trainer.reset_train_dataloader(trainer.lightning_module)
                
                # Log batch size change
                pl_module.log("training/batch_size", new_batch_size, on_step=True, on_epoch=False)
                
                # Print batch size change
                trainer.logger.info(
                    f"Step {trainer.global_step}: "
                    f"Increased batch size to {new_batch_size}"
                )


class GenerationSamplerCallback(Callback):
    """
    Callback for sampling text-to-speech generations during training.
    Samples from a fixed set of test inputs.
    """
    
    def __init__(
        self,
        test_inputs: List[Dict[str, torch.Tensor]],
        output_dir: str,
        sample_rate: int = 22050,
        log_steps: int = 5000,
        temperatures: List[float] = [0.5, 0.8, 1.0]
    ):
        """
        Initialize generation sampler.
        
        Args:
            test_inputs: List of test inputs
            output_dir: Directory to save generation samples
            sample_rate: Audio sample rate
            log_steps: Log every N steps
            temperatures: List of sampling temperatures
        """
        super().__init__()
        self.test_inputs = test_inputs
        self.output_dir = output_dir
        self.sample_rate = sample_rate
        self.log_steps = log_steps
        self.temperatures = temperatures
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule
    ) -> None:
        """Generate samples at the end of validation epoch."""
        # Check if it's time to log
        if trainer.global_step % self.log_steps != 0:
            return
        
        # Get model
        model = pl_module.model
        
        # Create output directory for this step
        step_dir = os.path.join(self.output_dir, f"step_{trainer.global_step}")
        os.makedirs(step_dir, exist_ok=True)
        
        # Generate samples for each test input
        for i, test_input in enumerate(self.test_inputs):
            # Get input tensors
            phonemes = test_input["phonemes"].to(pl_module.device)
            phoneme_lengths = test_input["phoneme_lengths"].to(pl_module.device)
            
            # Generate with different temperatures
            for temp in self.temperatures:
                # Generate
                outputs, _ = model(
                    phonemes=phonemes.unsqueeze(0),
                    phoneme_lengths=phoneme_lengths.unsqueeze(0),
                    temperature=temp
                )
                
                # Get waveform
                waveform = outputs["waveform"][0, 0].detach().cpu().numpy()
                
                # Save audio
                save_audio(
                    waveform,
                    os.path.join(step_dir, f"sample_{i}_temp_{temp:.1f}.wav"),
                    sample_rate=self.sample_rate
                )
                
                # Log to TensorBoard
                if trainer.logger:
                    # Log audio
                    trainer.logger.experiment.add_audio(
                        f"generation/sample_{i}_temp_{temp:.1f}",
                        waveform / np.max(np.abs(waveform)),
                        global_step=trainer.global_step,
                        sample_rate=self.sample_rate
                    )
                    
                    # Log mel spectrogram
                    mel = outputs["mel_pred"][0].detach().cpu().numpy()
                    trainer.logger.experiment.add_image(
                        f"generation/mel_{i}_temp_{temp:.1f}",
                        plot_spectrogram_to_numpy(mel.T),
                        global_step=trainer.global_step,
                        dataformats="HWC"
                    )