"""
Logging utilities for FutureVox.
Includes functions for visualizing spectrograms and alignments.
"""

import os
import logging
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend (avoid Qt dependency)
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import io
from PIL import Image
from typing import Optional, Union, List, Tuple

# Configure logging
logger = logging.getLogger('futurevox')


def setup_logger(log_dir: str, log_file: str = 'futurevox.log', level: int = logging.INFO) -> None:
    """
    Set up logger with file and console handlers.
    
    Args:
        log_dir: Directory for log files
        log_file: Log file name
        level: Logging level
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure logger
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create file handler
    file_handler = logging.FileHandler(
        os.path.join(log_dir, log_file)
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Logger initialized. Logs will be saved to: {os.path.join(log_dir, log_file)}")


def plot_spectrogram_to_numpy(spectrogram: np.ndarray, title: Optional[str] = None) -> np.ndarray:
    """
    Creates a spectrogram plot as a numpy array.
    
    Args:
        spectrogram: [n_frames, n_mels] Mel spectrogram
        title: Plot title
        
    Returns:
        [H, W, 3] RGB image as numpy array
    """
    fig, ax = plt.subplots(figsize=(10, 2.5))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    
    if title is not None:
        ax.set_title(title)
    
    plt.tight_layout()
    
    # Convert figure to numpy array
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    return data


def plot_alignment_to_numpy(alignment: np.ndarray, title: Optional[str] = None) -> np.ndarray:
    """
    Creates an alignment plot as a numpy array.
    
    Args:
        alignment: [text_len, n_frames] Alignment matrix
        title: Plot title
        
    Returns:
        [H, W, 3] RGB image as numpy array
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(alignment, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    
    if title is not None:
        ax.set_title(title)
    
    plt.tight_layout()
    
    # Convert figure to numpy array
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    return data


def save_figure_to_numpy(figure: Figure) -> np.ndarray:
    """
    Converts a matplotlib figure to a numpy array.
    
    Args:
        figure: Matplotlib figure
        
    Returns:
        [H, W, 3] RGB image as numpy array
    """
    # Save figure to a buffer
    buf = io.BytesIO()
    figure.savefig(buf, format="png")
    buf.seek(0)
    
    # Load image and convert to numpy array
    img = Image.open(buf)
    img_array = np.array(img)
    
    # Close the buffer
    buf.close()
    
    return img_array


def plot_audio_features(
    mel: Optional[np.ndarray] = None,
    f0: Optional[np.ndarray] = None,
    energy: Optional[np.ndarray] = None,
    waveform: Optional[np.ndarray] = None
) -> Figure:
    """
    Creates a plot with multiple audio features.
    
    Args:
        mel: [n_frames, n_mels] Mel spectrogram
        f0: [n_frames] F0 contour
        energy: [n_frames] Energy contour
        waveform: [n_samples] Audio waveform
        
    Returns:
        Matplotlib figure
    """
    n_plots = sum(x is not None for x in [mel, f0, energy, waveform])
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, n_plots * 2.5), squeeze=False)
    
    plot_idx = 0
    
    if mel is not None:
        im = axes[plot_idx, 0].imshow(mel.T, aspect="auto", origin="lower", interpolation="none")
        axes[plot_idx, 0].set_title("Mel Spectrogram")
        plt.colorbar(im, ax=axes[plot_idx, 0])
        plot_idx += 1
    
    if f0 is not None:
        axes[plot_idx, 0].plot(f0)
        axes[plot_idx, 0].set_title("F0 Contour")
        axes[plot_idx, 0].set_ylim(0, np.max(f0) * 1.1)
        plot_idx += 1
    
    if energy is not None:
        axes[plot_idx, 0].plot(energy)
        axes[plot_idx, 0].set_title("Energy Contour")
        plot_idx += 1
    
    if waveform is not None:
        axes[plot_idx, 0].plot(waveform)
        axes[plot_idx, 0].set_title("Waveform")
        plot_idx += 1
    
    plt.tight_layout()
    
    return fig


def plot_mel_f0_alignment(
    mel: np.ndarray,
    f0: Optional[np.ndarray] = None,
    alignment: Optional[np.ndarray] = None,
    title: Optional[str] = None
) -> Figure:
    """
    Creates a plot with mel spectrogram, F0 contour, and alignment.
    
    Args:
        mel: [n_frames, n_mels] Mel spectrogram
        f0: [n_frames] F0 contour
        alignment: [text_len, n_frames] Alignment matrix
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    n_plots = 1 + (f0 is not None) + (alignment is not None)
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, n_plots * 2.5), squeeze=False)
    
    if title is not None:
        fig.suptitle(title)
    
    # Plot mel spectrogram
    im = axes[0, 0].imshow(mel.T, aspect="auto", origin="lower", interpolation="none")
    axes[0, 0].set_title("Mel Spectrogram")
    plt.colorbar(im, ax=axes[0, 0])
    
    plot_idx = 1
    
    # Plot F0 contour
    if f0 is not None:
        axes[plot_idx, 0].plot(f0)
        axes[plot_idx, 0].set_title("F0 Contour")
        axes[plot_idx, 0].set_ylim(0, np.max(f0) * 1.1)
        plot_idx += 1
    
    # Plot alignment
    if alignment is not None:
        im = axes[plot_idx, 0].imshow(alignment, aspect="auto", origin="lower", interpolation="none")
        axes[plot_idx, 0].set_title("Alignment")
        plt.colorbar(im, ax=axes[plot_idx, 0])
    
    plt.tight_layout()
    
    return fig


class TensorBoardLogger:
    """Wrapper around TensorBoard for logging audio, spectrograms, etc."""
    
    def __init__(self, log_dir: str, name: Optional[str] = None):
        """
        Initialize TensorBoard logger.
        
        Args:
            log_dir: Log directory
            name: Run name
        """
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            raise ImportError("Please install tensorboard to use TensorBoardLogger")
        
        # Create log directory
        if name is not None:
            log_dir = os.path.join(log_dir, name)
        
        os.makedirs(log_dir, exist_ok=True)
        
        self.writer = SummaryWriter(log_dir)
        logger.info(f"TensorBoard logger initialized. Logs will be saved to: {log_dir}")
    
    def log_audio(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        tag: str,
        global_step: int
    ) -> None:
        """
        Log audio to TensorBoard.
        
        Args:
            waveform: Audio waveform
            sample_rate: Audio sample rate
            tag: Tag for TensorBoard
            global_step: Global step
        """
        # Normalize waveform
        waveform = waveform / np.max(np.abs(waveform))
        
        # Add audio to TensorBoard
        self.writer.add_audio(tag, waveform, global_step, sample_rate=sample_rate)
    
    def log_spectrogram(
        self,
        spectrogram: np.ndarray,
        tag: str,
        global_step: int,
        title: Optional[str] = None
    ) -> None:
        """
        Log spectrogram to TensorBoard.
        
        Args:
            spectrogram: Mel spectrogram
            tag: Tag for TensorBoard
            global_step: Global step
            title: Plot title
        """
        # Create spectrogram image
        spec_img = plot_spectrogram_to_numpy(spectrogram, title)
        
        # Add image to TensorBoard
        self.writer.add_image(tag, spec_img, global_step, dataformats="HWC")
    
    def log_alignment(
        self,
        alignment: np.ndarray,
        tag: str,
        global_step: int,
        title: Optional[str] = None
    ) -> None:
        """
        Log alignment to TensorBoard.
        
        Args:
            alignment: Alignment matrix
            tag: Tag for TensorBoard
            global_step: Global step
            title: Plot title
        """
        # Create alignment image
        align_img = plot_alignment_to_numpy(alignment, title)
        
        # Add image to TensorBoard
        self.writer.add_image(tag, align_img, global_step, dataformats="HWC")
    
    def log_figure(
        self,
        figure: Figure,
        tag: str,
        global_step: int
    ) -> None:
        """
        Log matplotlib figure to TensorBoard.
        
        Args:
            figure: Matplotlib figure
            tag: Tag for TensorBoard
            global_step: Global step
        """
        # Convert figure to numpy array
        fig_img = save_figure_to_numpy(figure)
        
        # Add image to TensorBoard
        self.writer.add_image(tag, fig_img, global_step, dataformats="HWC")
        
        # Close figure
        plt.close(figure)
    
    def log_scalar(
        self,
        scalar_value: Union[float, int, np.ndarray, torch.Tensor],
        tag: str,
        global_step: int
    ) -> None:
        """
        Log scalar to TensorBoard.
        
        Args:
            scalar_value: Scalar value
            tag: Tag for TensorBoard
            global_step: Global step
        """
        # Convert tensor to scalar if needed
        if isinstance(scalar_value, torch.Tensor):
            scalar_value = scalar_value.item()
        
        # Add scalar to TensorBoard
        self.writer.add_scalar(tag, scalar_value, global_step)
    
    def log_histogram(
        self,
        values: Union[np.ndarray, torch.Tensor],
        tag: str,
        global_step: int
    ) -> None:
        """
        Log histogram to TensorBoard.
        
        Args:
            values: Values for histogram
            tag: Tag for TensorBoard
            global_step: Global step
        """
        # Add histogram to TensorBoard
        self.writer.add_histogram(tag, values, global_step)
    
    def close(self) -> None:
        """Close TensorBoard writer."""
        self.writer.close()