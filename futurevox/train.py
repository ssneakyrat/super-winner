"""
Modified training script for FutureVox with fixed DataModule usage.
"""

import os
import sys
import argparse
import json
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from config.model_config import FutureVoxConfig
from data.LightSingerDataset import LightSingerDataModule
from training.lightning_model import FutureVoxLightning
from utils.logging import setup_logger


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train FutureVox model")
    
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--data_dir", type=str, default=None,
        help="Path to data directory (default: use datasets_root from config)"
    )
    
    parser.add_argument(
        "--output_dir", type=str, default="./outputs",
        help="Path to output directory"
    )
    
    parser.add_argument(
        "--log_dir", type=str, default=None,
        help="Path to log directory (default: output_dir/logs)"
    )
    
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to checkpoint for resuming training"
    )
    
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed (overrides config)"
    )
    
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help="Batch size (overrides config)"
    )
    
    parser.add_argument(
        "--precision", type=int, default=None,
        choices=[16, 32],
        help="Floating point precision (overrides config)"
    )
    
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Number of training epochs (overrides config)"
    )
    
    parser.add_argument(
        "--limit_dataset", type=int, default=None,
        help="Limit dataset size for debugging"
    )
    
    parser.add_argument(
        "--num_workers", type=int, default=4,
        help="Number of data loader workers"
    )
    
    parser.add_argument(
        "--save_every", type=int, default=None,
        help="Save checkpoint every N steps (overrides config)"
    )
    
    parser.add_argument(
        "--eval_every", type=int, default=None,
        help="Run validation every N steps (overrides config)"
    )
    
    parser.add_argument(
        "--no_precomputed_mels", action="store_true",
        help="Disable use of pre-computed mel spectrograms"
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set log directory
    log_dir = args.log_dir if args.log_dir else os.path.join(args.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Set up logger
    setup_logger(log_dir)
    
    # Load configuration
    config = FutureVoxConfig.from_yaml(args.config)
    
    # Override configuration with command-line arguments
    if args.seed is not None:
        config.training.seed = args.seed
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.precision is not None:
        config.training.precision = args.precision
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.save_every is not None:
        config.training.save_every = args.save_every
    if args.eval_every is not None:
        config.training.eval_every = args.eval_every
    
    # Set random seed
    pl.seed_everything(config.training.seed)
    
    # Determine data directory - either from args or from config
    if args.data_dir:
        data_dir = args.data_dir
    else:
        data_dir = config.data.datasets_root
        print(f"Using datasets_root from config: {data_dir}")
        
    print(f"datasets_root: {data_dir}")
    # Create data module
    data_module = LightSingerDataModule(
        data_dir=data_dir,
        config=config.data,
        batch_size=config.training.batch_size,
        num_workers=args.num_workers,
        limit_dataset_size=args.limit_dataset,
        use_precomputed_mels=not args.no_precomputed_mels
    )

    # Explicitly call setup to load phoneme dictionary
    data_module.setup("fit")

    # Load phoneme dictionary to get vocabulary size
    phoneme_dict_path = os.path.join(data_dir, config.data.phoneme_dict_file)
    if os.path.exists(phoneme_dict_path):
        with open(phoneme_dict_path, "r") as f:
            phoneme_dict = json.load(f)
        n_vocab = len(phoneme_dict) + 1  # +1 for padding/unknown token
    else:
        print(f"Warning: Phoneme dictionary not found at {phoneme_dict_path}. Using default vocabulary size.")
        n_vocab = 100  # Default fallback value
    
    # Create model
    if args.checkpoint:
        # Resume from checkpoint
        print(f"Resuming from checkpoint: {args.checkpoint}")
        model = FutureVoxLightning.load_from_checkpoint(
            checkpoint_path=args.checkpoint,
            config=config,
            n_vocab=n_vocab
        )
    else:
        # Create new model
        model = FutureVoxLightning(
            config=config,
            n_vocab=n_vocab
        )
    
    # Create callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(args.output_dir, "checkpoints"),
            filename="futurevox-{epoch:04d}-{val/total_loss:.4f}",
            monitor="val/total_loss",
            save_top_k=5,
            mode="min",
            save_last=True,
            every_n_train_steps=config.training.save_every
        ),
        LearningRateMonitor(logging_interval="epoch")
    ]
    
    # Create logger
    tb_logger = TensorBoardLogger(
        save_dir=log_dir,
        name="futurevox"
    )
    
    # Create trainer with explicit strategy
    trainer = pl.Trainer(
        max_epochs=config.training.epochs,
        callbacks=callbacks,
        logger=tb_logger,
        precision=config.training.precision,
        gradient_clip_val=config.training.clip_grad_norm,
        log_every_n_steps=100,
        val_check_interval=config.training.eval_every,  # Use eval_every from config
        default_root_dir=args.output_dir,
        accelerator="gpu"  # Use 'gpu' if CUDA is available
    )
    
    # Train model
    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()