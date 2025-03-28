import os
import argparse
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
import time
import shutil
from pathlib import Path

from data.singer_dataset import SingingVoxDataModule
from models.lightning_singer_module import FutureVoxSingerLightningModule

def main(args):
    """Main training function."""
    # Read configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update config with command line arguments
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.learning_rate is not None:
        config['training']['learning_rate'] = args.learning_rate
    if args.max_epochs is not None:
        config['training']['max_epochs'] = args.max_epochs
    
    # Create checkpoint directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = os.path.join(config['training']['checkpoint_dir'], f"run_{timestamp}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save a copy of the config file
    config_copy_path = os.path.join(checkpoint_dir, "config.yaml")
    shutil.copy(args.config, config_copy_path)
    
    # Set up logger - Modified to store logs directly in the checkpoint directory
    logger = TensorBoardLogger(
        save_dir=os.path.join( 'checkpoints/logs'),
        name=''  # Empty name to avoid creating a subdirectory
    )
    
    # Setup data module
    data_module = SingingVoxDataModule(config, args.data_path)
    
    # Prepare data to get phoneme and singer counts
    data_module.prepare_data()
    data_module.setup()
    
    # Get number of phonemes and singers
    num_phonemes = data_module.get_num_phonemes()
    num_singers = data_module.get_num_singers()
    
    print(f"Dataset contains {num_phonemes} unique phonemes and {num_singers} singers")
    
    # Update config with these values
    config['model']['phoneme_encoder']['num_phonemes'] = num_phonemes
    config['model']['variance_adaptor']['num_singers'] = num_singers
    
    # Create model
    model = FutureVoxSingerLightningModule(config, num_phonemes=num_phonemes)
    
    # Create callbacks
    callbacks = [
        # Checkpoint callback
        ModelCheckpoint(
            dirpath=os.path.join(checkpoint_dir, 'checkpoints'),
            filename='{epoch:02d}-{val/total_loss:.4f}',
            save_top_k=3,
            monitor='val/total_loss',
            mode='min',
            save_last=True,
            verbose=True
        ),
        # Learning rate monitor
        LearningRateMonitor(logging_interval='step'),
        # Early stopping
        EarlyStopping(
            monitor='val/total_loss',
            patience=args.patience,
            mode='min',
            verbose=True
        )
    ]
    
    # Create trainer - REMOVED gradient_clip_val to fix the error
    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        logger=logger,
        callbacks=callbacks,
        precision='16-mixed' if args.mixed_precision else '32',
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=args.gpus if args.gpus else 1,
        # gradient_clip_val parameter removed - manual clip in the model instead
        deterministic=args.deterministic,
        log_every_n_steps=10,
        val_check_interval=0.25,  # Validate 4 times per epoch
        accumulate_grad_batches=args.grad_accum,
    )
    
    # Print model size information
    model_size = sum(p.numel() for p in model.parameters()) / 1_000_000
    print(f"Model size: {model_size:.2f}M parameters")
    
    # Estimate memory usage
    batch_size = config['training']['batch_size']
    if torch.cuda.is_available():
        # Rough memory estimation (varies depending on model architecture)
        estimated_memory = model_size * 4 * 3 * batch_size / 1024  # GB
        print(f"Estimated GPU memory usage: {estimated_memory:.2f} GB")
        
        # Print actual GPU info
        device = torch.cuda.current_device()
        gpu_properties = torch.cuda.get_device_properties(device)
        print(f"Training on: {gpu_properties.name}")
        print(f"GPU memory: {gpu_properties.total_memory / 1024 / 1024 / 1024:.2f} GB")
    
    # Print training information
    print(f"Training with batch size: {batch_size}")
    print(f"Max epochs: {config['training']['max_epochs']}")
    print(f"Progressive training: {config['training']['progressive']}")
    
    # Train model
    print("Starting training...")
    trainer.fit(
        model=model,
        datamodule=data_module,
        ckpt_path=args.resume_from  # Add the checkpoint path here
    )
    
    # Get best model path
    best_model_path = trainer.checkpoint_callback.best_model_path
    print(f"Best model saved at: {best_model_path}")
    
    # Save best model path to a file for easy reference
    with open(os.path.join(checkpoint_dir, "best_model.txt"), "w") as f:
        f.write(f"Best model path: {best_model_path}\n")
        f.write(f"Best validation loss: {trainer.checkpoint_callback.best_model_score.item()}\n")
    
    return best_model_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train FutureVox-Singer model')
    parser.add_argument('--config', type=str, default='config/default.yaml',
                        help='Path to configuration file')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to HDF5 dataset file (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (overrides config)')
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='Learning rate (overrides config)')
    parser.add_argument('--max_epochs', type=int, default=None,
                        help='Maximum epochs (overrides config)')
    parser.add_argument('--gpus', type=int, default=None,
                        help='Number of GPUs to use')
    parser.add_argument('--mixed_precision', action='store_true',
                        help='Use mixed precision training')
    parser.add_argument('--deterministic', action='store_true',
                        help='Use deterministic algorithms for reproducibility')
    parser.add_argument('--grad_accum', type=int, default=1,
                        help='Gradient accumulation steps')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Resume training from checkpoint')
    args = parser.parse_args()
    
    # Print PyTorch and CUDA information
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    
    # Run training
    best_model_path = main(args)