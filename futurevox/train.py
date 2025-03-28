import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pathlib import Path

from data.dataset import FutureVoxDataset
from models.lightning_module import FutureVoxLightningModule
from utils.callbacks import AlignmentVisualizationCallback, ModelCheckpointCallback, LoggingCallback
from utils.utils import read_config, create_checkpoint_dir


def main(args):
    """Main training function."""
    # Read configuration
    config = read_config(args.config)
    
    # Create checkpoint directory
    checkpoint_dir = create_checkpoint_dir(config)
    
    # Set up logger
    logger = TensorBoardLogger(
        save_dir=os.path.join(checkpoint_dir, 'logs'),
        name='futurevox'
    )
    
    # Load datasets
    data_raw_path = config['datasets']['data_raw']
    h5_path = os.path.join(data_raw_path, "binary", "gin.h5")
    
    if not os.path.exists(h5_path):
        raise FileNotFoundError(
            f"Dataset file not found at {h5_path}. "
            f"Run preprocess.py first to create the dataset."
        )
    
    # Create datasets
    train_dataset = FutureVoxDataset(h5_path, config, split='train')
    val_dataset = FutureVoxDataset(h5_path, config, split='val')
    
    # Create data loaders
    train_loader = train_dataset.get_dataloader(
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        shuffle=True
    )
    
    val_loader = val_dataset.get_dataloader(
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        shuffle=False
    )
    
    # Create model
    model = FutureVoxLightningModule(config)
    
    # Create callbacks
    callbacks = [
        AlignmentVisualizationCallback(h5_path, config),
        ModelCheckpointCallback(config),
        LoggingCallback(),
    ]
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        logger=logger,
        callbacks=callbacks,
        precision='16-mixed',  # Use mixed precision
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        deterministic=True,  # For reproducibility
        log_every_n_steps=10,
        val_check_interval=0.25,  # Validate 4 times per epoch
    )
    
    # Train model
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train FutureVox model')
    parser.add_argument('--config', type=str, default='config/default.yaml',
                        help='Path to configuration file')
    args = parser.parse_args()
    
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    main(args)