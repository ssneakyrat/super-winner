import os
import argparse
import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from data.datamodule import FutureVoxDataModule
from models.base_model import FutureVoxBaseModel


def load_config(config_path="config/default.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train FutureVox model")
    parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to config file")
    parser.add_argument("--data_dir", type=str, help="Path to data directory (overrides config)")
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension size")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override config with command line arguments if provided
    data_dir = args.data_dir if args.data_dir else config['datasets']['data_raw']
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Set up loggers and callbacks
    logger = TensorBoardLogger(save_dir="logs", name="futurevox")
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename="futurevox-{epoch:02d}-{val_loss:.4f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min"
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="min"
    )
    
    # Create data module
    data_module = FutureVoxDataModule(
        data_dir=data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Create model
    model = FutureVoxBaseModel(
        n_mels=config['audio']['n_mels'],
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate
    )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        log_every_n_steps=10,
    )
    
    # Train model
    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()