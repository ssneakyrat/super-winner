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
    # Parse command line arguments - keeping only the config path
    parser = argparse.ArgumentParser(description="Train FutureVox model")
    parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to config file")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Get training parameters from config
    training_config = config.get('training', {})
    max_epochs = training_config.get('max_epochs', 100)
    batch_size = training_config.get('batch_size', 16)
    num_workers = training_config.get('num_workers', 4)
    learning_rate = training_config.get('learning_rate', 1e-4)
    hidden_dim = training_config.get('hidden_dim', 256)
    checkpoint_dir = training_config.get('checkpoint_dir', 'checkpoints')
    
    # Get audio configuration
    audio_config = config.get('audio', {})
    sample_rate = audio_config.get('sample_rate', 22050)
    hop_length = audio_config.get('hop_length', 256)
    n_mels = audio_config.get('n_mels', 80)
    
    # Get data directory from config
    data_dir = config['datasets']['data_raw']
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Set up loggers and callbacks
    logger = TensorBoardLogger(save_dir="logs", name="futurevox")
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
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
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    # Set up the datamodule to get the h5_file_path
    data_module.setup()
    
    # Create model with the h5_file_path for visualization
    model = FutureVoxBaseModel(
        n_mels=n_mels,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        h5_file_path=data_module.h5_file_path,
        hop_length=hop_length,
        sample_rate=sample_rate
    )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        log_every_n_steps=10,
    )
    
    # Train model
    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()