import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import h5py
from pathlib import Path
import time

# Import your modules
from models.singer_modules.phoneme_encoder import EnhancedPhonemeEncoder
from data.singer_dataset import SingingVoxDataset
from simplified_variance_adaptor import SimplifiedVarianceAdaptor


def read_config(config_path):
    """Read configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


class VarianceAdaptorTrainer:
    """Trainer for the variance adaptor."""
    
    def __init__(
        self,
        phoneme_encoder,
        variance_adaptor,
        train_dataloader,
        val_dataloader,
        optimizer,
        device,
        log_dir,
        max_epochs=50,
        patience=10,
        checkpoint_interval=5
    ):
        self.phoneme_encoder = phoneme_encoder
        self.variance_adaptor = variance_adaptor
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.device = device
        self.log_dir = log_dir
        self.max_epochs = max_epochs
        self.patience = patience
        self.checkpoint_interval = checkpoint_interval
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(os.path.join(log_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(log_dir, 'plots'), exist_ok=True)
        
        # Loss function
        self.duration_loss_fn = nn.MSELoss()
        self.f0_loss_fn = nn.L1Loss()
        self.energy_loss_fn = nn.L1Loss()
        
        # Initialize lists for tracking metrics
        self.train_losses = []
        self.val_losses = []
        self.early_stop_counter = 0
        self.best_val_loss = float('inf')
        
        # Move models to device
        self.phoneme_encoder = self.phoneme_encoder.to(device)
        self.variance_adaptor = self.variance_adaptor.to(device)
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.phoneme_encoder.eval()  # Freeze the phoneme encoder
        self.variance_adaptor.train()
        epoch_loss = 0.0
        duration_losses = []
        f0_losses = []
        energy_losses = []
        
        with tqdm(self.train_dataloader, desc=f"Epoch {epoch}/{self.max_epochs}") as pbar:
            for batch_idx, batch in enumerate(pbar):
                # Prepare inputs
                phone_indices = batch['phone_indices'].to(self.device)
                phone_masks = batch['phone_masks'].to(self.device) if 'phone_masks' in batch else None
                note_indices = batch['note_indices'].to(self.device) if 'note_indices' in batch else None
                
                # Get ground truth data
                # Handle tensors and lists differently
                if 'f0_values' in batch:
                    if isinstance(batch['f0_values'], list):
                        # Convert list of tensors to a single stacked tensor with padding
                        max_len = max(item.size(0) for item in batch['f0_values'])
                        f0_values = torch.zeros(len(batch['f0_values']), max_len)
                        for i, item in enumerate(batch['f0_values']):
                            f0_values[i, :item.size(0)] = item
                        f0_values = f0_values.to(self.device)
                    else:
                        f0_values = batch['f0_values'].to(self.device)
                else:
                    f0_values = None
                    
                if 'durations' in batch:
                    if isinstance(batch['durations'], list):
                        # Pad durations
                        max_len = max(d.size(0) for d in batch['durations'])
                        padded_durations = torch.zeros(len(batch['durations']), max_len)
                        for i, d in enumerate(batch['durations']):
                            padded_durations[i, :d.size(0)] = d
                        durations = padded_durations.to(self.device)
                    else:
                        durations = batch['durations'].to(self.device)
                else:
                    durations = None
                    
                if 'energy' in batch:
                    if isinstance(batch['energy'], list):
                        # Pad energy
                        max_len = max(item.size(0) for item in batch['energy'])
                        energy = torch.zeros(len(batch['energy']), max_len)
                        for i, item in enumerate(batch['energy']):
                            energy[i, :item.size(0)] = item
                        energy = energy.to(self.device)
                    else:
                        energy = batch['energy'].to(self.device)
                else:
                    energy = None
                
                # Create ground truth dict with careful checks
                ground_truth = {}
                
                if durations is not None:
                    # Ensure durations are properly formatted
                    if isinstance(durations, torch.Tensor):
                        # Add to ground truth
                        ground_truth['durations'] = durations
                    else:
                        print("Warning: durations is not a tensor, skipping")
                
                if f0_values is not None:
                    # Ensure f0_values are properly formatted
                    if isinstance(f0_values, torch.Tensor):
                        # Add to ground truth
                        ground_truth['f0_values'] = f0_values
                    else:
                        print("Warning: f0_values is not a tensor, skipping")
                
                if energy is not None:
                    # Ensure energy is properly formatted
                    if isinstance(energy, torch.Tensor):
                        # Add to ground truth
                        ground_truth['energy'] = energy
                    else:
                        print("Warning: energy is not a tensor, skipping")
                
                # Skip batch if missing essential data
                if 'durations' not in ground_truth:
                    print("Skipping batch due to missing durations")
                    continue
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass through phoneme encoder (no gradient)
                with torch.no_grad():
                    encoded_phonemes, _ = self.phoneme_encoder(phone_indices, None, phone_masks)
                
                # Create note embedding if needed
                note_embedding = None
                if note_indices is not None:
                    embedding_dim = encoded_phonemes.size(-1)
                    note_emb = nn.Embedding(128, embedding_dim).to(self.device)
                    note_embedding = note_emb(note_indices)
                
                # Forward pass through variance adaptor (with gradient)
                output_dict = self.variance_adaptor(
                    encoded_phonemes,
                    phone_masks=phone_masks,
                    note_pitch=note_embedding,
                    ground_truth=None  # Don't use ground truth for prediction
                )
                
                # Calculate losses
                loss = 0.0
                
                # Duration loss - compare log durations with log of ground truth
                if durations is not None and 'log_durations' in output_dict:
                    # Convert ground truth durations to log domain with small epsilon
                    log_gt_durations = torch.log(durations + 1e-5)
                    # Get predicted log durations
                    log_pred_durations = output_dict['log_durations']
                    
                    # Ensure dimensions match
                    min_len = min(log_gt_durations.size(1), log_pred_durations.size(1))
                    log_gt_durations = log_gt_durations[:, :min_len]
                    log_pred_durations = log_pred_durations[:, :min_len]
                    
                    # Apply mask if available, but ensure dimensions match
                    if phone_masks is not None:
                        # Ensure mask matches the truncated duration length
                        if phone_masks.size(1) > min_len:
                            valid_mask = ~phone_masks[:, :min_len]
                        elif phone_masks.size(1) < min_len:
                            # Pad mask if needed
                            padding = torch.ones(phone_masks.size(0), min_len - phone_masks.size(1), 
                                               dtype=phone_masks.dtype, device=phone_masks.device)
                            valid_mask = ~torch.cat([phone_masks, padding], dim=1)
                        else:
                            valid_mask = ~phone_masks
                            
                        # Apply mask with correct dimensions
                        if log_gt_durations.dim() == 3 and valid_mask.dim() == 2:
                            valid_mask = valid_mask.unsqueeze(-1)
                        
                        # Safe masking with dimension check
                        if log_gt_durations.size(1) == valid_mask.size(1):
                            log_gt_durations = log_gt_durations * valid_mask
                            log_pred_durations = log_pred_durations * valid_mask
                    
                    # Calculate loss
                    duration_loss = self.duration_loss_fn(log_pred_durations, log_gt_durations)
                    loss += duration_loss
                    duration_losses.append(duration_loss.item())
                
                # F0 loss - compare with ground truth
                if f0_values is not None and 'f0_contour' in output_dict:
                    # Get predicted F0 contour
                    f0_contour = output_dict['f0_contour']
                    # Calculate F0 loss on valid positions (ignoring zeros in ground truth)
                    # Create a mask for valid F0 values
                    valid_f0_mask = (f0_values > 0)
                    f0_loss = 0.0
                    # Only calculate loss if there are valid F0 values
                    if torch.any(valid_f0_mask):
                        # Get minimum valid length
                        valid_len = min(f0_values.size(1), f0_contour.size(1))
                        # Compute loss only on valid values
                        f0_loss = self.f0_loss_fn(
                            f0_contour[:, :valid_len] * valid_f0_mask[:, :valid_len],
                            f0_values[:, :valid_len] * valid_f0_mask[:, :valid_len]
                        )
                        loss += f0_loss
                        f0_losses.append(f0_loss.item())
                
                # Energy loss - compare with ground truth
                if energy is not None and 'energy' in output_dict:
                    # Get predicted energy
                    pred_energy = output_dict['energy']
                    # Calculate energy loss on valid positions
                    energy_loss = 0.0
                    # Get minimum valid length
                    valid_len = min(energy.size(1), pred_energy.size(1))
                    # Compute loss
                    energy_loss = self.energy_loss_fn(
                        pred_energy[:, :valid_len],
                        energy[:, :valid_len]
                    )
                    loss += energy_loss
                    energy_losses.append(energy_loss.item())
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.variance_adaptor.parameters(), max_norm=1.0)
                
                # Update weights
                self.optimizer.step()
                
                # Update epoch loss
                epoch_loss += loss.item()
                
                # Update progress bar
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "dur_loss": f"{duration_losses[-1] if duration_losses else 0:.4f}",
                    "f0_loss": f"{f0_losses[-1] if f0_losses else 0:.4f}"
                })
        
        # Compute average epoch loss
        if len(self.train_dataloader) > 0:
            epoch_loss /= len(self.train_dataloader)
        self.train_losses.append(epoch_loss)
        
        # Log component losses
        avg_duration_loss = np.mean(duration_losses) if duration_losses else 0
        avg_f0_loss = np.mean(f0_losses) if f0_losses else 0
        avg_energy_loss = np.mean(energy_losses) if energy_losses else 0
        
        print(f"Train Loss: {epoch_loss:.4f} | Duration: {avg_duration_loss:.4f} | F0: {avg_f0_loss:.4f} | Energy: {avg_energy_loss:.4f}")
        
        return epoch_loss
    
    def validate(self, epoch):
        """Validate model."""
        self.phoneme_encoder.eval()
        self.variance_adaptor.eval()
        val_loss = 0.0
        num_valid_batches = 0
        duration_losses = []
        f0_losses = []
        energy_losses = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_dataloader, desc="Validating")):
                # Prepare inputs
                phone_indices = batch['phone_indices'].to(self.device)
                phone_masks = batch['phone_masks'].to(self.device) if 'phone_masks' in batch else None
                note_indices = batch['note_indices'].to(self.device) if 'note_indices' in batch else None
                
                # Get ground truth data
                # Handle tensors and lists differently
                if 'f0_values' in batch:
                    if isinstance(batch['f0_values'], list):
                        # Convert list of tensors to a single stacked tensor with padding
                        max_len = max(item.size(0) for item in batch['f0_values'])
                        f0_values = torch.zeros(len(batch['f0_values']), max_len)
                        for i, item in enumerate(batch['f0_values']):
                            f0_values[i, :item.size(0)] = item
                        f0_values = f0_values.to(self.device)
                    else:
                        f0_values = batch['f0_values'].to(self.device)
                else:
                    f0_values = None
                    
                if 'durations' in batch:
                    if isinstance(batch['durations'], list):
                        # Pad durations
                        max_len = max(d.size(0) for d in batch['durations'])
                        padded_durations = torch.zeros(len(batch['durations']), max_len)
                        for i, d in enumerate(batch['durations']):
                            padded_durations[i, :d.size(0)] = d
                        durations = padded_durations.to(self.device)
                    else:
                        durations = batch['durations'].to(self.device)
                else:
                    durations = None
                    
                if 'energy' in batch:
                    if isinstance(batch['energy'], list):
                        # Pad energy
                        max_len = max(item.size(0) for item in batch['energy'])
                        energy = torch.zeros(len(batch['energy']), max_len)
                        for i, item in enumerate(batch['energy']):
                            energy[i, :item.size(0)] = item
                        energy = energy.to(self.device)
                    else:
                        energy = batch['energy'].to(self.device)
                else:
                    energy = None
                
                # Skip batch if missing essential data
                if durations is None:
                    continue
                
                # Forward pass through phoneme encoder
                encoded_phonemes, _ = self.phoneme_encoder(phone_indices, None, phone_masks)
                
                # Create note embedding if needed
                note_embedding = None
                if note_indices is not None:
                    embedding_dim = encoded_phonemes.size(-1)
                    note_emb = nn.Embedding(128, embedding_dim).to(self.device)
                    note_embedding = note_emb(note_indices)
                
                # Forward pass through variance adaptor
                output_dict = self.variance_adaptor(
                    encoded_phonemes,
                    phone_masks=phone_masks,
                    note_pitch=note_embedding,
                    ground_truth=None  # Don't use ground truth for prediction
                )
                
                # Calculate losses
                loss = 0.0
                
                # Duration loss - compare log durations with log of ground truth
                if durations is not None and 'log_durations' in output_dict:
                    # Convert ground truth durations to log domain with small epsilon
                    log_gt_durations = torch.log(durations + 1e-5)
                    # Get predicted log durations
                    log_pred_durations = output_dict['log_durations']
                    
                    # Ensure dimensions match
                    min_len = min(log_gt_durations.size(1), log_pred_durations.size(1))
                    log_gt_durations = log_gt_durations[:, :min_len]
                    log_pred_durations = log_pred_durations[:, :min_len]
                    
                    # Apply mask if available, but ensure dimensions match
                    if phone_masks is not None:
                        # Ensure mask matches the truncated duration length
                        if phone_masks.size(1) > min_len:
                            valid_mask = ~phone_masks[:, :min_len]
                        elif phone_masks.size(1) < min_len:
                            # Pad mask if needed
                            padding = torch.ones(phone_masks.size(0), min_len - phone_masks.size(1), 
                                               dtype=phone_masks.dtype, device=phone_masks.device)
                            valid_mask = ~torch.cat([phone_masks, padding], dim=1)
                        else:
                            valid_mask = ~phone_masks
                            
                        # Apply mask with correct dimensions
                        if log_gt_durations.dim() == 3 and valid_mask.dim() == 2:
                            valid_mask = valid_mask.unsqueeze(-1)
                        
                        # Safe masking with dimension check
                        if log_gt_durations.size(1) == valid_mask.size(1):
                            log_gt_durations = log_gt_durations * valid_mask
                            log_pred_durations = log_pred_durations * valid_mask
                    
                    # Calculate loss
                    duration_loss = self.duration_loss_fn(log_pred_durations, log_gt_durations)
                    loss += duration_loss
                    duration_losses.append(duration_loss.item())
                
                # F0 loss - compare with ground truth
                if f0_values is not None and 'f0_contour' in output_dict:
                    # Get predicted F0 contour
                    f0_contour = output_dict['f0_contour']
                    # Calculate F0 loss on valid positions (ignoring zeros in ground truth)
                    # Create a mask for valid F0 values
                    valid_f0_mask = (f0_values > 0)
                    f0_loss = 0.0
                    # Only calculate loss if there are valid F0 values
                    if torch.any(valid_f0_mask):
                        # Get minimum valid length
                        valid_len = min(f0_values.size(1), f0_contour.size(1))
                        # Compute loss only on valid values
                        f0_loss = self.f0_loss_fn(
                            f0_contour[:, :valid_len] * valid_f0_mask[:, :valid_len],
                            f0_values[:, :valid_len] * valid_f0_mask[:, :valid_len]
                        )
                        loss += f0_loss
                        f0_losses.append(f0_loss.item())
                
                # Energy loss - compare with ground truth
                if energy is not None and 'energy' in output_dict:
                    # Get predicted energy
                    pred_energy = output_dict['energy']
                    # Calculate energy loss on valid positions
                    energy_loss = 0.0
                    # Get minimum valid length
                    valid_len = min(energy.size(1), pred_energy.size(1))
                    # Compute loss
                    energy_loss = self.energy_loss_fn(
                        pred_energy[:, :valid_len],
                        energy[:, :valid_len]
                    )
                    loss += energy_loss
                    energy_losses.append(energy_loss.item())
                
                # Update validation loss
                val_loss += loss.item()
                num_valid_batches += 1
        
        # Compute average validation loss
        if num_valid_batches > 0:
            val_loss /= num_valid_batches
        self.val_losses.append(val_loss)
        
        # Log component losses
        avg_duration_loss = np.mean(duration_losses) if duration_losses else 0
        avg_f0_loss = np.mean(f0_losses) if f0_losses else 0
        avg_energy_loss = np.mean(energy_losses) if energy_losses else 0
        
        print(f"Val Loss: {val_loss:.4f} | Duration: {avg_duration_loss:.4f} | F0: {avg_f0_loss:.4f} | Energy: {avg_energy_loss:.4f}")
        
        # Early stopping logic
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.early_stop_counter = 0
            # Save best model
            self.save_checkpoint('best')
            return True
        else:
            self.early_stop_counter += 1
            return False
    
    def train(self):
        """Train the model for multiple epochs."""
        start_time = time.time()
        
        print(f"Starting training for {self.max_epochs} epochs...")
        for epoch in range(1, self.max_epochs + 1):
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            improved = self.validate(epoch)
            
            # Save checkpoint
            if epoch % self.checkpoint_interval == 0:
                self.save_checkpoint(f'epoch_{epoch}')
            
            # Plot loss curves
            self.plot_loss_curves()
            
            # Early stopping check
            if self.early_stop_counter >= self.patience:
                print(f"Early stopping triggered after {epoch} epochs")
                break
        
        # Save final model
        self.save_checkpoint('final')
        
        # Report training time
        train_time = time.time() - start_time
        print(f"Training complete in {train_time/60:.2f} minutes")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
    
    def save_checkpoint(self, name):
        """Save model checkpoint."""
        checkpoint_path = os.path.join(self.log_dir, 'checkpoints', f'variance_adaptor_{name}.pt')
        torch.save(self.variance_adaptor.state_dict(), checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
    
    def plot_loss_curves(self):
        """Plot training and validation loss curves."""
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.train_losses) + 1), self.train_losses, 'b-', label='Training Loss')
        plt.plot(range(1, len(self.val_losses) + 1), self.val_losses, 'r-', label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.log_dir, 'plots', 'loss_curves.png'), dpi=300)
        plt.close()


def main(args):
    """Main function."""
    # Read configuration
    config = read_config(args.config)
    
    # Update config with command line arguments
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    
    # Load dataset
    print(f"Loading dataset from {args.h5_path}...")
    train_dataset = SingingVoxDataset(args.h5_path, config, split='train')
    val_dataset = SingingVoxDataset(args.h5_path, config, split='val')
    
    # Create dataloaders
    train_dataloader = train_dataset.get_dataloader(
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        shuffle=True
    )
    
    val_dataloader = val_dataset.get_dataloader(
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        shuffle=False
    )
    
    # Get number of phonemes
    with h5py.File(args.h5_path, 'r') as f:
        # Get file list
        file_list = f['metadata']['file_list'][:]
        file_ids = [name.decode('utf-8') for name in file_list]
        
        # Collect all phonemes
        all_phonemes = set()
        for sample_id in file_ids:
            if sample_id in f:
                sample = f[sample_id]
                if 'phonemes' in sample and 'phones' in sample['phonemes']:
                    phones_bytes = sample['phonemes']['phones'][:]
                    phones = [p.decode('utf-8') for p in phones_bytes]
                    all_phonemes.update(phones)
        
        num_phonemes = len(all_phonemes)
        print(f"Found {num_phonemes} unique phonemes in the dataset")
    
    # Initialize phoneme encoder
    print("Initializing phoneme encoder...")
    phoneme_encoder = EnhancedPhonemeEncoder(
        config,
        num_phonemes=num_phonemes,
        num_notes=128
    )
    
    # Load phoneme encoder checkpoint if provided
    device = torch.device(args.device)
    
    if args.phoneme_encoder_checkpoint:
        print(f"Loading phoneme encoder checkpoint from {args.phoneme_encoder_checkpoint}")
        checkpoint = torch.load(args.phoneme_encoder_checkpoint, map_location=device)
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            # Remove 'model.' prefix if present
            if all(k.startswith('model.') for k in state_dict.keys()):
                state_dict = {k[6:]: v for k, v in state_dict.items()}
            # Remove 'phoneme_encoder.' prefix if present
            elif all(k.startswith('phoneme_encoder.') for k in state_dict.keys()):
                state_dict = {k[16:]: v for k, v in state_dict.items()}
            phoneme_encoder.load_state_dict(state_dict)
        else:
            phoneme_encoder.load_state_dict(checkpoint)
    
    # Initialize variance adaptor
    print("Initializing variance adaptor...")
    input_dim = config.get('model', {}).get('phoneme_encoder', {}).get('d_model', 256)
    variance_adaptor = SimplifiedVarianceAdaptor(input_dim=input_dim)
    
    # Load variance adaptor checkpoint if provided (for resuming training)
    if args.variance_adaptor_checkpoint:
        print(f"Loading variance adaptor checkpoint from {args.variance_adaptor_checkpoint}")
        checkpoint = torch.load(args.variance_adaptor_checkpoint, map_location=device)
        variance_adaptor.load_state_dict(checkpoint)
    
    # Configure optimizer
    optimizer = optim.Adam(
        variance_adaptor.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Create trainer
    trainer = VarianceAdaptorTrainer(
        phoneme_encoder=phoneme_encoder,
        variance_adaptor=variance_adaptor,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        device=device,
        log_dir=args.log_dir,
        max_epochs=args.epochs,
        patience=args.patience,
        checkpoint_interval=args.checkpoint_interval
    )
    
    # Train model
    trainer.train()
    
    print("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Variance Adaptor")
    parser.add_argument("--config", type=str, default="config/default.yaml",
                        help="Path to config file")
    parser.add_argument("--h5_path", type=str, required=True,
                        help="Path to HDF5 dataset file")
    parser.add_argument("--phoneme_encoder_checkpoint", type=str, default=None,
                        help="Path to phoneme encoder checkpoint")
    parser.add_argument("--variance_adaptor_checkpoint", type=str, default=None,
                        help="Path to variance adaptor checkpoint for resuming training")
    parser.add_argument("--log_dir", type=str, default="logs/variance_adaptor",
                        help="Log directory for checkpoints and visualizations")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size for training (overrides config)")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate for optimizer")
    parser.add_argument("--weight_decay", type=float, default=1e-6,
                        help="Weight decay for optimizer")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience (epochs)")
    parser.add_argument("--checkpoint_interval", type=int, default=5,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run model on")
    args = parser.parse_args()
    
    main(args)