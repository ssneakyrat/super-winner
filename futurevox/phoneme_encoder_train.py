import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import time

# Import your existing modules
from models.singer_modules.phoneme_encoder import EnhancedPhonemeEncoder
from data.singer_dataset import SingingVoxDataModule


def read_config(config_path):
    """Read configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


class PhonemeEncoderTrainer:
    """Trainer for phoneme encoder."""
    
    def __init__(
        self,
        phoneme_encoder,
        train_dataloader,
        val_dataloader,
        optimizer,
        criterion,
        device,
        log_dir,
        max_epochs=10
    ):
        self.phoneme_encoder = phoneme_encoder
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.log_dir = log_dir
        self.max_epochs = max_epochs
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize lists for tracking metrics
        self.train_losses = []
        self.val_losses = []
        self.gradients = []
        self.weights = []
        
        # Move model to device
        self.phoneme_encoder = self.phoneme_encoder.to(device)
    
    def stable_contrastive_loss(self, embeddings, temperature=0.5, labels=None):
        """
        Numerically stable contrastive loss implementation.
        
        Args:
            embeddings: Tensor of shape [batch_size, seq_len, d_model]
            temperature: Temperature parameter (higher is more stable but less strict)
            labels: Tensor of shape [batch_size, seq_len] (optional - for supervised contrastive)
            
        Returns:
            loss: Contrastive loss value
        """
        # Replace any NaN values with zeros (safety check)
        embeddings = torch.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Get batch size
        batch_size = embeddings.size(0)
        if batch_size <= 1:
            # Need at least 2 samples for contrastive loss
            # Return a dummy loss that's differentiable
            return torch.tensor(0.0, requires_grad=True, device=embeddings.device)
        
        # Average across time dimension to get one embedding per sequence
        # [batch_size, seq_len, d_model] -> [batch_size, d_model]
        pooled_embeddings = embeddings.mean(dim=1)
        
        # Normalize embeddings (add small epsilon to avoid division by zero)
        norm = torch.norm(pooled_embeddings, p=2, dim=1, keepdim=True)
        norm = torch.clamp(norm, min=1e-8)  # Avoid division by zero
        pooled_embeddings = pooled_embeddings / norm
        
        # Compute cosine similarity matrix: [batch_size, batch_size]
        similarity = torch.matmul(pooled_embeddings, pooled_embeddings.transpose(0, 1))
        
        # For numerical stability, we use the InfoNCE formulation
        # First, mask out the diagonal (self-similarity)
        mask = torch.eye(batch_size, device=similarity.device)
        
        # Mask the similarity matrix to exclude self-similarity
        masked_similarity = similarity * (1 - mask)
        
        # Apply temperature scaling to similarity scores
        logits = masked_similarity / temperature
        
        # Create labels: for each row i, the positive example is at column i
        # We'll use a dummy labels tensor where each sample is a separate class
        labels = torch.arange(batch_size, device=logits.device)
        
        # Calculate the log_prob using log_softmax
        log_prob = F.log_softmax(logits, dim=1)
        
        # The contrastive loss is the negative log-likelihood of correct predictions
        # Since we don't have explicit positive pairs, we'll use dummy zero targets
        # and return a mean loss across all examples
        loss = -log_prob.mean()
        
        return loss
    
    def simpler_loss(self, embeddings):
        """
        A simpler loss function that's less prone to numerical issues.
        Just tries to make embeddings unit length.
        
        Args:
            embeddings: Tensor of shape [batch_size, seq_len, d_model]
            
        Returns:
            loss: Loss value
        """
        # Average across time dimension
        pooled = embeddings.mean(dim=1)
        
        # Calculate L2 norm of each embedding
        norms = torch.norm(pooled, p=2, dim=1)
        
        # Loss: make all embeddings have norm=1
        # This encourages the model to produce normalized embeddings
        loss = ((norms - 1.0) ** 2).mean()
        
        return loss
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.phoneme_encoder.train()
        epoch_loss = 0.0
        
        # Store gradients for the first batch only to avoid memory issues
        store_gradients = (epoch == 1)
        
        with tqdm(self.train_dataloader, desc=f"Epoch {epoch}/{self.max_epochs}") as pbar:
            for batch_idx, batch in enumerate(pbar):
                # Prepare inputs
                phone_indices = batch['phone_indices'].to(self.device)
                phone_masks = batch['phone_masks'].to(self.device) if 'phone_masks' in batch else None
                note_indices = batch['note_indices'].to(self.device) if 'note_indices' in batch else None
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass through phoneme encoder
                encoded_phonemes, _ = self.phoneme_encoder(
                    phone_indices, note_indices, phone_masks
                )
                
                # Check for NaN values
                if torch.isnan(encoded_phonemes).any():
                    print("Warning: NaN values in encoded_phonemes")
                    # Skip this batch
                    continue
                
                # Compute loss - use the simpler loss for stability
                # loss = self.stable_contrastive_loss(encoded_phonemes)
                loss = self.simpler_loss(encoded_phonemes)
                
                # Check if loss is NaN
                if torch.isnan(loss):
                    print("Warning: NaN loss encountered, skipping batch")
                    continue
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping to prevent explosion
                torch.nn.utils.clip_grad_norm_(self.phoneme_encoder.parameters(), max_norm=1.0)
                
                # Store gradients for visualization
                if store_gradients and batch_idx == 0:
                    gradients = []
                    for name, param in self.phoneme_encoder.named_parameters():
                        if param.requires_grad and param.grad is not None:
                            gradients.append(param.grad.norm().item())
                    self.gradients.append(gradients)
                    
                    # Store weights for the first layer only
                    for name, param in self.phoneme_encoder.named_parameters():
                        if 'phoneme_embedding.weight' in name:
                            self.weights.append(param.data.clone().cpu())
                            break
                
                # Update weights
                self.optimizer.step()
                
                # Update epoch loss
                epoch_loss += loss.item()
                
                # Update progress bar
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # Compute average epoch loss
        if len(self.train_dataloader) > 0:
            epoch_loss /= len(self.train_dataloader)
        self.train_losses.append(epoch_loss)
        
        return epoch_loss
    
    def validate(self, epoch):
        """Validate model."""
        self.phoneme_encoder.eval()
        val_loss = 0.0
        num_valid_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validating"):
                # Prepare inputs
                phone_indices = batch['phone_indices'].to(self.device)
                phone_masks = batch['phone_masks'].to(self.device) if 'phone_masks' in batch else None
                note_indices = batch['note_indices'].to(self.device) if 'note_indices' in batch else None
                
                # Forward pass through phoneme encoder
                encoded_phonemes, _ = self.phoneme_encoder(
                    phone_indices, note_indices, phone_masks
                )
                
                # Check for NaN values
                if torch.isnan(encoded_phonemes).any():
                    print("Warning: NaN values in validation encoded_phonemes")
                    continue
                
                # Compute loss - use the simpler loss for stability
                # loss = self.stable_contrastive_loss(encoded_phonemes)
                loss = self.simpler_loss(encoded_phonemes)
                
                # Check if loss is NaN
                if torch.isnan(loss):
                    print("Warning: NaN validation loss encountered, skipping batch")
                    continue
                
                # Update validation loss
                val_loss += loss.item()
                num_valid_batches += 1
        
        # Compute average validation loss
        if num_valid_batches > 0:
            val_loss /= num_valid_batches
        self.val_losses.append(val_loss)
        
        return val_loss
    
    def train(self):
        """Train the model for multiple epochs."""
        start_time = time.time()
        
        # Track best validation loss
        best_val_loss = float('inf')
        best_epoch = 0
        
        for epoch in range(1, self.max_epochs + 1):
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss = self.validate(epoch)
            
            # Print results
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
            
            # Save model if it's the best so far
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                model_path = os.path.join(self.log_dir, "best_phoneme_encoder.pt")
                torch.save(self.phoneme_encoder.state_dict(), model_path)
                print(f"New best model saved to {model_path}")
            
            # Save checkpoint every epoch
            checkpoint_path = os.path.join(self.log_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.phoneme_encoder.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
        
        # Save final model
        model_path = os.path.join(self.log_dir, "final_phoneme_encoder.pt")
        torch.save(self.phoneme_encoder.state_dict(), model_path)
        
        # Plot training curves
        self.plot_training_curves()
        
        # Plot gradient norms
        self.plot_gradient_norms()
        
        # Plot embedding PCA
        self.plot_embedding_pca()
        
        # Report training time
        train_time = time.time() - start_time
        print(f"Training complete in {train_time/60:.2f} minutes")
        print(f"Best model at epoch {best_epoch} with validation loss {best_val_loss:.4f}")
    
    def plot_training_curves(self):
        """Plot training and validation loss curves."""
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.train_losses) + 1), self.train_losses, 'b', label='Training Loss')
        plt.plot(range(1, len(self.val_losses) + 1), self.val_losses, 'r', label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.log_dir, 'loss_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_gradient_norms(self):
        """Plot gradient norms during training."""
        if not self.gradients:
            return
        
        plt.figure(figsize=(10, 6))
        for i in range(len(self.gradients[0])):
            plt.plot(range(1, len(self.gradients) + 1), [g[i] for g in self.gradients], label=f'Layer {i+1}')
        plt.xlabel('Epoch')
        plt.ylabel('Gradient Norm')
        plt.title('Gradient Norms During Training')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.log_dir, 'gradient_norms.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_embedding_pca(self):
        """Plot PCA of embedding weights during training."""
        if not self.weights:
            return
        
        # Use PCA to reduce embeddings to 2D
        from sklearn.decomposition import PCA
        
        # Take a subset of embeddings for visualization
        num_embeddings = 100  # Display first 100 embedding vectors
        weights_subset = [w[:num_embeddings].numpy() for w in self.weights]
        
        # Create a figure with subplots for each epoch
        num_epochs = len(weights_subset)
        fig, axes = plt.subplots(1, num_epochs, figsize=(5*num_epochs, 5))
        if num_epochs == 1:
            axes = [axes]
        
        for i, (ax, weights) in enumerate(zip(axes, weights_subset)):
            # Apply PCA to weights
            pca = PCA(n_components=2)
            weights_2d = pca.fit_transform(weights)
            
            # Plot PCA of embeddings
            ax.scatter(weights_2d[:, 0], weights_2d[:, 1], alpha=0.7)
            ax.set_title(f'Epoch {i+1}')
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, 'embedding_pca.png'), dpi=300, bbox_inches='tight')
        plt.close()


def main(args):
    """Main function."""
    # Read configuration
    config = read_config(args.config)
    
    # Update batch size if specified in args
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    
    # Create data module
    data_module = SingingVoxDataModule(config, args.h5_path)
    
    # Setup data module
    data_module.prepare_data()
    data_module.setup()
    
    # Get number of phonemes
    num_phonemes = data_module.get_num_phonemes()
    print(f"Found {num_phonemes} unique phonemes in the dataset")
    
    # Initialize phoneme encoder
    print("Initializing phoneme encoder...")
    phoneme_encoder = EnhancedPhonemeEncoder(
        config,
        num_phonemes=num_phonemes,
        num_notes=128
    )
    
    # Print model information
    num_params = sum(p.numel() for p in phoneme_encoder.parameters())
    print(f"Phoneme Encoder has {num_params:,} parameters")
    
    # Define optimizer with smaller learning rate
    optimizer = optim.Adam(
        phoneme_encoder.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Define loss function
    criterion = nn.MSELoss()
    
    # Create trainer
    trainer = PhonemeEncoderTrainer(
        phoneme_encoder=phoneme_encoder,
        train_dataloader=data_module.train_dataloader(),
        val_dataloader=data_module.val_dataloader(),
        optimizer=optimizer,
        criterion=criterion,
        device=args.device,
        log_dir=args.log_dir,
        max_epochs=args.epochs
    )
    
    # Train model
    trainer.train()
    
    print("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Phoneme Encoder")
    parser.add_argument("--config", type=str, default="config/default.yaml",
                        help="Path to config file")
    parser.add_argument("--h5_path", type=str, required=True,
                        help="Path to HDF5 dataset file")
    parser.add_argument("--log_dir", type=str, default="logs/phoneme_encoder",
                        help="Log directory for checkpoints and visualizations")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training (overrides config)")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate (default: 1e-5, try a smaller value)")
    parser.add_argument("--weight_decay", type=float, default=1e-6,
                        help="Weight decay (L2 regularization)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run model on")
    args = parser.parse_args()
    
    main(args)