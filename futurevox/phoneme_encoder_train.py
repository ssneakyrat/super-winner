import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  # Added missing import for F
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
    
    def contrastive_loss(self, embeddings, labels=None):
        """
        Simple contrastive loss to train encoder to produce discriminative embeddings.
        
        Args:
            embeddings: Tensor of shape [batch_size, seq_len, d_model]
            labels: Tensor of shape [batch_size, seq_len] (optional - for supervised contrastive)
            
        Returns:
            loss: Contrastive loss value
        """
        # Average across time dimension to get one embedding per sequence
        # [batch_size, seq_len, d_model] -> [batch_size, d_model]
        embeddings = embeddings.mean(dim=1)
        
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Compute similarity matrix
        similarity = torch.matmul(embeddings, embeddings.transpose(0, 1))
        
        # Mask out self-similarity
        mask = torch.eye(similarity.size(0), device=similarity.device)
        similarity = similarity * (1 - mask) - mask * 1e9
        
        # Compute positive and negative pairs
        # In self-supervised setting, we assume that different sequences are negative pairs
        temperature = 0.1
        loss = -torch.log(
            torch.exp(similarity / temperature) / 
            torch.sum(torch.exp(similarity / temperature), dim=1, keepdim=True)
        ).mean()
        
        return loss
    
    def cosine_similarity_loss(self, embeddings, target_embeddings):
        """
        Cosine similarity loss between predicted and target embeddings.
        
        Args:
            embeddings: Predicted embeddings [batch_size, seq_len, d_model]
            target_embeddings: Target embeddings [batch_size, seq_len, d_model]
            
        Returns:
            loss: Cosine similarity loss
        """
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=2)
        target_embeddings = F.normalize(target_embeddings, p=2, dim=2)
        
        # Compute cosine similarity
        similarity = torch.sum(embeddings * target_embeddings, dim=2)
        
        # Loss is negative similarity (we want to maximize similarity)
        loss = -similarity.mean()
        
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
                
                # Create target embeddings as a simple autoencoder objective
                # We'll try to make encoded_phonemes predict themselves
                # This is just a self-supervised training signal for demonstration
                target_embeddings = encoded_phonemes.detach()
                
                # Compute loss
                # loss = self.criterion(encoded_phonemes, target_embeddings)
                loss = self.contrastive_loss(encoded_phonemes)
                
                # Backward pass
                loss.backward()
                
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
        epoch_loss /= len(self.train_dataloader)
        self.train_losses.append(epoch_loss)
        
        return epoch_loss
    
    def validate(self, epoch):
        """Validate model."""
        self.phoneme_encoder.eval()
        val_loss = 0.0
        
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
                
                # Create target embeddings
                target_embeddings = encoded_phonemes.detach()
                
                # Compute loss
                # loss = self.criterion(encoded_phonemes, target_embeddings)
                loss = self.contrastive_loss(encoded_phonemes)
                
                # Update validation loss
                val_loss += loss.item()
        
        # Compute average validation loss
        val_loss /= len(self.val_dataloader)
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
    
    # Define optimizer
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
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size for training (overrides config)")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-6,
                        help="Weight decay (L2 regularization)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run model on")
    args = parser.parse_args()
    
    main(args)