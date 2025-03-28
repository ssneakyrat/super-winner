import torch
import torch.nn as nn
import torch.nn.functional as F


class FutureVoxEncoder(nn.Module):
    """Encoder module for FutureVox."""
    
    def __init__(self, config):
        super().__init__()
        hidden_dim = config['training']['hidden_dim']
        
        # Define encoder layers
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(config['audio']['n_mels'], hidden_dim, kernel_size=5, padding=2),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
        ])
        
        self.norm_layers = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        ])
        
        # Final projection
        self.projection = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        
    def forward(self, mel_spectrograms, mel_masks=None):
        """
        Forward pass through the encoder.
        
        Args:
            mel_spectrograms: Tensor of shape [B, n_mels, T]
            mel_masks: Tensor of shape [B, T] where True indicates padding
            
        Returns:
            encoded: Tensor of shape [B, hidden_dim, T]
        """
        x = mel_spectrograms
        
        # Apply convolutional layers
        for conv, norm in zip(self.conv_layers, self.norm_layers):
            x = F.relu(norm(conv(x)))
        
        # Final projection
        encoded = self.projection(x)
        
        # Apply mask if provided
        if mel_masks is not None:
            encoded = encoded.masked_fill(mel_masks.unsqueeze(1), 0.0)
            
        return encoded


class FutureVoxPhonemeEncoder(nn.Module):
    """Encodes phoneme information."""
    
    def __init__(self, config, num_phonemes=100):  # Default value, adjust based on your phoneme inventory
        super().__init__()
        hidden_dim = config['training']['hidden_dim']
        
        # Phoneme embedding
        self.phoneme_embedding = nn.Embedding(num_phonemes, hidden_dim)
        
        # Projection
        self.projection = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, phone_indices, phone_masks=None):
        """
        Forward pass through the phoneme encoder.
        
        Args:
            phone_indices: Tensor of shape [B, L] with phoneme indices
            phone_masks: Tensor of shape [B, L] where True indicates padding
            
        Returns:
            encoded_phonemes: Tensor of shape [B, L, hidden_dim]
        """
        # Get phoneme embeddings
        phoneme_embeddings = self.phoneme_embedding(phone_indices)
        
        # Apply projection
        encoded_phonemes = F.relu(self.projection(phoneme_embeddings))
        
        # Apply mask if provided
        if phone_masks is not None:
            encoded_phonemes = encoded_phonemes.masked_fill(phone_masks.unsqueeze(-1), 0.0)
            
        return encoded_phonemes


class FutureVoxModel(nn.Module):
    """Main FutureVox model."""
    
    def __init__(self, config, num_phonemes=100):
        super().__init__()
        self.config = config
        hidden_dim = config['training']['hidden_dim']
        
        # Encoders
        self.encoder = FutureVoxEncoder(config)
        self.phoneme_encoder = FutureVoxPhonemeEncoder(config, num_phonemes)
        
        # F0 predictor (pitch prediction)
        self.f0_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Duration predictor
        self.duration_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, mel_spectrograms, phone_indices, frames=None, mel_masks=None, phone_masks=None):
        """
        Forward pass through the model.
        
        Args:
            mel_spectrograms: Tensor of shape [B, n_mels, T]
            phone_indices: Tensor of shape [B, L] with phoneme indices
            frames: Tensor of shape [B, L, 2] with start and end frames for each phoneme
            mel_masks: Tensor of shape [B, T] where True indicates padding
            phone_masks: Tensor of shape [B, L] where True indicates padding
            
        Returns:
            output_dict: Dictionary containing model outputs
        """
        # Encode mel spectrograms
        encoded_mel = self.encoder(mel_spectrograms, mel_masks)
        
        # Encode phonemes
        encoded_phonemes = self.phoneme_encoder(phone_indices, phone_masks)
        
        # For each phoneme, extract the corresponding acoustic features
        # In a real implementation, you would align phonemes with acoustic features
        # For now, we'll just use a dummy approach
        
        # Predict F0 (fundamental frequency)
        f0_predictions = self.f0_predictor(encoded_phonemes).squeeze(-1)
        
        # Predict durations
        duration_predictions = self.duration_predictor(encoded_phonemes).squeeze(-1)
        duration_predictions = F.softplus(duration_predictions)  # Ensure positive durations
        
        # Create output dictionary
        output_dict = {
            'encoded_mel': encoded_mel,
            'encoded_phonemes': encoded_phonemes,
            'f0_predictions': f0_predictions,
            'duration_predictions': duration_predictions
        }
        
        return output_dict