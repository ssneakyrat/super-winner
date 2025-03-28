import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer-based models.
    """
    def __init__(self, d_model, max_seq_len=5000):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # Register buffer (persistent state)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, d_model]
        """
        return x + self.pe[:, :x.size(1), :]

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism for transformer-based models.
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # Linear projections and reshape for multi-head attention
        q = self.W_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        # Apply softmax and dropout
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention weights to values
        output = torch.matmul(attn, v)
        
        # Reshape and apply output projection
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(output)
        
        return output, attn

class TransformerEncoderLayer(nn.Module):
    """
    Single layer of transformer encoder.
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection and layer norm
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class EnhancedPhonemeEncoder(nn.Module):
    """
    Enhanced phoneme encoder with transformer architecture for singing voice synthesis.
    """
    def __init__(self, config, num_phonemes, num_notes=128):
        super().__init__()
        self.config = config
        
        # Model dimensions
        d_model = config.get('phoneme_encoder', {}).get('d_model', 256)
        num_layers = config.get('phoneme_encoder', {}).get('num_layers', 4)
        num_heads = config.get('phoneme_encoder', {}).get('num_heads', 8)
        d_ff = config.get('phoneme_encoder', {}).get('d_ff', 1024)
        dropout = config.get('phoneme_encoder', {}).get('dropout', 0.1)
        
        # Phoneme embedding
        self.phoneme_embedding = nn.Embedding(num_phonemes, d_model)
        
        # Musical note embedding
        self.note_embedding = nn.Embedding(num_notes, d_model)
        
        # Position encoding
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Transformer encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.projection = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, phone_indices, note_indices=None, phone_masks=None):
        """
        Forward pass through the enhanced phoneme encoder.
        
        Args:
            phone_indices: Tensor of shape [B, L] with phoneme indices
            note_indices: Tensor of shape [B, L] with note indices (optional)
            phone_masks: Tensor of shape [B, L] where True indicates padding
            
        Returns:
            encoded_phonemes: Tensor of shape [B, L, d_model]
            attention_weights: Dictionary of attention weights for visualization
        """
        # Get phoneme embeddings
        x = self.phoneme_embedding(phone_indices)
        
        # Add note embeddings if provided
        if note_indices is not None:
            note_emb = self.note_embedding(note_indices)
            x = x + note_emb
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Create attention mask from phone_masks
        if phone_masks is not None:
            # Convert to attention mask [B, 1, 1, L]
            attention_mask = (~phone_masks).unsqueeze(1).unsqueeze(2)
        else:
            attention_mask = None
        
        # Apply transformer encoder layers
        attention_weights = {}
        for i, layer in enumerate(self.layers):
            x = layer(x, attention_mask)
        
        # Apply final projection
        encoded_phonemes = self.projection(x)
        
        # Apply mask if provided
        if phone_masks is not None:
            encoded_phonemes = encoded_phonemes.masked_fill(phone_masks.unsqueeze(-1), 0.0)
            
        return encoded_phonemes, attention_weights
    
    def get_attention_visualization(self, attention_weights, phone_indices, batch_idx=0):
        """
        Create attention visualizations for TensorBoard.
        
        Args:
            attention_weights: Dictionary of attention weights
            phone_indices: Tensor of shape [B, L] with phoneme indices
            batch_idx: Index of the batch to visualize
            
        Returns:
            attn_vis: Dictionary of attention visualizations
        """
        # This is a placeholder for attention visualization
        # In actual implementation, this would generate heatmaps or other visualizations
        return None