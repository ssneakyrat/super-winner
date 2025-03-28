import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FeedForwardTransformerLayer(nn.Module):
    """
    Feed-forward Transformer decoder layer for the acoustic decoder.
    Similar to a standard transformer but with additional conditioning.
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout)
        
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
        
    def forward(self, x, mask=None, conditioning=None):
        """
        Forward pass through the FFT layer.
        
        Args:
            x: Input tensor [seq_len, batch_size, d_model]
            mask: Attention mask [batch_size, seq_len, seq_len]
            conditioning: Optional conditioning tensor [seq_len, batch_size, d_model]
            
        Returns:
            output: Processed tensor [seq_len, batch_size, d_model]
        """
        # Apply self-attention
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = residual + self.dropout(x)
        
        # Add conditioning if provided
        if conditioning is not None:
            x = x + conditioning
        
        # Apply feed-forward network
        residual = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = residual + self.dropout(x)
        
        return x

class ReferenceEncoder(nn.Module):
    """
    Reference encoder for style transfer.
    Encodes a reference audio into a fixed-length style embedding.
    """
    def __init__(self, in_channels, ref_enc_filters, kernel_size, 
                 strides, ref_enc_gru_size, ref_attention=True):
        super().__init__()
        
        # 2D convolutional layers
        filters = [in_channels] + ref_enc_filters
        self.convs = nn.ModuleList()
        
        for i in range(len(ref_enc_filters)):
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(filters[i], filters[i+1], kernel_size, 
                             stride=strides, padding=1),
                    nn.BatchNorm2d(filters[i+1]),
                    nn.ReLU()
                )
            )
        
        # GRU layer
        self.gru = nn.GRU(
            input_size=ref_enc_filters[-1],
            hidden_size=ref_enc_gru_size,
            batch_first=True
        )
        
        # Attention layer if enabled
        self.ref_attention = ref_attention
        if ref_attention:
            self.attention = nn.Linear(ref_enc_gru_size, 1)
        
        # Output size
        self.output_size = ref_enc_gru_size
        
    def forward(self, inputs):
        """
        Encode reference audio to style embedding.
        
        Args:
            inputs: Reference mel spectrogram [batch_size, n_mels, time]
            
        Returns:
            style_embedding: Style embedding vector [batch_size, output_size]
        """
        # Add channel dimension: [batch_size, 1, n_mels, time]
        x = inputs.unsqueeze(1)
        
        # Apply convolutional layers
        for conv in self.convs:
            x = conv(x)
        
        # Get final feature map dimensions
        batch_size, channels, height, width = x.size()
        
        # Reshape for GRU: [batch_size, width, height*channels]
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(batch_size, width, channels * height)
        
        # Apply GRU
        self.gru.flatten_parameters()
        output, hidden = self.gru(x)
        
        # Apply attention if enabled, otherwise use last hidden state
        if self.ref_attention:
            # Compute attention weights
            attention_weights = F.softmax(self.attention(output).squeeze(-1), dim=1)
            # Apply attention
            style_embedding = torch.bmm(attention_weights.unsqueeze(1), output).squeeze(1)
        else:
            style_embedding = hidden.squeeze(0)
        
        return style_embedding

class ConvUpsampling(nn.Module):
    """
    Progressive upsampling with 1D convolutions.
    """
    def __init__(self, d_model, upsample_scales, conv_kernel_sizes, dropout=0.1):
        super().__init__()
        
        self.upsample_layers = nn.ModuleList()
        
        for i, (scale, kernel_size) in enumerate(zip(upsample_scales, conv_kernel_sizes)):
            # Create upsampling layers
            self.upsample_layers.append(
                nn.Sequential(
                    nn.ConvTranspose1d(
                        d_model if i == 0 else d_model // (2 ** i),
                        d_model // (2 ** (i + 1)) if i < len(upsample_scales) - 1 else d_model // (2 ** i),
                        kernel_size=2 * scale,
                        stride=scale,
                        padding=scale // 2 + scale % 2,
                        output_padding=scale % 2
                    ),
                    nn.ReLU(),
                    nn.BatchNorm1d(d_model // (2 ** (i + 1)) if i < len(upsample_scales) - 1 else d_model // (2 ** i)),
                    nn.Dropout(dropout)
                )
            )
    
    def forward(self, x):
        """
        Forward pass through upsampling layers.
        
        Args:
            x: Input tensor [batch_size, d_model, time]
            
        Returns:
            output: Upsampled tensor [batch_size, d_model//2^n, time*scale1*scale2*...]
        """
        for layer in self.upsample_layers:
            x = layer(x)
        return x

class AcousticDecoder(nn.Module):
    """
    Acoustic decoder for generating mel spectrograms.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Model dimensions
        d_model = config.get('acoustic_decoder', {}).get('d_model', 256)
        num_layers = config.get('acoustic_decoder', {}).get('num_layers', 6)
        num_heads = config.get('acoustic_decoder', {}).get('num_heads', 8)
        d_ff = config.get('acoustic_decoder', {}).get('d_ff', 1024)
        dropout = config.get('acoustic_decoder', {}).get('dropout', 0.1)
        
        # Reference encoder parameters
        use_ref_encoder = config.get('acoustic_decoder', {}).get('use_ref_encoder', True)
        n_mels = config['audio']['n_mels']
        ref_enc_filters = config.get('acoustic_decoder', {}).get('ref_enc_filters', [32, 32, 64, 64, 128, 128])
        ref_enc_kernel_size = config.get('acoustic_decoder', {}).get('ref_enc_kernel_size', 3)
        ref_enc_strides = config.get('acoustic_decoder', {}).get('ref_enc_strides', 2)
        ref_enc_gru_size = config.get('acoustic_decoder', {}).get('ref_enc_gru_size', 128)
        
        # Upsampling parameters
        upsample_scales = config.get('acoustic_decoder', {}).get('upsample_scales', [2, 2, 2, 2])
        conv_kernel_sizes = config.get('acoustic_decoder', {}).get('conv_kernel_sizes', [8, 8, 4, 4])
        
        # Create reference encoder if enabled
        self.use_ref_encoder = use_ref_encoder
        if use_ref_encoder:
            self.reference_encoder = ReferenceEncoder(
                in_channels=1,  # Single channel input
                ref_enc_filters=ref_enc_filters,
                kernel_size=ref_enc_kernel_size,
                strides=ref_enc_strides,
                ref_enc_gru_size=ref_enc_gru_size
            )
            
            # Project reference embedding to model dimension
            self.ref_projection = nn.Linear(ref_enc_gru_size, d_model)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            FeedForwardTransformerLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Progressive upsampling
        self.upsampling = ConvUpsampling(
            d_model=d_model,
            upsample_scales=upsample_scales,
            conv_kernel_sizes=conv_kernel_sizes,
            dropout=dropout
        )
        
        # Final mel spectrogram projection
        self.mel_projection = nn.Conv1d(
            d_model // (2 ** len(upsample_scales)) if len(upsample_scales) > 0 else d_model,
            n_mels,
            kernel_size=1
        )
        
        # Post-net (improves reconstruction quality)
        self.postnet = nn.Sequential(
            nn.Conv1d(n_mels, n_mels, kernel_size=5, padding=2),
            nn.BatchNorm1d(n_mels),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Conv1d(n_mels, n_mels, kernel_size=5, padding=2),
            nn.BatchNorm1d(n_mels),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Conv1d(n_mels, n_mels, kernel_size=5, padding=2),
            nn.BatchNorm1d(n_mels),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Conv1d(n_mels, n_mels, kernel_size=5, padding=2),
            nn.BatchNorm1d(n_mels),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Conv1d(n_mels, n_mels, kernel_size=5, padding=2),
        )
        
    def forward(self, x, masks=None, ref_mels=None, f0=None, energy=None):
        """
        Forward pass through the acoustic decoder.
        
        Args:
            x: Expanded features from variance adaptor [batch_size, seq_len, d_model]
            masks: Masks for padded positions [batch_size, seq_len]
            ref_mels: Reference mel spectrograms for style transfer [batch_size, n_mels, time]
            f0: F0 contour [batch_size, seq_len]
            energy: Energy values [batch_size, seq_len]
            
        Returns:
            output_dict: Dictionary containing model outputs
        """
        batch_size, seq_len, d_model = x.size()
        
        # Process reference audio if provided and reference encoder is enabled
        ref_embedding = None
        if self.use_ref_encoder and ref_mels is not None:
            ref_embedding = self.reference_encoder(ref_mels)
            ref_embedding = self.ref_projection(ref_embedding)
        
        # Create attention mask from masks
        attention_mask = None
        if masks is not None:
            # Create causal mask (lower triangular)
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len), diagonal=1
            ).bool().to(masks.device)
            
            # Combine with padding mask
            padding_mask = masks.unsqueeze(1).expand(-1, seq_len, -1)
            attention_mask = causal_mask.unsqueeze(0) | padding_mask
            
            # Convert to format needed by nn.MultiheadAttention
            attention_mask = (~attention_mask).float().masked_fill(attention_mask, float('-inf'))
        
        # Transpose for transformer layers [seq_len, batch_size, d_model]
        x = x.transpose(0, 1)
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            # Create conditioning from reference embedding if available
            conditioning = None
            if ref_embedding is not None:
                conditioning = ref_embedding.unsqueeze(0).expand(seq_len, -1, -1)
            
            x = layer(x, attention_mask, conditioning)
        
        # Transpose back [batch_size, seq_len, d_model]
        x = x.transpose(0, 1)
        
        # Transpose for 1D convolutions [batch_size, d_model, seq_len]
        x = x.transpose(1, 2)
        
        # Apply upsampling
        x = self.upsampling(x)
        
        # Project to mel spectrogram
        mel_output = self.mel_projection(x)
        
        # Apply postnet
        postnet_output = self.postnet(mel_output)
        mel_postnet = mel_output + postnet_output
        
        # Create output dictionary
        output_dict = {
            'mel_output': mel_output,
            'mel_postnet': mel_postnet,
            'ref_embedding': ref_embedding
        }
        
        return output_dict
    
    def visualize_spectrograms(self, output_dict, ground_truth=None):
        """
        Create visualizations for predicted spectrograms.
        
        Args:
            output_dict: Dictionary with model outputs
            ground_truth: Dictionary with ground truth values
            
        Returns:
            vis_dict: Dictionary with visualization tensors
        """
        # Placeholder for visualization tensors
        vis_dict = {}
        
        # Extract spectrograms
        mel_output = output_dict['mel_output']
        mel_postnet = output_dict['mel_postnet']
        
        # Get ground truth if available
        gt_mel = None
        if ground_truth is not None and 'mel_spectrograms' in ground_truth:
            gt_mel = ground_truth['mel_spectrograms']
            
            # Create visualization tensor (later will be converted to matplotlib figure)
            # This would show predicted vs ground truth spectrograms side by side
            # Implementation would depend on tensorboard visualization methods
            pass
        
        return vis_dict