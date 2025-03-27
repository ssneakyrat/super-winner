"""
FutureVox model implementation.
Main model combining text encoder, duration and pitch predictors, flow decoder, and vocoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union

from config.model_config import FutureVoxConfig, ModelConfig


class TextEncoder(nn.Module):
    """Transformer-based text encoder for phoneme sequences."""
    
    def __init__(self, config, n_vocab):
        """
        Initialize text encoder.
        
        Args:
            config: Text encoder configuration
            n_vocab: Vocabulary size
        """
        super().__init__()
        
        self.embedding = nn.Embedding(
            n_vocab, config.hidden_dim, padding_idx=0
        )
        
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, 1000, config.hidden_dim)
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.n_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            batch_first=True
        )
        
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.n_layers
        )
        
    def forward(self, x, lengths=None):
        """
        Forward pass.
        
        Args:
            x: Phoneme token IDs [B, L]
            lengths: Sequence lengths [B]
        
        Returns:
            Encoded phoneme features [B, L, H]
        """
        # Create padding mask if lengths are provided
        padding_mask = None
        if lengths is not None:
            batch_size, max_len = x.shape
            padding_mask = torch.arange(max_len, device=x.device).expand(batch_size, max_len) >= lengths.unsqueeze(1)
        
        # Embed phoneme tokens
        x = self.embedding(x)  # [B, L, H]
        
        # Add positional encoding
        seq_len = x.shape[1]
        x = x + self.pos_embedding[:, :seq_len]
        
        # Apply transformer encoder
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        
        return x


class DurationPredictor(nn.Module):
    """
    Duration predictor module.
    Predicts phoneme durations in log domain.
    """
    
    def __init__(self, config, input_dim):
        """
        Initialize duration predictor.
        
        Args:
            config: Duration predictor configuration
            input_dim: Input feature dimension
        """
        super().__init__()
        
        self.conv_layers = nn.ModuleList()
        self.kernel_size = config.kernel_size
        
        # Initial layer
        self.conv_layers.append(
            nn.Sequential(
                nn.Conv1d(
                    input_dim, input_dim,
                    kernel_size=config.kernel_size,
                    padding=(config.kernel_size - 1) // 2
                ),
                nn.ReLU(),
                # Use GroupNorm instead of LayerNorm for Conv1D outputs
                nn.GroupNorm(1, input_dim),  # 1 group = InstanceNorm behavior
                nn.Dropout(config.dropout)
            )
        )
        
        # Second layer
        self.conv_layers.append(
            nn.Sequential(
                nn.Conv1d(
                    input_dim, input_dim,
                    kernel_size=config.kernel_size,
                    padding=(config.kernel_size - 1) // 2
                ),
                nn.ReLU(),
                # Use GroupNorm instead of LayerNorm for Conv1D outputs
                nn.GroupNorm(1, input_dim),  # 1 group = InstanceNorm behavior
                nn.Dropout(config.dropout)
            )
        )
        
        # Projection to scalar output
        self.proj = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input features [B, L, H]
            
        Returns:
            Log durations [B, L, 1]
        """
        # Transpose for 1D convolution
        x_conv = x.transpose(1, 2)  # [B, H, L]
        
        # Check if sequence length is too small and pad if necessary
        min_length = self.kernel_size
        if x_conv.size(2) < min_length:
            padding_needed = min_length - x_conv.size(2)
            x_conv = F.pad(x_conv, (0, padding_needed))
        
        # Apply convolution layers
        for layer in self.conv_layers:
            x_conv = layer(x_conv)
        
        # Transpose back
        x = x_conv.transpose(1, 2)  # [B, L, H]
        
        # Trim to original length if padded
        original_len = min(x.size(1), x.size(1))
        x = x[:, :original_len, :]
        
        # Project to scalar
        log_durations = self.proj(x)  # [B, L, 1]
        
        return log_durations


class F0Predictor(nn.Module):
    """
    F0 predictor module.
    Predicts F0 contour from phoneme features.
    """
    
    def __init__(self, config, input_dim):
        """
        Initialize F0 predictor.
        
        Args:
            config: F0 predictor configuration
            input_dim: Input feature dimension
        """
        super().__init__()
        
        self.conv_layers = nn.ModuleList()
        self.kernel_size = config.kernel_size
        
        # Initial layer
        self.conv_layers.append(
            nn.Sequential(
                nn.Conv1d(
                    input_dim, input_dim,
                    kernel_size=config.kernel_size,
                    padding=(config.kernel_size - 1) // 2
                ),
                nn.ReLU(),
                # Use GroupNorm instead of LayerNorm for Conv1D outputs
                nn.GroupNorm(1, input_dim),  # 1 group = InstanceNorm behavior
                nn.Dropout(config.dropout)
            )
        )
        
        # Second layer
        self.conv_layers.append(
            nn.Sequential(
                nn.Conv1d(
                    input_dim, input_dim,
                    kernel_size=config.kernel_size,
                    padding=(config.kernel_size - 1) // 2
                ),
                nn.ReLU(),
                # Use GroupNorm instead of LayerNorm for Conv1D outputs
                nn.GroupNorm(1, input_dim),  # 1 group = InstanceNorm behavior
                nn.Dropout(config.dropout)
            )
        )
        
        # Projection to scalar output
        self.proj = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input features [B, L, H]
            
        Returns:
            F0 values [B, L, 1]
        """
        # Transpose for 1D convolution
        x_conv = x.transpose(1, 2)  # [B, H, L]
        
        # Check if sequence length is too small and pad if necessary
        min_length = self.kernel_size
        if x_conv.size(2) < min_length:
            padding_needed = min_length - x_conv.size(2)
            x_conv = F.pad(x_conv, (0, padding_needed))
        
        # Apply convolution layers
        for layer in self.conv_layers:
            x_conv = layer(x_conv)
        
        # Transpose back and make sure we return to original sequence length
        x = x_conv.transpose(1, 2)  # [B, L, H]
        
        # Trim to original length if padded
        original_len = min(x.size(1), x.size(1))
        x = x[:, :original_len, :]
        
        # Project to scalar
        f0 = self.proj(x)  # [B, L, 1]
        
        return f0

class AffineCouplingLayer(nn.Module):
    """
    Affine coupling layer for normalizing flow.
    Based on the design in VITS and Flow-TTS.
    """
    
    def __init__(self, config, in_channels):
        """
        Initialize affine coupling layer.
        
        Args:
            config: Flow decoder configuration
            in_channels: Input channel dimension
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_dim = config.hidden_dim
        
        # Split in half
        self.half_channels = in_channels // 2
        
        # WaveNet-like dilated convolution network
        self.pre = nn.Conv1d(self.half_channels, self.hidden_dim, 1)
        
        self.convs = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()
        
        for i in range(3):  # 3 dilation cycles
            dilation_rates = [1, 3, 9]  # Dilation rates
            
            # Dilated convolution
            self.convs.append(
                nn.Conv1d(
                    self.hidden_dim, self.hidden_dim * 2,
                    kernel_size=config.kernel_size,
                    dilation=dilation_rates[i],
                    padding=int((config.kernel_size * dilation_rates[i] - dilation_rates[i]) / 2)
                )
            )
            
            # Residual and skip connections
            self.res_skip_layers.append(
                nn.Conv1d(self.hidden_dim, self.hidden_dim + self.half_channels * 2, 1)
            )
            
        # Output projection
        self.out = nn.Conv1d(self.hidden_dim, self.half_channels * 2, 1)
        
    def forward(self, x, reverse=False):
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, C, T]
            reverse: Whether to reverse the flow direction
            
        Returns:
            Transformed tensor [B, C, T] and log determinant
        """
        # Split in half
        xa, xb = torch.split(x, [self.half_channels, self.half_channels], 1)
        
        if not reverse:
            # Forward direction (encoding)
            h = self.pre(xa)
            log_s_total = 0
            
            for i in range(len(self.convs)):
                h_conv = self.convs[i](h)
                h_conv = F.gelu(h_conv)
                
                h_conv_chunks = torch.chunk(h_conv, 2, 1)
                h = h + h_conv_chunks[0]
                
                res_skip = self.res_skip_layers[i](h)
                skip = torch.chunk(res_skip, 2, 1)[1]
                
                if i < len(self.convs) - 1:
                    h = res_skip[:, :self.hidden_dim]
                
                # FIX: Ensure log_s and b have the same number of channels as xb
                log_s = skip[:, :self.half_channels]
                b = skip[:, self.half_channels:2*self.half_channels]  # Only take half_channels for b too
                
                # Ensure dimensions match before operating
                if log_s.size(1) != xb.size(1) or b.size(1) != xb.size(1):
                    # Debug info - can be removed after fix is confirmed
                    print(f"Dimension mismatch: log_s={log_s.size()}, b={b.size()}, xb={xb.size()}")
                    # Explicitly reshape to match dimensions if needed
                    log_s = log_s[:, :xb.size(1)]
                    b = b[:, :xb.size(1)]
                
                xb = torch.exp(log_s) * xb + b
                log_s_total = log_s_total + log_s
            
            # Concatenate transformed halves
            x = torch.cat([xa, xb], 1)
            logdet = torch.sum(log_s_total, [1, 2])
            
            return x, logdet
            
        else:
            # Reverse direction (decoding)
            h = self.pre(xa)
            
            for i in range(len(self.convs)):
                h_conv = self.convs[i](h)
                h_conv = F.gelu(h_conv)
                
                h_conv_chunks = torch.chunk(h_conv, 2, 1)
                h = h + h_conv_chunks[0]
                
                res_skip = self.res_skip_layers[i](h)
                skip = torch.chunk(res_skip, 2, 1)[1]
                
                if i < len(self.convs) - 1:
                    h = res_skip[:, :self.hidden_dim]
                
                # FIX: Ensure log_s and b have the same number of channels as xb
                log_s = skip[:, :self.half_channels]
                b = skip[:, self.half_channels:2*self.half_channels]  # Only take half_channels for b too
                
                # Ensure dimensions match before operating
                if log_s.size(1) != xb.size(1) or b.size(1) != xb.size(1):
                    log_s = log_s[:, :xb.size(1)]
                    b = b[:, :xb.size(1)]
                
                xb = (xb - b) / torch.exp(log_s)
            
            # Concatenate transformed halves
            x = torch.cat([xa, xb], 1)
            
            return x

class FlowDecoder(nn.Module):
    """
    Flow-based decoder.
    Transforms phoneme features to mel-spectrograms using normalizing flows.
    """
    
    def __init__(self, config, input_dim, output_dim):
        """
        Initialize flow decoder.
        
        Args:
            config: Flow decoder configuration
            input_dim: Input feature dimension
            output_dim: Output feature dimension (mel channels)
        """
        super().__init__()
        
        self.flows = nn.ModuleList()
        
        # Prior distribution parameters
        self.prior_lstm = nn.LSTM(
            input_dim, input_dim // 2,
            batch_first=True, bidirectional=True
        )
        
        self.prior_proj = nn.Linear(input_dim, output_dim * 2)
        
        # Flow layers
        for _ in range(config.n_flows):
            self.flows.append(
                AffineCouplingLayer(config, output_dim)
            )
            
            # Add invertible 1x1 convolution (permutation)
            conv = nn.Conv1d(output_dim, output_dim, kernel_size=1)
            nn.init.orthogonal_(conv.weight)
            conv.bias.data.zero_()
            self.flows.append(conv)
        
    def forward(self, x, x_mask=None, temperature=1.0, reverse=False):
        """
        Forward pass.
        
        Args:
            x: Input features [B, L, H]
            x_mask: Feature mask [B, 1, L]
            temperature: Sampling temperature
            reverse: Whether to run in reverse direction
            
        Returns:
            Mel-spectrogram and KL divergence
        """
        if not reverse:
            # Forward direction (during training)
            # Calculate prior distribution parameters
            x_lstm, _ = self.prior_lstm(x)
            prior_params = self.prior_proj(x_lstm)  # [B, L, output_dim*2]
            
            # Split into mean and log variance
            prior_mean, prior_log_var = torch.chunk(
                prior_params, 2, dim=2
            )
            
            # Sample from prior
            z = prior_mean + torch.randn_like(prior_mean) * torch.exp(prior_log_var * 0.5) * temperature
            
            # Apply mask if provided
            if x_mask is not None:
                z = z * x_mask.transpose(1, 2)
            
            # Transpose for the flow operations
            z = z.transpose(1, 2)  # [B, output_dim, L]
            
            # Apply flow transforms
            logdet_sum = 0
            
            for i, flow in enumerate(self.flows):
                if isinstance(flow, AffineCouplingLayer):
                    z, logdet = flow(z, reverse=False)
                    logdet_sum = logdet_sum + logdet
                else:
                    # 1x1 convolution - FIX: Use the weight directly without unsqueeze
                    z = F.conv1d(z, flow.weight, bias=flow.bias)
                    logdet = torch.logdet(flow.weight.squeeze(2)) * z.shape[2]
                    logdet_sum = logdet_sum + logdet
            
            # Calculate KL divergence
            prior_var = torch.exp(prior_log_var)
            kl = 0.5 * torch.sum(
                prior_var + prior_mean**2 - 1 - prior_log_var,
                dim=[1, 2]
            )
            
            # Transpose back
            z = z.transpose(1, 2)  # [B, L, output_dim]
            
            return z, kl
            
        else:
            # Reverse direction (during inference)
            # Calculate prior distribution parameters
            x_lstm, _ = self.prior_lstm(x)
            prior_params = self.prior_proj(x_lstm)  # [B, L, output_dim*2]
            
            # Split into mean and log variance
            prior_mean, prior_log_var = torch.chunk(
                prior_params, 2, dim=2
            )
            
            # Sample from prior
            z = prior_mean + torch.randn_like(prior_mean) * torch.exp(prior_log_var * 0.5) * temperature
            
            # Apply mask if provided
            if x_mask is not None:
                z = z * x_mask.transpose(1, 2)
            
            # Transpose for the flow operations
            z = z.transpose(1, 2)  # [B, output_dim, L]
            
            # Apply flow transforms in reverse order
            for flow in reversed(self.flows):
                if isinstance(flow, AffineCouplingLayer):
                    z = flow(z, reverse=True)
                else:
                    # Inverse 1x1 convolution - FIX: Properly handle dimensions for inverse
                    weight_2d = flow.weight.squeeze(2)  # Remove kernel dimension to get 2D tensor
                    weight_inv_2d = torch.inverse(weight_2d)  # Compute inverse of 2D tensor
                    weight_inv_3d = weight_inv_2d.unsqueeze(2)  # Add kernel dimension back to get 3D tensor
                    z = F.conv1d(z - flow.bias.unsqueeze(1), weight_inv_3d)
            
            # Transpose back
            z = z.transpose(1, 2)  # [B, L, output_dim]
            
            return z

class ResBlock(nn.Module):
    """
    Residual block for HiFi-GAN vocoder.
    """
    
    def __init__(self, channels, kernel_size, dilations):
        """
        Initialize residual block.
        
        Args:
            channels: Number of channels
            kernel_size: Kernel size
            dilations: List of dilation rates
        """
        super().__init__()
        
        self.convs = nn.ModuleList()
        
        for dilation in dilations:
            padding = (kernel_size * dilation - dilation) // 2
            self.convs.append(
                nn.Sequential(
                    nn.LeakyReLU(0.1),
                    nn.Conv1d(
                        channels, channels,
                        kernel_size=kernel_size, dilation=dilation,
                        padding=padding
                    ),
                    nn.LeakyReLU(0.1),
                    nn.Conv1d(
                        channels, channels,
                        kernel_size=1
                    )
                )
            )
            
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, C, T]
            
        Returns:
            Output tensor [B, C, T]
        """
        for conv in self.convs:
            x = x + conv(x)
        return x


class HiFiGAN(nn.Module):
    """
    Lightweight HiFi-GAN vocoder.
    Upsamples mel-spectrograms to waveform.
    """
    
    def __init__(self, config, input_dim):
        """
        Initialize HiFi-GAN vocoder.
        
        Args:
            config: Vocoder configuration
            input_dim: Input feature dimension (mel channels)
        """
        super().__init__()
        
        self.input_dim = input_dim
        
        # Initial convolution
        self.conv_pre = nn.Conv1d(input_dim, 256, kernel_size=7, padding=3)
        
        # Upsampling layers
        self.ups = nn.ModuleList()
        for i, (u_rate, k_size) in enumerate(zip(
            config.upsample_rates, config.upsample_kernel_sizes
        )):
            self.ups.append(
                nn.ConvTranspose1d(
                    256 // (2**i), 256 // (2**(i+1)),
                    k_size, stride=u_rate,
                    padding=(k_size - u_rate) // 2
                )
            )
        
        # Residual blocks
        self.resblocks = nn.ModuleList()
        for i, (k_size, d_rates) in enumerate(zip(
            config.resblock_kernel_sizes, config.resblock_dilation_sizes
        )):
            self.resblocks.append(
                ResBlock(
                    256 // (2**(len(config.upsample_rates))),
                    k_size, d_rates
                )
            )
        
        # Output convolution
        self.conv_post = nn.Conv1d(
            256 // (2**(len(config.upsample_rates))),
            1, kernel_size=7, padding=3
        )
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input mel-spectrogram [B, L, M]
            
        Returns:
            Waveform [B, 1, T]
        """
        # Transpose for 1D convolution
        x = x.transpose(1, 2)  # [B, M, L]
        
        # Initial convolution
        x = self.conv_pre(x)
        
        # Upsampling
        for up in self.ups:
            x = F.leaky_relu(x, 0.1)
            x = up(x)
        
        # Residual blocks
        for resblock in self.resblocks:
            x = resblock(x)
        
        # Output convolution
        x = F.leaky_relu(x, 0.1)
        x = self.conv_post(x)
        x = torch.tanh(x)
        
        return x


class FutureVox(nn.Module):
    """
    FutureVox model.
    Singing voice synthesis model with flow-based decoder and neural vocoder.
    """
    
    def __init__(self, config, n_vocab=100):
        """
        Initialize FutureVox model.
        
        Args:
            config: Model configuration
            n_vocab: Vocabulary size for phoneme tokens
        """
        super().__init__()
        
        self.config = config
        self.n_vocab = n_vocab
        
        # Text encoder
        self.text_encoder = TextEncoder(
            config.model.text_encoder, n_vocab
        )
        
        # Duration predictor
        self.duration_predictor = DurationPredictor(
            config.model.predictors.duration_predictor,
            config.model.text_encoder.hidden_dim
        )
        
        # F0 predictor
        self.f0_predictor = F0Predictor(
            config.model.predictors.f0_predictor,
            config.model.text_encoder.hidden_dim
        )
        
        # Flow decoder
        self.flow_decoder = FlowDecoder(
            config.model.flow_decoder,
            config.model.text_encoder.hidden_dim,
            config.data.mel_channels
        )
        
        # Neural vocoder
        self.vocoder = HiFiGAN(
            config.model.vocoder,
            config.data.mel_channels
        )
        
    def length_regulate(self, x, durations):
        """
        Expand phoneme features according to predicted durations.
        
        Args:
            x: Phoneme features [B, L, H]
            durations: Duration for each phoneme [B, L]
            
        Returns:
            Expanded features [B, T, H]
        """
        batch_size, seq_len, hidden_dim = x.shape
        
        # Ensure durations are integers
        durations = durations.long()
        
        # Calculate output length
        output_lengths = torch.sum(durations, dim=1)
        max_output_len = max(output_lengths.max().item(), 1)  # Ensure at least length 1 to avoid empty tensors
        
        # Initialize output tensor
        expanded = torch.zeros(
            batch_size, max_output_len, hidden_dim,
            device=x.device
        )
        
        # Fill expanded tensor
        for b in range(batch_size):
            current_pos = 0
            for i in range(seq_len):
                current_duration = durations[b, i].item()
                if current_duration > 0:
                    if current_pos + current_duration <= max_output_len:  # Add boundary check
                        expanded[b, current_pos:current_pos + current_duration] = x[b, i].unsqueeze(0).expand(
                            current_duration, -1
                        )
                        current_pos += current_duration
                    else:
                        # Truncate if needed
                        remaining = max_output_len - current_pos
                        if remaining > 0:
                            expanded[b, current_pos:max_output_len] = x[b, i].unsqueeze(0).expand(
                                remaining, -1
                            )
                        break
        
        return expanded, output_lengths
    
    def forward(
        self,
        phonemes,
        phoneme_lengths,
        durations=None,
        f0=None,
        mel=None,
        mel_lengths=None,
        temperature=1.0
    ):
        """
        Forward pass with dimension fixes.
        
        Args:
            phonemes: Phoneme token IDs [B, L]
            phoneme_lengths: Phoneme sequence lengths [B]
            durations: Ground truth durations (optional) [B, L]
            f0: Ground truth F0 contour (optional) [B, T]
            mel: Ground truth mel-spectrogram (optional) [B, M, T]
            mel_lengths: Mel-spectrogram lengths [B]
            temperature: Sampling temperature
            
        Returns:
            Dictionary of outputs and losses
        """
        # Text encoding
        encoded = self.text_encoder(phonemes, phoneme_lengths)  # [B, L, H]
        
        # Create masks
        phoneme_mask = torch.unsqueeze(
            torch.arange(0, phonemes.size(1), device=phonemes.device)[None, :] < phoneme_lengths[:, None],
            1
        ).float()  # [B, 1, L]
        
        # Duration prediction
        log_durations_pred = self.duration_predictor(encoded).squeeze(-1)  # [B, L]
        
        # Apply mask to predicted durations
        log_durations_pred = log_durations_pred * phoneme_mask.squeeze(1)
        
        # Use ground truth or predicted durations
        if durations is not None:
            # Training mode
            durations_for_expansion = durations
            # Ensure same shape for loss calculation
            log_durations_gt = torch.log(durations.float().clamp(min=1))  # [B, L]
            
            # Ensure shapes match before computing loss
            if log_durations_pred.shape == log_durations_gt.shape:
                duration_loss = F.mse_loss(log_durations_pred, log_durations_gt, reduction='none')
                duration_loss = (duration_loss * phoneme_mask.squeeze(1)).sum() / phoneme_mask.sum().clamp(min=1e-5)
            else:
                # Handle shape mismatch - trim to match sizes
                min_len = min(log_durations_pred.size(1), log_durations_gt.size(1))
                duration_loss = F.mse_loss(
                    log_durations_pred[:, :min_len], 
                    log_durations_gt[:, :min_len], 
                    reduction='none'
                )
                mask_trimmed = phoneme_mask.squeeze(1)[:, :min_len]
                duration_loss = (duration_loss * mask_trimmed).sum() / mask_trimmed.sum().clamp(min=1e-5)
        else:
            # Inference mode
            durations_for_expansion = torch.exp(log_durations_pred) - 1
            durations_for_expansion = torch.clamp_min(durations_for_expansion, 0)
            durations_for_expansion = torch.round(durations_for_expansion).long()
            duration_loss = None
        
        # Length regulation with safety checks
        expanded, output_lengths = self.length_regulate(encoded, durations_for_expansion)  # [B, T, H]
        
        # Create expanded mask - ensure dimensions are valid
        if expanded.size(1) > 0:  # Only create mask if we have a valid sequence length
            expanded_mask = torch.unsqueeze(
                torch.arange(0, expanded.size(1), device=expanded.device)[None, :] < output_lengths[:, None],
                1
            ).float()  # [B, 1, T]
            
            # F0 prediction
            f0_pred = self.f0_predictor(expanded).squeeze(-1)  # [B, T]
            
            # Safety check for dimensions before masking
            if f0_pred.size(1) == expanded_mask.size(2):
                # Apply mask to predicted F0
                f0_pred = f0_pred * expanded_mask.squeeze(1)
            else:
                # Handle dimension mismatch
                min_len = min(f0_pred.size(1), expanded_mask.size(2))
                f0_pred_safe = torch.zeros_like(f0_pred)
                f0_pred_safe[:, :min_len] = f0_pred[:, :min_len] * expanded_mask.squeeze(1)[:, :min_len]
                f0_pred = f0_pred_safe
        else:
            # Handle empty sequence case
            batch_size = expanded.size(0)
            expanded = torch.zeros(batch_size, 1, expanded.size(2), device=expanded.device)
            expanded_mask = torch.ones(batch_size, 1, 1, device=expanded.device)
            f0_pred = torch.zeros(batch_size, 1, device=expanded.device)
            output_lengths = torch.ones(batch_size, device=expanded.device)
        
        # F0 loss if ground truth is provided
        if f0 is not None:
            # Calculate F0 loss only on voiced frames
            voiced_mask = (f0 > 0).float() * expanded_mask.squeeze(1)
            
            # Handle dimension mismatch
            if f0_pred.size(1) != f0.size(1):
                min_len = min(f0_pred.size(1), f0.size(1))
                voiced_mask_trimmed = voiced_mask[:, :min_len]
                
                if voiced_mask_trimmed.sum() > 0:
                    f0_loss = F.mse_loss(
                        f0_pred[:, :min_len] * voiced_mask_trimmed, 
                        f0[:, :min_len] * voiced_mask_trimmed, 
                        reduction='sum'
                    ) / (voiced_mask_trimmed.sum() + 1e-6)
                else:
                    f0_loss = torch.tensor(0.0, device=f0.device)
            else:
                if voiced_mask.sum() > 0:
                    f0_loss = F.mse_loss(f0_pred * voiced_mask, f0 * voiced_mask, reduction='sum') / (voiced_mask.sum() + 1e-6)
                else:
                    f0_loss = torch.tensor(0.0, device=f0.device)
        else:
            f0_loss = None
        
        # Flow-based decoder
        if mel is not None:
            # Training mode - forward flow
            mel_pred, kl_loss = self.flow_decoder(
                expanded, expanded_mask, temperature=temperature
            )
        else:
            # Inference mode - reverse flow
            mel_pred = self.flow_decoder(
                expanded, expanded_mask, temperature=temperature, reverse=True
            )
            kl_loss = None
        
        # Apply expanded mask to predicted mel
        if mel_pred.size(1) == expanded_mask.size(2):
            mel_pred = mel_pred * expanded_mask.transpose(1, 2)
        else:
            # Handle dimension mismatch
            min_len = min(mel_pred.size(1), expanded_mask.size(2))
            safe_mask = expanded_mask[:, :, :min_len]
            mel_pred_safe = torch.zeros_like(mel_pred)
            mel_pred_safe[:, :min_len] = mel_pred[:, :min_len] * safe_mask.transpose(1, 2)
            mel_pred = mel_pred_safe
        
        # Calculate mel loss if ground truth is provided
        if mel is not None and mel_lengths is not None:
            # Create mel mask
            mel_mask = torch.unsqueeze(
                torch.arange(0, mel.size(2), device=mel.device)[None, :] < mel_lengths[:, None],
                1
            ).float()  # [B, 1, T]
            
            # Transport mel to match the model's output format
            mel = mel.transpose(1, 2)  # [B, T, M]
            
            # Handle dimension mismatch for mel loss calculation
            if mel_pred.size(1) != mel.size(1):
                min_len = min(mel_pred.size(1), mel.size(1))
                mel_pred_trimmed = mel_pred[:, :min_len]
                mel_trimmed = mel[:, :min_len]
                mask_trimmed = mel_mask.transpose(1, 2)[:, :min_len]
                
                mel_loss = F.l1_loss(
                    mel_pred_trimmed * mask_trimmed, 
                    mel_trimmed * mask_trimmed, 
                    reduction='sum'
                ) / (mask_trimmed.sum() + 1e-6)
            else:
                # Calculate masked L1 loss
                mel_loss = F.l1_loss(
                    mel_pred * mel_mask.transpose(1, 2), 
                    mel * mel_mask.transpose(1, 2), 
                    reduction='sum'
                ) / (mel_mask.sum() + 1e-6)
        else:
            mel_loss = None
        
        # Generate waveform with vocoder
        if mel_pred is not None:
            # Ensure mel_pred_for_vocoder has shape [B, 80, T]
            if mel_pred.size(1) == self.config.data.mel_channels:
                # already [B, 80, T]
                mel_pred_for_vocoder = mel_pred
            elif mel_pred.size(2) == self.config.data.mel_channels:
                # [B, T, 80] -> [B, 80, T]
                mel_pred_for_vocoder = mel_pred.transpose(1, 2)
            else:
                # Need to reshape to ensure 80 is in the middle dimension
                B = mel_pred.size(0)
                rest = mel_pred.numel() // (B * self.config.data.mel_channels)
                mel_pred_for_vocoder = mel_pred.reshape(B, self.config.data.mel_channels, rest)
            
            waveform = self.vocoder(mel_pred_for_vocoder)
        else:
            waveform = None
        
        # Collect outputs and losses
        outputs = {
            "encoded": encoded,
            "expanded": expanded,
            "log_durations_pred": log_durations_pred,
            "durations_pred": durations_for_expansion,
            "f0_pred": f0_pred,
            "mel_pred": mel_pred,
            "waveform": waveform
        }
        
        losses = {}
        if duration_loss is not None:
            losses["duration_loss"] = duration_loss
        if f0_loss is not None:
            losses["f0_loss"] = f0_loss
        if mel_loss is not None:
            losses["mel_loss"] = mel_loss
        if kl_loss is not None:
            losses["kl_loss"] = kl_loss
        
        return outputs, losses