import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class LengthRegulator(nn.Module):
    """
    Length regulator to expand phoneme-level features to frame-level features
    based on predicted durations.
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x, durations, max_length=None):
        """
        Expand input according to predicted durations.
        
        Args:
            x: Input tensor [batch_size, seq_len, channels]
            durations: Predicted durations [batch_size, seq_len]
            max_length: Maximum length for output sequences
            
        Returns:
            expanded: Expanded tensor [batch_size, expanded_len, channels]
            mel_masks: Mask for padded frames [batch_size, expanded_len]
        """
        batch_size, seq_len, channels = x.size()
        
        # Handle edge case
        if max_length is None:
            max_length = torch.max(torch.sum(durations, dim=1)).item()
        
        # Round durations and convert to int
        durations = torch.round(durations).long()
        
        # Limit durations to ensure we don't exceed max_length
        if max_length is not None:
            durations = self._limit_durations(durations, max_length)
            
        expanded = torch.zeros(batch_size, max_length, channels).to(x.device)
        mel_masks = torch.ones(batch_size, max_length).to(x.device).bool()
        
        for i in range(batch_size):
            current_len = 0
            for j in range(seq_len):
                if durations[i, j] > 0:  # Skip padding phones
                    end = current_len + durations[i, j]
                    expanded[i, current_len:end, :] = x[i, j, :].unsqueeze(0).expand(durations[i, j], -1)
                    mel_masks[i, current_len:end] = False  # Mark as not padded
                    current_len = end
                    
                    if current_len >= max_length:
                        break
        
        return expanded, mel_masks
    
    def _limit_durations(self, durations, max_length):
        """Limit durations to ensure total length doesn't exceed max_length."""
        sums = torch.cumsum(durations, dim=1)
        beyond = sums > max_length
        
        # Truncate durations if they go beyond max_length
        if torch.any(beyond):
            durations = durations.clone()
            for i in range(durations.size(0)):
                if torch.any(beyond[i]):
                    idx = torch.argmax(beyond[i].long())
                    durations[i, idx] -= (sums[i, idx] - max_length)
                    durations[i, idx+1:] = 0
        
        return durations

class VariancePredictor(nn.Module):
    """
    Base class for pitch, duration, and energy predictors.
    """
    def __init__(self, d_model, kernel_size=3, dropout=0.1, out_dim=1):
        super().__init__()
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size, padding=(kernel_size-1)//2),
            nn.ReLU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size, padding=(kernel_size-1)//2),
            nn.ReLU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        
        # Output layer
        self.output_layer = nn.Linear(d_model, out_dim)
        
    def forward(self, x, masks=None):
        """
        Forward pass through the variance predictor.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            masks: Masks for padded positions [batch_size, seq_len]
            
        Returns:
            predictions: Predicted values [batch_size, seq_len, out_dim]
        """
        # Transpose for 1D convolution [batch_size, d_model, seq_len]
        x = x.transpose(1, 2)
        
        # Apply convolutional layers
        x = self.conv_layers(x)
        
        # Transpose back to [batch_size, seq_len, d_model]
        x = x.transpose(1, 2)
        
        # Apply output layer
        predictions = self.output_layer(x)
        
        # Apply mask
        if masks is not None:
            predictions = predictions.masked_fill(masks.unsqueeze(-1), 0.0)
            
        return predictions

class PitchPredictor(VariancePredictor):
    """
    Pitch predictor with support for vibrato modeling.
    """
    def __init__(self, d_model, kernel_size=3, dropout=0.1):
        # Predict F0, vibrato rate, and vibrato depth (3 values)
        super().__init__(d_model, kernel_size, dropout, out_dim=3)
        
        # Additional projection for note-based pitch reference
        self.note_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x, masks=None, note_pitch=None):
        """
        Forward pass with vibrato parameters.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            masks: Masks for padded positions [batch_size, seq_len]
            note_pitch: Optional tensor with note pitches [batch_size, seq_len]
            
        Returns:
            pitch_params: Predicted F0, vibrato rate, and depth [batch_size, seq_len, 3]
        """
        # If note pitch is provided, use it as additional conditioning
        if note_pitch is not None:
            note_features = self.note_proj(note_pitch)
            x = x + note_features
        
        return super().forward(x, masks)
    
    def generate_f0_with_vibrato(self, pitch_params, durations, sample_rate=22050, hop_length=256):
        """
        Generate frame-level F0 contour with vibrato.
        
        Args:
            pitch_params: Predicted pitch parameters [batch_size, seq_len, 3]
            durations: Predicted durations [batch_size, seq_len]
            sample_rate: Audio sample rate
            hop_length: Hop length for frames
            
        Returns:
            f0_contour: Frame-level F0 contour with vibrato [batch_size, frame_len]
        """
        batch_size, seq_len = durations.shape
        
        # Split parameters
        f0_base = pitch_params[:, :, 0]  # Base F0 (Hz)
        vibrato_rate = pitch_params[:, :, 1]  # Vibrato rate (Hz)
        vibrato_depth = pitch_params[:, :, 2]  # Vibrato depth (semitones)
        
        # Length regulator to expand to frame level
        lr = LengthRegulator()
        expanded_f0, _ = lr(f0_base.unsqueeze(-1), durations)
        expanded_rate, _ = lr(vibrato_rate.unsqueeze(-1), durations)
        expanded_depth, _ = lr(vibrato_depth.unsqueeze(-1), durations)
        
        # Calculate time in seconds for each frame
        frame_len = expanded_f0.size(1)
        times = torch.arange(frame_len).to(expanded_f0.device) * hop_length / sample_rate
        times = times.unsqueeze(0).expand(batch_size, -1)
        
        # Apply vibrato: F0_vibrato(t) = F0(t) + depth * sin(2π * rate * t)
        vibrato_mod = expanded_depth * torch.sin(2 * math.pi * expanded_rate * times)
        
        # Convert semitone modulation to Hz (multiplicative)
        f0_with_vibrato = expanded_f0 * torch.pow(2, vibrato_mod / 12.0)
        
        return f0_with_vibrato.squeeze(-1)

class DurationPredictor(VariancePredictor):
    """
    Duration predictor with musical rhythm awareness.
    """
    def __init__(self, d_model, kernel_size=3, dropout=0.1):
        super().__init__(d_model, kernel_size, dropout, out_dim=1)
        
        # Additional projection for rhythm information
        self.rhythm_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x, masks=None, rhythm_info=None):
        """
        Forward pass with rhythm conditioning.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            masks: Masks for padded positions [batch_size, seq_len]
            rhythm_info: Optional tensor with rhythm information [batch_size, seq_len, d_model]
            
        Returns:
            durations: Predicted log durations [batch_size, seq_len, 1]
        """
        # If rhythm information is provided, use it as additional conditioning
        if rhythm_info is not None:
            rhythm_features = self.rhythm_proj(rhythm_info)
            x = x + rhythm_features
        
        # Get log durations
        log_durations = super().forward(x, masks)
        
        return log_durations
    
    def get_durations(self, log_durations, tempo_factor=1.0):
        """
        Convert log durations to actual frame durations with tempo adjustment.
        
        Args:
            log_durations: Predicted log durations [batch_size, seq_len, 1]
            tempo_factor: Factor to adjust the tempo (1.0 = normal, <1.0 = faster, >1.0 = slower)
            
        Returns:
            durations: Frame durations [batch_size, seq_len]
        """
        # Apply softplus to ensure positive durations
        durations = F.softplus(log_durations).squeeze(-1)
        
        # Apply tempo factor
        durations = durations * tempo_factor
        
        return durations

class EnergyPredictor(VariancePredictor):
    """
    Energy predictor for controlling dynamics.
    """
    def __init__(self, d_model, kernel_size=3, dropout=0.1):
        super().__init__(d_model, kernel_size, dropout, out_dim=1)
        
    def forward(self, x, masks=None):
        """
        Forward pass for energy prediction.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            masks: Masks for padded positions [batch_size, seq_len]
            
        Returns:
            energy: Predicted energy values [batch_size, seq_len, 1]
        """
        return super().forward(x, masks)

class AdvancedVarianceAdaptor(nn.Module):
    """
    Advanced variance adaptor with pitch, duration, and energy prediction
    for singing voice synthesis.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Model dimensions
        d_model = config.get('variance_adaptor', {}).get('d_model', 256)
        dropout = config.get('variance_adaptor', {}).get('dropout', 0.1)
        
        # Variance predictors
        self.pitch_predictor = PitchPredictor(d_model, dropout=dropout)
        self.duration_predictor = DurationPredictor(d_model, dropout=dropout)
        self.energy_predictor = EnergyPredictor(d_model, dropout=dropout)
        
        # Length regulator
        self.length_regulator = LengthRegulator()
        
        # Singer identity conditioning
        self.use_singer_embedding = config.get('variance_adaptor', {}).get('use_singer_embedding', False)
        if self.use_singer_embedding:
            self.singer_emb_dim = config.get('variance_adaptor', {}).get('singer_emb_dim', 64)
            num_singers = config.get('variance_adaptor', {}).get('num_singers', 10)
            self.singer_embedding = nn.Embedding(num_singers, self.singer_emb_dim)
            self.singer_projection = nn.Linear(self.singer_emb_dim, d_model)
        
    def forward(self, x, phone_masks=None, note_pitch=None, rhythm_info=None, 
                singer_ids=None, tempo_factor=1.0, ground_truth=None):
        """
        Forward pass through the advanced variance adaptor.
        
        Args:
            x: Input features from encoder [batch_size, seq_len, d_model]
            phone_masks: Masks for padded positions [batch_size, seq_len]
            note_pitch: Optional tensor with note pitches [batch_size, seq_len]
            rhythm_info: Optional tensor with rhythm information [batch_size, seq_len, d_model]
            singer_ids: Optional tensor with singer identity indices [batch_size]
            tempo_factor: Factor to adjust the tempo (1.0 = normal)
            ground_truth: Dictionary with ground truth values for teacher forcing
            
        Returns:
            output_dict: Dictionary containing model outputs and predictions
        """
        # Apply singer identity conditioning if enabled
        if self.use_singer_embedding and singer_ids is not None:
            singer_emb = self.singer_embedding(singer_ids)
            singer_features = self.singer_projection(singer_emb).unsqueeze(1)
            x = x + singer_features
        
        # Predict variances
        pitch_params = self.pitch_predictor(x, phone_masks, note_pitch)
        log_durations = self.duration_predictor(x, phone_masks, rhythm_info)
        energy = self.energy_predictor(x, phone_masks)
        
        # Get actual durations
        durations = self.duration_predictor.get_durations(log_durations, tempo_factor)
        
        # Get F0 contour with vibrato
        f0_contour = None
        if ground_truth is None or 'durations' not in ground_truth:
            f0_contour = self.pitch_predictor.generate_f0_with_vibrato(
                pitch_params, durations, 
                sample_rate=self.config['audio']['sample_rate'],
                hop_length=self.config['audio']['hop_length']
            )
        
        # Use ground truth values for teacher forcing during training if provided
        use_durations = ground_truth['durations'] if ground_truth and 'durations' in ground_truth else durations
        
        # Expand input using length regulator
        expanded_features, mel_masks = self.length_regulator(x, use_durations)
        
        # Create output dictionary
        output_dict = {
            'expanded_features': expanded_features,
            'mel_masks': mel_masks,
            'pitch_params': pitch_params,
            'log_durations': log_durations,
            'durations': durations,
            'energy': energy,
            'f0_contour': f0_contour
        }
        
        return output_dict
    
    def visualize_predictions(self, output_dict, ground_truth=None):
        """
        Create visualizations for predicted variances.
        
        Args:
            output_dict: Dictionary with model outputs
            ground_truth: Dictionary with ground truth values
            
        Returns:
            vis_dict: Dictionary with visualization tensors
        """
        # Placeholder for visualization tensors
        vis_dict = {}
        
        # Visualization implementations would be added here
        # This would include plots of predicted vs ground truth values for:
        # - F0 contours with vibrato
        # - Durations
        # - Energy levels
        
        return vis_dict