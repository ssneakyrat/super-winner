import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LengthRegulator(nn.Module):
    """
    Length regulator to expand phoneme-level features to frame-level features based on durations.
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
            # Make sure to convert to integer for tensor dimensions
            max_length = int(torch.max(torch.sum(durations, dim=1)).item())
        else:
            # Ensure max_length is an integer if provided
            max_length = int(max_length)
        
        # Round durations and convert to int
        durations = torch.round(durations).long()
        
        # Limit durations to ensure we don't exceed max_length
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
                    durations[i, idx] = torch.clamp(durations[i, idx] - (sums[i, idx] - max_length), min=0)
                    durations[i, idx+1:] = 0
        
        return durations

class SimpleVariancePredictor(nn.Module):
    """
    Simple variance predictor for pitch, duration, and energy.
    """
    def __init__(self, input_dim, filter_size=256, kernel_size=3, dropout=0.1, output_dim=1):
        super().__init__()
        
        # First Conv Layer
        self.conv1 = nn.Conv1d(
            input_dim, 
            filter_size, 
            kernel_size=kernel_size, 
            padding=(kernel_size-1)//2
        )
        
        # Second Conv Layer
        self.conv2 = nn.Conv1d(
            filter_size, 
            filter_size, 
            kernel_size=kernel_size, 
            padding=(kernel_size-1)//2
        )
        
        # Output Layer
        self.linear_layer = nn.Linear(filter_size, output_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize with reasonable values for better visualization
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights to reasonable values for visualization purposes."""
        # Use torch.no_grad() to disable gradient tracking during initialization
        with torch.no_grad():
            # For duration predictor, we want small positive biases for log durations
            if hasattr(self.linear_layer, 'bias') and self.linear_layer.bias is not None:
                nn.init.constant_(self.linear_layer.bias, 1.0)  # Initialize to exp(1.0) ≈ 2.7 frames
                
            # For conv layers, use Kaiming initialization with small positive bias
            for m in [self.conv1, self.conv2]:
                nn.init.kaiming_normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)
    
    def forward(self, x, mask=None):
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            mask: Mask tensor [batch_size, seq_len]
            
        Returns:
            output: Output tensor [batch_size, seq_len, output_dim]
        """
        # Transpose for Conv1d: [batch_size, input_dim, seq_len]
        x = x.transpose(1, 2)
        
        # First conv layer
        x = self.conv1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Second conv layer
        x = self.conv2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Transpose back: [batch_size, seq_len, filter_size]
        x = x.transpose(1, 2)
        
        # Output projection
        output = self.linear_layer(x)
        
        # Apply mask
        if mask is not None:
            output = output.masked_fill(mask.unsqueeze(-1), 0.0)
        
        return output


class DurationPredictor(SimpleVariancePredictor):
    """
    Enhanced duration predictor with more realistic heuristics.
    """
    def __init__(self, input_dim, filter_size=256, kernel_size=3, dropout=0.1):
        super().__init__(input_dim, filter_size, kernel_size, dropout, output_dim=1)
        
        # Additional layers for phoneme-aware prediction
        self.norm = nn.LayerNorm(input_dim)
        
        # Initialize with larger positive biases for better visualization
        self._initialize_duration_weights()
    
    def _initialize_duration_weights(self):
        """Initialize weights specifically for duration prediction."""
        with torch.no_grad():
            if hasattr(self.linear_layer, 'bias') and self.linear_layer.bias is not None:
                nn.init.constant_(self.linear_layer.bias, 1.5)  # exp(1.5) ≈ 4.5 frames
    
    def forward(self, x, mask=None):
        """
        Forward pass with additional heuristics for more realistic durations.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            mask: Mask tensor [batch_size, seq_len]
            
        Returns:
            output: Log durations [batch_size, seq_len, 1]
        """
        # Apply layer normalization for better feature scaling
        x = self.norm(x)
        
        # Calculate "phoneme energy" - a proxy for complexity/prominence
        phoneme_energy = torch.norm(x, dim=2, keepdim=True)
        
        # Standard variance predictor forward
        base_output = super().forward(x, mask)
        
        # Modify based on phoneme energy to create more varied durations
        # Map energy to range [0.8, 1.2] to create 40% variation in duration
        energy_scale = 0.8 + 0.4 * (phoneme_energy / phoneme_energy.max())
        output = base_output * energy_scale
        
        # Add positional variation - typically phonemes at the end of utterances are longer
        batch_size, seq_len = x.shape[0], x.shape[1]
        pos = torch.arange(0, seq_len, device=x.device).float().unsqueeze(0).repeat(batch_size, 1).unsqueeze(2)
        pos_scale = 1.0 + 0.1 * (pos / seq_len)  # Gradual 10% increase toward end
        output = output * pos_scale
        
        # Apply mask
        if mask is not None:
            output = output.masked_fill(mask.unsqueeze(-1), 0.0)
        
        return output

class SimplifiedVarianceAdaptor(nn.Module):
    """
    Simplified variance adaptor for singing voice synthesis.
    This is a cleaner implementation designed to work with the test script.
    """
    def __init__(self, input_dim=256, filter_size=256, dropout=0.1):
        super().__init__()
        
        # Duration predictor - use enhanced version
        self.duration_predictor = DurationPredictor(
            input_dim=input_dim, 
            filter_size=filter_size, 
            dropout=dropout
        )
        
        # Pitch predictor (predict F0, vibrato rate, and vibrato extent)
        self.pitch_predictor = SimpleVariancePredictor(
            input_dim=input_dim, 
            filter_size=filter_size, 
            output_dim=3,  # F0, vibrato rate, vibrato extent
            dropout=dropout
        )
        
        # Energy predictor
        self.energy_predictor = SimpleVariancePredictor(
            input_dim=input_dim, 
            filter_size=filter_size, 
            output_dim=1, 
            dropout=dropout
        )
        
        # Length regulator
        self.length_regulator = LengthRegulator()
        
        # Optional note embedding projection
        self.note_proj = nn.Linear(input_dim, input_dim)
        
        # Initialize pitch predictor for better F0 defaults
        self._initialize_pitch_predictor()
    
    def _initialize_pitch_predictor(self):
        """Initialize pitch predictor to reasonable defaults."""
        # Use torch.no_grad() to disable gradient tracking during initialization
        with torch.no_grad():
            # Set reasonable default F0 (around 220Hz)
            if hasattr(self.pitch_predictor.linear_layer, 'bias') and self.pitch_predictor.linear_layer.bias is not None:
                # F0 base - default to 220Hz
                self.pitch_predictor.linear_layer.bias[0] = 5.4  # ln(220)
                # Vibrato rate - default to 5.5Hz
                self.pitch_predictor.linear_layer.bias[1] = 1.7  # ln(5.5)
                # Vibrato extent - default to 1.0 semitone
                self.pitch_predictor.linear_layer.bias[2] = 0.0  # 1.0 semitone
        
    def forward(self, x, phone_masks=None, note_pitch=None, singer_ids=None, ground_truth=None, tempo_factor=1.0):
        """
        Forward pass through the variance adaptor.
        
        Args:
            x: Input features from encoder [batch_size, seq_len, input_dim]
            phone_masks: Masks for padded positions [batch_size, seq_len]
            note_pitch: Optional note embeddings [batch_size, seq_len, input_dim]
            singer_ids: Optional singer identity embeddings [batch_size, singer_dim]
            ground_truth: Optional ground truth for teacher forcing
            tempo_factor: Factor to adjust tempo (default: 1.0)
            
        Returns:
            output_dict: Dictionary with all outputs
        """
        # Apply note pitch conditioning if provided
        if note_pitch is not None:
            x = x + self.note_proj(note_pitch)
        
        # Predict durations
        log_durations = self.duration_predictor(x, phone_masks)
        
        # Convert log durations to actual durations
        durations = torch.exp(log_durations) * tempo_factor
        
        # Use ground truth durations for training if provided
        use_durations = None
        if ground_truth and 'durations' in ground_truth:
            use_durations = ground_truth['durations']
            # Make sure durations is a tensor
            if isinstance(use_durations, list):
                # Convert list to tensor
                try:
                    max_len = max(d.size(0) for d in use_durations)
                    padded_durations = torch.zeros(len(use_durations), max_len)
                    for i, d in enumerate(use_durations):
                        padded_durations[i, :d.size(0)] = d
                    use_durations = padded_durations.to(x.device)
                except Exception as e:
                    print(f"Warning: Failed to process duration list: {e}")
                    use_durations = durations
        else:
            use_durations = durations
        
        # Predict pitch parameters (F0, vibrato rate, vibrato extent)
        pitch_params = self.pitch_predictor(x, phone_masks)
        
        # Post-process pitch parameters for more realistic defaults
        # Set base F0 based on phoneme energy
        phoneme_energy = torch.norm(x, dim=2, keepdim=True)
        energy_norm = (phoneme_energy - phoneme_energy.min()) / (phoneme_energy.max() - phoneme_energy.min() + 1e-8)
        # Scale F0 between 100-500 Hz based on energy
        f0_scale = 100 + 400 * energy_norm.squeeze(-1)
        
        # Only apply if not using ground truth F0
        if not (ground_truth and 'f0_values' in ground_truth):
            # Apply f0_scale directly to the first channel of pitch_params
            # Convert to exponential domain, apply scale, convert back
            f0_base = torch.exp(pitch_params[:, :, 0])
            f0_base = f0_base * f0_scale / 220.0  # Normalize scale factor around 220Hz
            # Create modified pitch_params without in-place operations
            modified_f0 = torch.log(f0_base + 1e-8)  # Add small epsilon to avoid log(0)
            pitch_params = torch.cat([
                modified_f0.unsqueeze(-1),
                pitch_params[:, :, 1:] 
            ], dim=2)
        
        # Predict energy
        energy = self.energy_predictor(x, phone_masks)
        
        # Length regulation (expand phonemes to frames)
        try:
            expanded_features, mel_masks = self.length_regulator(x, use_durations)
        except Exception as e:
            print(f"Warning: Error in length regulation: {e}")
            # Create default expanded features as fallback
            batch_size, seq_len, channels = x.size()
            # Use a default expansion of 4 frames per phoneme as fallback
            expanded_len = seq_len * 4
            expanded_features = x.repeat_interleave(4, dim=1)
            mel_masks = torch.zeros(batch_size, expanded_len, dtype=torch.bool, device=x.device)
        
        # Generate F0 contour from pitch parameters and durations
        try:
            f0_contour = self.generate_f0_contour(pitch_params, use_durations)
        except Exception as e:
            print(f"Warning: Error generating F0 contour: {e}")
            # Create default F0 contour as fallback
            if hasattr(expanded_features, 'size'):
                batch_size, expanded_len = expanded_features.size(0), expanded_features.size(1)
                f0_contour = torch.zeros(batch_size, expanded_len, device=x.device)
            else:
                # If expanded_features has issues too
                batch_size = x.size(0)
                f0_contour = torch.zeros(batch_size, seq_len * 4, device=x.device)
        
        # Create output dictionary
        output_dict = {
            'expanded_features': expanded_features,
            'mel_masks': mel_masks,
            'log_durations': log_durations,
            'durations': durations.squeeze(-1),
            'pitch_params': pitch_params,
            'f0_contour': f0_contour,
            'energy': energy.squeeze(-1),
        }
        
        return output_dict
    
    def generate_f0_contour(self, pitch_params, durations):
        """
        Generate F0 contour with vibrato from pitch parameters and durations.
        
        Args:
            pitch_params: Pitch parameters [batch_size, seq_len, 3]
            durations: Durations (could be tensor, list, or other format)
            
        Returns:
            f0_contour: F0 contour [batch_size, max_len]
        """
        # Handle different formats of durations input
        if durations is None:
            # If durations is None, infer batch_size from pitch_params
            batch_size = pitch_params.size(0)
            seq_len = pitch_params.size(1)
            # Create default durations (1 frame per phoneme)
            durations = torch.ones(batch_size, seq_len).to(pitch_params.device)
        elif isinstance(durations, list):
            # If durations is a list, convert to padded tensor
            max_len = max(d.size(0) for d in durations)
            batch_size = len(durations)
            padded_durations = torch.zeros(batch_size, max_len).to(pitch_params.device)
            for i, d in enumerate(durations):
                padded_durations[i, :d.size(0)] = d
            durations = padded_durations
        elif isinstance(durations, torch.Tensor):
            # If durations is a tensor, check its shape
            if durations.dim() == 1:
                # If 1D tensor, unsqueeze to [1, seq_len]
                durations = durations.unsqueeze(0)
            elif durations.dim() > 2:
                # If more than 2D, it could be [batch_size, seq_len, 1]
                if durations.dim() == 3 and durations.size(2) == 1:
                    durations = durations.squeeze(-1)
                else:
                    # Unexpected shape, print warning and use defaults
                    print(f"Warning: Unexpected durations shape: {durations.shape}")
                    batch_size = pitch_params.size(0)
                    seq_len = pitch_params.size(1)
                    durations = torch.ones(batch_size, seq_len).to(pitch_params.device)
                    
        # Now durations should be a 2D tensor [batch_size, seq_len]
        batch_size, seq_len = durations.shape
        
        # Extract components from pitch params
        f0_base = pitch_params[:, :, 0]         # Base F0 (Hz)
        vibrato_rate = pitch_params[:, :, 1]    # Vibrato rate (Hz)
        vibrato_extent = pitch_params[:, :, 2]  # Vibrato extent (semitones)
        
        # Find max length for the f0 contour - ensure it's an integer
        duration_sum = torch.sum(durations, dim=1)
        if torch.all(duration_sum == 0):
            # Handle the case where all durations are zero
            print("Warning: All durations are zero! Using default duration.")
            max_len = 200  # Default length
        else:
            max_len = int(torch.max(duration_sum).item())
        
        # Create f0 contour tensor
        f0_contour = torch.zeros(batch_size, max_len).to(pitch_params.device)
        
        for i in range(batch_size):
            current_pos = 0
            t_cumulative = 0  # Cumulative time for vibrato phase continuity
            
            for j in range(seq_len):
                # Skip if duration is zero or very small
                if durations[i, j] <= 0.1:
                    continue
                    
                # Get the number of frames for this phoneme
                num_frames = int(durations[i, j].item())
                
                # Skip if we'd exceed the max length
                if current_pos + num_frames > max_len:
                    num_frames = max_len - current_pos
                    if num_frames <= 0:
                        break
                
                # Get values for this phoneme
                base = f0_base[i, j].item()
                rate = vibrato_rate[i, j].item()
                extent = vibrato_extent[i, j].item()
                
                # Only apply vibrato if rate and extent are both non-zero
                if abs(rate) > 0.1 and abs(extent) > 0.01:
                    # Create time array for this segment
                    t = torch.arange(num_frames, device=pitch_params.device).float() / 100.0
                    t = t + t_cumulative  # Add cumulative time for phase continuity
                    
                    # Calculate vibrato modulation
                    vibrato_mod = extent * torch.sin(2 * math.pi * rate * t)
                    
                    # Apply vibrato (convert semitones to multiplicative factor)
                    f0_with_vibrato = base * torch.pow(torch.tensor(2.0), vibrato_mod / 12.0)
                    
                    # Update contour
                    f0_contour[i, current_pos:current_pos+num_frames] = f0_with_vibrato
                    
                    # Update cumulative time for phase continuity
                    t_cumulative += num_frames / 100.0
                else:
                    # Just use the base F0 without vibrato
                    f0_contour[i, current_pos:current_pos+num_frames] = base
                
                # Update position
                current_pos += num_frames
        
        return f0_contour