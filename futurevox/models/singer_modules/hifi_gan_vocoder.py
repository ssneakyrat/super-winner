import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResBlock(nn.Module):
    """
    Residual block with dilated convolutions for HiFi-GAN.
    """
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        
        self.convs = nn.ModuleList()
        
        for i, d in enumerate(dilation):
            self.convs.append(
                nn.Sequential(
                    nn.LeakyReLU(0.1),
                    nn.Conv1d(
                        channels, channels, kernel_size, 
                        dilation=d, padding=self._get_padding(kernel_size, d)
                    ),
                    nn.LeakyReLU(0.1),
                    nn.Conv1d(channels, channels, kernel_size=1)
                )
            )
        
    def _get_padding(self, kernel_size, dilation):
        return (kernel_size * dilation - dilation) // 2
        
    def forward(self, x):
        """
        Forward pass with residual connections.
        
        Args:
            x: Input tensor [B, C, T]
            
        Returns:
            output: Output tensor [B, C, T]
        """
        for conv in self.convs:
            residual = x
            x = conv(x) + residual
            
        return x


class MRF(nn.Module):
    """
    Multi-Receptive Field Fusion module.
    """
    def __init__(self, channels, kernel_sizes=(3, 7, 11), dilations=((1, 3, 5), (1, 3, 5), (1, 3, 5))):
        super().__init__()
        
        self.resblocks = nn.ModuleList()
        
        for k, d in zip(kernel_sizes, dilations):
            self.resblocks.append(ResBlock(channels, kernel_size=k, dilation=d))
            
    def forward(self, x):
        """
        Forward pass with sum fusion.
        
        Args:
            x: Input tensor [B, C, T]
            
        Returns:
            output: Output tensor [B, C, T]
        """
        output = None
        
        for resblock in self.resblocks:
            if output is None:
                output = resblock(x)
            else:
                output += resblock(x)
                
        return output / len(self.resblocks)


class Generator(nn.Module):
    """
    HiFi-GAN generator with multi-receptive field fusion.
    """
    def __init__(self, 
                 in_channels=80,
                 upsample_initial_channel=512,
                 upsample_rates=(8, 8, 2, 2),
                 upsample_kernel_sizes=(16, 16, 4, 4),
                 resblock_kernel_sizes=(3, 7, 11),
                 resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5))):
        super().__init__()
        
        self.num_upsamples = len(upsample_rates)
        
        # Initial convolution
        self.conv_pre = nn.Conv1d(in_channels, upsample_initial_channel, kernel_size=7, stride=1, padding=3)
        
        # Upsampling layers
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            # Calculate channels for this layer
            in_channels = upsample_initial_channel // (2 ** i) if i < self.num_upsamples else upsample_initial_channel
            out_channels = upsample_initial_channel // (2 ** (i + 1)) if i < self.num_upsamples - 1 else upsample_initial_channel // (2 ** i)
            
            self.ups.append(
                nn.ConvTranspose1d(
                    in_channels, out_channels,
                    kernel_size=k, stride=u,
                    padding=(k - u) // 2
                )
            )
        
        # Multi-Receptive Field Fusion layers
        self.mrfs = nn.ModuleList()
        for i in range(self.num_upsamples):
            channels = upsample_initial_channel // (2 ** (i + 1)) if i < self.num_upsamples - 1 else upsample_initial_channel // (2 ** i)
            self.mrfs.append(MRF(channels, resblock_kernel_sizes, resblock_dilation_sizes))
        
        # Final convolution
        self.conv_post = nn.Conv1d(
            upsample_initial_channel // (2 ** (self.num_upsamples - 1)) if self.num_upsamples > 0 else upsample_initial_channel,
            1, kernel_size=7, stride=1, padding=3
        )
        
        # Harmonic-plus-noise model enhancement
        self.use_hnm = True
        if self.use_hnm:
            self.harmonic_conv = nn.Conv1d(1, 1, kernel_size=7, stride=1, padding=3)
            self.noise_conv = nn.Conv1d(1, 1, kernel_size=31, stride=1, padding=15)
            self.harmonic_gate = nn.Conv1d(1, 1, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x, f0=None):
        """
        Forward pass through the generator.
        
        Args:
            x: Input mel spectrogram [B, in_channels, T]
            f0: Optional F0 contour for harmonic enhancement [B, T]
            
        Returns:
            output: Generated waveform [B, 1, T*prod(upsample_rates)]
        """
        # Initial convolution
        x = self.conv_pre(x)
        
        # Apply upsampling and MRFs
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, 0.1)
            x = self.ups[i](x)
            x = self.mrfs[i](x)
        
        # Final convolution
        x = F.leaky_relu(x, 0.1)
        x = self.conv_post(x)
        x = torch.tanh(x)
        
        # Apply harmonic-plus-noise model if enabled and F0 is provided
        if self.use_hnm and f0 is not None:
            # Upsample F0 to match audio dimensions
            target_length = x.size(-1)
            f0_upsampled = F.interpolate(f0.unsqueeze(1), size=target_length, mode='linear', align_corners=False)
            
            # Generate harmonic and noise components
            harmonic = self.harmonic_conv(x)
            noise = self.noise_conv(x)
            
            # Create harmonic-noise gate based on F0 presence
            f0_gate = torch.sigmoid(self.harmonic_gate(f0_upsampled))
            
            # Mix harmonic and noise components
            x = harmonic * f0_gate + noise * (1 - f0_gate)
        
        return x


class PeriodDiscriminator(nn.Module):
    """
    Period-based discriminator for HiFi-GAN.
    """
    def __init__(self, period, kernel_size=5, stride=3, channels=32, downsample_rates=(2, 2, 2)):
        super().__init__()
        
        self.period = period
        norm_f = weight_norm
        
        # Initial convolution
        self.convs = nn.ModuleList([
            norm_f(nn.Conv2d(1, channels, (kernel_size, 1), (stride, 1), padding=(kernel_size//2, 0))),
            norm_f(nn.Conv2d(channels, channels*2, (kernel_size, 1), (stride, 1), padding=(kernel_size//2, 0))),
            norm_f(nn.Conv2d(channels*2, channels*4, (kernel_size, 1), (stride, 1), padding=(kernel_size//2, 0))),
            norm_f(nn.Conv2d(channels*4, channels*4, (kernel_size, 1), 1, padding=(kernel_size//2, 0))),
            norm_f(nn.Conv2d(channels*4, channels*2, (kernel_size, 1), 1, padding=(kernel_size//2, 0))),
        ])
        
        # Final convolution
        self.conv_post = norm_f(nn.Conv2d(channels*2, 1, (3, 1), 1, padding=(1, 0)))
        
    def forward(self, x):
        """
        Forward pass through the period discriminator.
        
        Args:
            x: Input waveform [B, 1, T]
            
        Returns:
            output: Discrimination outputs
            fmap: Feature maps for feature matching loss
        """
        batch_size, _, length = x.shape
        
        # Handle edge case where audio length is not divisible by period
        if length % self.period != 0:
            pad_length = self.period - (length % self.period)
            x = F.pad(x, (0, pad_length))
            length += pad_length
            
        # Reshape input for 2D convolution based on period
        x = x.view(batch_size, 1, length // self.period, self.period)
        
        fmap = []
        
        # Apply convolutions and collect feature maps
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
            
        # Final convolution
        x = self.conv_post(x)
        fmap.append(x)
        
        # Flatten output
        x = torch.flatten(x, 1, -1)
        
        return x, fmap


class MultiPeriodDiscriminator(nn.Module):
    """
    Multi-period discriminator with different periods.
    """
    def __init__(self, periods=(2, 3, 5, 7, 11)):
        super().__init__()
        
        self.discriminators = nn.ModuleList()
        for period in periods:
            self.discriminators.append(PeriodDiscriminator(period))
            
    def forward(self, x):
        """
        Forward pass through all period discriminators.
        
        Args:
            x: Input waveform [B, 1, T]
            
        Returns:
            outputs: List of discrimination outputs
            fmaps: List of feature maps
        """
        outputs = []
        fmaps = []
        
        for disc in self.discriminators:
            output, fmap = disc(x)
            outputs.append(output)
            fmaps.append(fmap)
            
        return outputs, fmaps


class ScaleDiscriminator(nn.Module):
    """
    Scale-based discriminator for HiFi-GAN.
    """
    def __init__(self, channels=32, norm=None):
        super().__init__()
        
        norm_f = weight_norm if norm is None else norm
        
        # Initial convolution
        self.convs = nn.ModuleList([
            norm_f(nn.Conv1d(1, channels, 15, 1, padding=7)),
            norm_f(nn.Conv1d(channels, channels*2, 41, 4, groups=4, padding=20)),
            norm_f(nn.Conv1d(channels*2, channels*4, 41, 4, groups=16, padding=20)),
            norm_f(nn.Conv1d(channels*4, channels*8, 41, 4, groups=16, padding=20)),
            norm_f(nn.Conv1d(channels*8, channels*16, 41, 4, groups=16, padding=20)),
            norm_f(nn.Conv1d(channels*16, channels*16, 5, 1, padding=2)),
        ])
        
        # Final convolution
        self.conv_post = norm_f(nn.Conv1d(channels*16, 1, 3, 1, padding=1))
        
    def forward(self, x):
        """
        Forward pass through the scale discriminator.
        
        Args:
            x: Input waveform [B, 1, T]
            
        Returns:
            output: Discrimination output
            fmap: Feature maps for feature matching loss
        """
        fmap = []
        
        # Apply convolutions and collect feature maps
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
            
        # Final convolution
        x = self.conv_post(x)
        fmap.append(x)
        
        # Flatten output
        x = torch.flatten(x, 1, -1)
        
        return x, fmap


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-scale discriminator with different scales.
    """
    def __init__(self, scales=3):
        super().__init__()
        
        self.discriminators = nn.ModuleList()
        for _ in range(scales):
            self.discriminators.append(ScaleDiscriminator())
            
        self.downsample = nn.AvgPool1d(4, stride=2, padding=1)
        
    def forward(self, x):
        """
        Forward pass through all scale discriminators.
        
        Args:
            x: Input waveform [B, 1, T]
            
        Returns:
            outputs: List of discrimination outputs
            fmaps: List of feature maps
        """
        outputs = []
        fmaps = []
        
        for i, disc in enumerate(self.discriminators):
            if i > 0:
                x = self.downsample(x)
            output, fmap = disc(x)
            outputs.append(output)
            fmaps.append(fmap)
            
        return outputs, fmaps


# For weight normalization
def weight_norm(module):
    return nn.utils.weight_norm(module)


class HiFiGANVocoder(nn.Module):
    """
    Complete HiFi-GAN vocoder with generator and discriminators.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Model configuration
        mel_channels = config['audio']['n_mels']
        upsample_rates = config.get('vocoder', {}).get('upsample_rates', [8, 8, 2, 2])
        upsample_kernel_sizes = config.get('vocoder', {}).get('upsample_kernel_sizes', [16, 16, 4, 4])
        upsample_initial_channel = config.get('vocoder', {}).get('upsample_initial_channel', 512)
        resblock_kernel_sizes = config.get('vocoder', {}).get('resblock_kernel_sizes', [3, 7, 11])
        resblock_dilation_sizes = config.get('vocoder', {}).get('resblock_dilation_sizes', 
                                          [[1, 3, 5], [1, 3, 5], [1, 3, 5]])
        
        # Create generator
        self.generator = Generator(
            in_channels=mel_channels,
            upsample_initial_channel=upsample_initial_channel,
            upsample_rates=upsample_rates,
            upsample_kernel_sizes=upsample_kernel_sizes,
            resblock_kernel_sizes=resblock_kernel_sizes,
            resblock_dilation_sizes=resblock_dilation_sizes
        )
        
        # Create discriminators
        self.mpd = MultiPeriodDiscriminator()
        self.msd = MultiScaleDiscriminator()
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        """Initialize module weights."""
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Conv2d)):
            nn.init.kaiming_normal_(m.weight, a=0.1)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
                
    def forward_generator(self, mel_spectrogram, f0=None):
        """
        Forward pass through generator.
        
        Args:
            mel_spectrogram: Input mel spectrogram [B, n_mels, T]
            f0: Optional F0 contour for harmonic enhancement [B, T]
            
        Returns:
            waveform: Generated waveform [B, 1, T*prod(upsample_rates)]
        """
        return self.generator(mel_spectrogram, f0)
    
    def forward_discriminator(self, waveform):
        """
        Forward pass through discriminators.
        
        Args:
            waveform: Input waveform [B, 1, T]
            
        Returns:
            mpd_outputs: Outputs from multi-period discriminator
            mpd_fmaps: Feature maps from multi-period discriminator
            msd_outputs: Outputs from multi-scale discriminator
            msd_fmaps: Feature maps from multi-scale discriminator
        """
        mpd_outputs, mpd_fmaps = self.mpd(waveform)
        msd_outputs, msd_fmaps = self.msd(waveform)
        
        return mpd_outputs, mpd_fmaps, msd_outputs, msd_fmaps
    
    def forward(self, mel_spectrogram, f0=None):
        """
        Forward pass through the vocoder (generator only for inference).
        
        Args:
            mel_spectrogram: Input mel spectrogram [B, n_mels, T]
            f0: Optional F0 contour for harmonic enhancement [B, T]
            
        Returns:
            waveform: Generated waveform [B, 1, T*prod(upsample_rates)]
        """
        return self.forward_generator(mel_spectrogram, f0)
    
    def calculate_generator_loss(self, waveform_pred, waveform_target, mel_target=None):
        """
        Calculate generator loss.
        
        Args:
            waveform_pred: Predicted waveform [B, 1, T]
            waveform_target: Target waveform [B, 1, T]
            mel_target: Target mel spectrogram [B, n_mels, T]
            
        Returns:
            gen_loss: Generator loss
            loss_dict: Dictionary with loss components
        """
        # Run discriminators on predicted waveform
        mpd_pred_outputs, mpd_pred_fmaps, msd_pred_outputs, msd_pred_fmaps = self.forward_discriminator(waveform_pred)
        
        # Run discriminators on target waveform (for feature matching)
        _, mpd_target_fmaps, _, msd_target_fmaps = self.forward_discriminator(waveform_target)
        
        # 1. Adversarial loss
        mpd_adv_loss = 0
        for output in mpd_pred_outputs:
            mpd_adv_loss += torch.mean((1 - output) ** 2)
            
        msd_adv_loss = 0
        for output in msd_pred_outputs:
            msd_adv_loss += torch.mean((1 - output) ** 2)
            
        adv_loss = mpd_adv_loss + msd_adv_loss
        
        # 2. Feature matching loss
        mpd_fm_loss = 0
        for pred_fmap, target_fmap in zip(mpd_pred_fmaps, mpd_target_fmaps):
            for p_fmap, t_fmap in zip(pred_fmap, target_fmap):
                mpd_fm_loss += F.l1_loss(p_fmap, t_fmap.detach())
                
        msd_fm_loss = 0
        for pred_fmap, target_fmap in zip(msd_pred_fmaps, msd_target_fmaps):
            for p_fmap, t_fmap in zip(pred_fmap, target_fmap):
                msd_fm_loss += F.l1_loss(p_fmap, t_fmap.detach())
                
        fm_loss = mpd_fm_loss + msd_fm_loss
        
        # 3. Mel spectrogram loss (optional)
        mel_loss = 0
        if mel_target is not None:
            # Calculate mel spectrogram from predicted waveform
            from librosa.filters import mel as librosa_mel
            import librosa
            
            n_fft = self.config['audio']['n_fft']
            hop_length = self.config['audio']['hop_length']
            sample_rate = self.config['audio']['sample_rate']
            n_mels = self.config['audio']['n_mels']
            fmin = self.config['audio']['fmin']
            fmax = self.config['audio']['fmax']
            
            # Create mel filter bank
            mel_basis = librosa_mel(
                sr=sample_rate,
                n_fft=n_fft,
                n_mels=n_mels,
                fmin=fmin,
                fmax=fmax
            )
            mel_basis = torch.from_numpy(mel_basis).to(waveform_pred.device)
            
            # Compute spectrogram
            pred_spec = torch.stft(
                waveform_pred.squeeze(1),
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=n_fft,
                window=torch.hann_window(n_fft).to(waveform_pred.device),
                return_complex=True
            )
            
            # Convert to power spectrogram
            pred_spec = torch.abs(pred_spec) ** 2
            
            # Convert to mel scale
            pred_mel = torch.matmul(mel_basis, pred_spec.transpose(1, 2))
            
            # Convert to log scale
            pred_mel = torch.log10(torch.clamp(pred_mel, min=1e-5))
            
            # Calculate L1 loss
            mel_loss = F.l1_loss(pred_mel, mel_target)
        
        # 4. Weighted sum of losses
        gen_loss = adv_loss + 2 * fm_loss + 45 * mel_loss
        
        # Create loss dictionary
        loss_dict = {
            'gen_loss': gen_loss.item(),
            'mpd_adv_loss': mpd_adv_loss.item(),
            'msd_adv_loss': msd_adv_loss.item(),
            'fm_loss': fm_loss.item(),
            'mel_loss': mel_loss.item()
        }
        
        return gen_loss, loss_dict
        
    def calculate_discriminator_loss(self, waveform_pred, waveform_target):
        """
        Calculate discriminator loss.
        
        Args:
            waveform_pred: Predicted waveform [B, 1, T]
            waveform_target: Target waveform [B, 1, T]
            
        Returns:
            disc_loss: Discriminator loss
            loss_dict: Dictionary with loss components
        """
        # Run discriminators on predicted waveform
        mpd_pred_outputs, _, msd_pred_outputs, _ = self.forward_discriminator(waveform_pred.detach())
        
        # Run discriminators on target waveform
        mpd_target_outputs, _, msd_target_outputs, _ = self.forward_discriminator(waveform_target)
        
        # Multi-period discriminator loss
        mpd_loss = 0
        for pred, target in zip(mpd_pred_outputs, mpd_target_outputs):
            mpd_loss += torch.mean((1 - target) ** 2) + torch.mean(pred ** 2)
            
        # Multi-scale discriminator loss
        msd_loss = 0
        for pred, target in zip(msd_pred_outputs, msd_target_outputs):
            msd_loss += torch.mean((1 - target) ** 2) + torch.mean(pred ** 2)
            
        # Total discriminator loss
        disc_loss = mpd_loss + msd_loss
        
        # Create loss dictionary
        loss_dict = {
            'disc_loss': disc_loss.item(),
            'mpd_loss': mpd_loss.item(),
            'msd_loss': msd_loss.item()
        }
        
        return disc_loss, loss_dict
    
    def inference(self, mel_spectrogram, f0=None):
        """
        Generate waveform from mel spectrogram.
        
        Args:
            mel_spectrogram: Input mel spectrogram [B, n_mels, T]
            f0: Optional F0 contour for harmonic enhancement [B, T]
            
        Returns:
            waveform: Generated waveform [B, 1, T*prod(upsample_rates)]
        """
        self.eval()
        with torch.no_grad():
            waveform = self.forward_generator(mel_spectrogram, f0)
        return waveform
    
    def visualize_vocoder_output(self, waveform_pred, waveform_target=None):
        """
        Create visualizations for vocoder output.
        
        Args:
            waveform_pred: Predicted waveform [B, 1, T]
            waveform_target: Target waveform [B, 1, T]
            
        Returns:
            vis_dict: Dictionary with visualization tensors
        """
        # Placeholder for visualization tensors
        vis_dict = {}
        
        # Implementation would create visualization tensors for:
        # - Waveform plot (pred vs target)
        # - Spectrogram comparison
        # - Harmonics analysis
        
        return vis_dict