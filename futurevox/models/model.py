import torch
import torch.nn as nn
import torch.nn.functional as F

from models.singer_modules.phoneme_encoder import EnhancedPhonemeEncoder
from models.singer_modules.variance_adaptor import AdvancedVarianceAdaptor
from models.singer_modules.acoustic_decoder import AcousticDecoder
from models.singer_modules.hifi_gan_vocoder import HiFiGANVocoder

class FutureVoxSinger(nn.Module):
    """
    FutureVox-Singer: Advanced singing voice synthesis model.
    """
    def __init__(self, config, num_phonemes=100, num_notes=128, num_singers=10):
        super().__init__()
        self.config = config
        
        # 1. Enhanced Phoneme Encoder
        self.phoneme_encoder = EnhancedPhonemeEncoder(
            config, 
            num_phonemes=num_phonemes,
            num_notes=num_notes
        )
        
        # 2. Advanced Variance Adaptor
        variance_config = config.copy()
        variance_config['variance_adaptor'] = variance_config.get('variance_adaptor', {})
        variance_config['variance_adaptor']['num_singers'] = num_singers
        
        self.variance_adaptor = AdvancedVarianceAdaptor(variance_config)
        
        # 3. Acoustic Decoder
        self.acoustic_decoder = AcousticDecoder(config)
        
        # 4. HiFi-GAN Vocoder
        self.vocoder = HiFiGANVocoder(config)
        
        # Track whether vocoder is trainable
        self.train_vocoder = config.get('training', {}).get('train_vocoder', False)
        
    def forward(self, batch, phase='all'):
        """
        Forward pass through the complete model.
        
        Args:
            batch: Dictionary containing input tensors
            phase: Training phase ('phoneme_encoder', 'variance_adaptor', 
                                'acoustic_decoder', 'vocoder', or 'all')
            
        Returns:
            output_dict: Dictionary containing model outputs
        """
        # Initialize output dictionary
        output_dict = {}
        
        # Get input tensors
        phone_indices = batch['phone_indices']
        phone_masks = batch['phone_masks'] if 'phone_masks' in batch else None
        note_indices = batch.get('note_indices', None)
        rhythm_info = batch.get('rhythm_info', None)
        singer_ids = batch.get('singer_ids', None)
        mel_spectrograms = batch.get('mel_spectrograms', None)
        waveform = batch.get('audio', None)
        
        # Always get ground truth data for training if available
        ground_truth = None
        if 'f0_values' in batch and 'durations' in batch:
            ground_truth = {
                'f0_values': batch['f0_values'],
                'durations': batch['durations'],
                'energy': batch.get('energy', None),
                'mel_spectrograms': mel_spectrograms,
                'waveform': waveform
            }
        
        # 1. Phoneme Encoder
        if phase in ['phoneme_encoder', 'variance_adaptor', 'acoustic_decoder', 'vocoder', 'all']:
            encoded_phonemes, attn_weights = self.phoneme_encoder(
                phone_indices, note_indices, phone_masks
            )
            output_dict['encoded_phonemes'] = encoded_phonemes
            output_dict['phoneme_attention_weights'] = attn_weights
        
        # Exit if we're only training the phoneme encoder
        if phase == 'phoneme_encoder':
            return output_dict
        
        # 2. Variance Adaptor (only run if not in phoneme_encoder phase)
        if phase in ['variance_adaptor', 'acoustic_decoder', 'vocoder', 'all']:
            variance_outputs = self.variance_adaptor(
                encoded_phonemes, phone_masks, note_indices, rhythm_info,
                singer_ids, ground_truth=ground_truth
            )
            output_dict.update(variance_outputs)
        
        # Exit if we're only training up to variance adaptor
        if phase == 'variance_adaptor':
            return output_dict
    
    def inference(self, phone_indices, note_indices=None, rhythm_info=None, 
                 singer_id=None, tempo_factor=1.0, ref_mel=None):
        """
        Generate singing voice from input phonemes.
        
        Args:
            phone_indices: Phoneme indices [batch_size, seq_len]
            note_indices: Note indices [batch_size, seq_len]
            rhythm_info: Rhythm information [batch_size, seq_len, channels]
            singer_id: Singer identity index [batch_size]
            tempo_factor: Factor to adjust the tempo (1.0 = normal)
            ref_mel: Reference mel spectrogram for style transfer [batch_size, n_mels, time]
            
        Returns:
            output_dict: Dictionary containing model outputs
        """
        self.eval()
        output_dict = {}
        
        with torch.no_grad():
            # 1. Phoneme Encoder
            encoded_phonemes, _ = self.phoneme_encoder(phone_indices, note_indices)
            
            # 2. Variance Adaptor
            variance_outputs = self.variance_adaptor(
                encoded_phonemes, None, note_indices, rhythm_info,
                singer_id, tempo_factor=tempo_factor
            )
            
            # 3. Acoustic Decoder
            acoustic_outputs = self.acoustic_decoder(
                variance_outputs['expanded_features'],
                variance_outputs['mel_masks'],
                ref_mels=ref_mel,
                f0=variance_outputs['f0_contour'],
                energy=variance_outputs['energy']
            )
            
            # 4. Vocoder
            waveform = self.vocoder.inference(
                acoustic_outputs['mel_postnet'],
                f0=variance_outputs.get('f0_contour', None)
            )
            
            # Combine outputs
            output_dict.update(variance_outputs)
            output_dict.update(acoustic_outputs)
            output_dict['waveform'] = waveform
            
        return output_dict
    
    def calculate_losses(self, output_dict, batch):
        """
        Calculate all losses for training.
        
        Args:
            output_dict: Dictionary containing model outputs
            batch: Dictionary containing input tensors and targets
            
        Returns:
            loss_dict: Dictionary containing all losses
            total_loss: Weighted sum of all losses
        """
        loss_dict = {}
        
        # Get ground truth data
        mel_spectrograms = batch.get('mel_spectrograms', None)
        f0_values = batch.get('f0_values', None)
        durations = batch.get('durations', None)
        energy = batch.get('energy', None)
        waveform = batch.get('audio', None)
        
        # 1. Phoneme encoder losses - no direct supervision
        
        # 2. Variance adaptor losses
        if 'pitch_params' in output_dict and f0_values is not None:
            # F0 prediction loss
            f0_loss = F.l1_loss(output_dict['f0_contour'], f0_values)
            loss_dict['f0_loss'] = f0_loss
            
            # Duration prediction loss
            if 'log_durations' in output_dict and durations is not None:
                duration_loss = F.mse_loss(output_dict['durations'], durations)
                loss_dict['duration_loss'] = duration_loss
            
            # Energy prediction loss
            if 'energy' in output_dict and energy is not None:
                energy_loss = F.l1_loss(output_dict['energy'].squeeze(-1), energy)
                loss_dict['energy_loss'] = energy_loss
        
        # 3. Acoustic decoder losses
        if 'mel_output' in output_dict and mel_spectrograms is not None:
            # Mel spectrogram reconstruction loss
            mel_loss = F.l1_loss(output_dict['mel_output'], mel_spectrograms)
            mel_postnet_loss = F.l1_loss(output_dict['mel_postnet'], mel_spectrograms)
            
            loss_dict['mel_loss'] = mel_loss
            loss_dict['mel_postnet_loss'] = mel_postnet_loss
        
        # 4. Vocoder losses (already calculated in forward pass)
        if 'vocoder_gen_loss' in output_dict:
            loss_dict['vocoder_gen_loss'] = output_dict['vocoder_gen_loss']
            
        if 'vocoder_disc_loss' in output_dict:
            loss_dict['vocoder_disc_loss'] = output_dict['vocoder_disc_loss']
        
        # Calculate weighted total loss
        # Weights could be configured in the config file
        weights = {
            'f0_loss': 1.0,
            'duration_loss': 1.0,
            'energy_loss': 1.0,
            'mel_loss': 1.0,
            'mel_postnet_loss': 1.0,
            'vocoder_gen_loss': 1.0,
            'vocoder_disc_loss': 1.0
        }
        
        # Initialize with a zero tensor that has requires_grad=True
        # Get device from any parameter in the model
        device = next(self.parameters()).device
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # Add all valid tensor losses
        for name, loss in loss_dict.items():
            if isinstance(loss, torch.Tensor):
                total_loss = total_loss + loss * weights.get(name, 1.0)
        
        loss_dict['total_loss'] = total_loss
        
        return loss_dict, total_loss
    
    def visualize_outputs(self, output_dict, batch):
        """
        Create visualizations for model outputs.
        
        Args:
            output_dict: Dictionary containing model outputs
            batch: Dictionary containing input tensors and targets
            
        Returns:
            vis_dict: Dictionary with visualization tensors
        """
        # Initialize visualization dictionary
        vis_dict = {}
        
        # Phoneme encoder visualizations
        if 'encoded_phonemes' in output_dict and 'phoneme_attention_weights' in output_dict:
            # Implement attention visualization
            pass
        
        # Variance adaptor visualizations
        if 'f0_contour' in output_dict:
            # Visualize F0 predictions vs ground truth
            if 'f0_values' in batch:
                pass
                
            # Visualize duration predictions vs ground truth
            if 'durations' in batch and 'log_durations' in output_dict:
                pass
        
        # Acoustic decoder visualizations
        if 'mel_output' in output_dict and 'mel_spectrograms' in batch:
            # Visualize mel spectrograms
            pass
        
        # Vocoder visualizations
        if 'waveform_pred' in output_dict and 'audio' in batch:
            # Visualize waveforms
            pass
        
        return vis_dict