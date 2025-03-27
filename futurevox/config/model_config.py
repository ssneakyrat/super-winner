"""
Configuration handler for FutureVox.
Loads and manages hyperparameters from YAML files.
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional


@dataclass
class TextEncoderConfig:
    n_layers: int = 4
    n_heads: int = 4
    hidden_dim: int = 256
    dropout: float = 0.1


@dataclass
class PredictorConfig:
    kernel_size: int = 3
    dropout: float = 0.5


@dataclass
class PredictorsConfig:
    duration_predictor: PredictorConfig = field(default_factory=PredictorConfig)
    f0_predictor: PredictorConfig = field(default_factory=PredictorConfig)


@dataclass
class FlowDecoderConfig:
    n_flows: int = 4
    hidden_dim: int = 256
    kernel_size: int = 5
    dilation_rate: int = 1


@dataclass
class VocoderConfig:
    upsample_rates: List[int] = field(default_factory=lambda: [8, 8, 2, 2])
    upsample_kernel_sizes: List[int] = field(default_factory=lambda: [16, 16, 4, 4])
    resblock_kernel_sizes: List[int] = field(default_factory=lambda: [3, 7, 11])
    resblock_dilation_sizes: List[List[int]] = field(
        default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    )


@dataclass
class DiscriminatorConfig:
    periods: List[int] = field(default_factory=lambda: [2, 3, 5, 7, 11])


@dataclass
class ModelConfig:
    text_encoder: TextEncoderConfig = field(default_factory=TextEncoderConfig)
    predictors: PredictorsConfig = field(default_factory=PredictorsConfig)
    flow_decoder: FlowDecoderConfig = field(default_factory=FlowDecoderConfig)
    vocoder: VocoderConfig = field(default_factory=VocoderConfig)
    discriminator: DiscriminatorConfig = field(default_factory=DiscriminatorConfig)


@dataclass
class DataConfig:
    # Audio processing
    sample_rate: int = 22050
    mel_channels: int = 80
    fft_size: int = 1024
    hop_size: int = 256
    win_size: int = 1024
    fmin: int = 0
    fmax: int = 8000
    max_wav_length: int = 8192
    
    # Dataset paths
    datasets_root: str = "./datasets"
    phoneme_cache_path: str = "./phoneme_cache"
    
    # Dataset splits
    train_file: str = "train_filelist.txt"
    val_file: str = "val_filelist.txt"
    test_file: str = "test_filelist.txt"
    
    # Phoneme settings
    phoneme_dict_file: str = "phoneme_dict.json"


@dataclass
class OptimizerConfig:
    name: str = "AdamW"
    weight_decay: float = 0.01


@dataclass
class SchedulerConfig:
    name: str = "ExponentialLR"
    gamma: float = 0.999


@dataclass
class LossWeightsConfig:
    mel: float = 45.0
    duration: float = 1.0
    f0: float = 1.0
    gan: float = 1.0
    fm: float = 2.0


@dataclass
class TrainingConfig:
    batch_size: int = 16
    learning_rate: float = 0.0002
    betas: List[float] = field(default_factory=lambda: [0.8, 0.99])
    lr_decay: float = 0.999
    seed: int = 1234
    epochs: int = 1000
    save_every: int = 5000
    eval_every: int = 1000
    precision: int = 16  # For mixed precision training
    clip_grad_norm: float = 1.0
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    loss_weights: LossWeightsConfig = field(default_factory=LossWeightsConfig)


@dataclass
class LoggingConfig:
    log_dir: str = "./logs"
    audio_samples: int = 4  # Number of audio samples to log per validation


@dataclass
class FutureVoxConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, config_path: str) -> "FutureVoxConfig":
        """
        Load configuration from a YAML file.
        
        Args:
            config_path: Path to the YAML configuration file
            
        Returns:
            FutureVoxConfig: Loaded configuration
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
            
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "FutureVoxConfig":
        """
        Create a configuration from a dictionary.
        
        Args:
            config_dict: Dictionary containing configuration
            
        Returns:
            FutureVoxConfig: Created configuration
        """
        data_config = DataConfig(**config_dict.get("data", {}))
        
        model_dict = config_dict.get("model", {})
        text_encoder = TextEncoderConfig(**model_dict.get("text_encoder", {}))
        
        predictors_dict = model_dict.get("predictors", {})
        duration_predictor = PredictorConfig(**predictors_dict.get("duration_predictor", {}))
        f0_predictor = PredictorConfig(**predictors_dict.get("f0_predictor", {}))
        predictors = PredictorsConfig(
            duration_predictor=duration_predictor,
            f0_predictor=f0_predictor
        )
        
        flow_decoder = FlowDecoderConfig(**model_dict.get("flow_decoder", {}))
        vocoder = VocoderConfig(**model_dict.get("vocoder", {}))
        discriminator = DiscriminatorConfig(**model_dict.get("discriminator", {}))
        
        model_config = ModelConfig(
            text_encoder=text_encoder,
            predictors=predictors,
            flow_decoder=flow_decoder,
            vocoder=vocoder,
            discriminator=discriminator
        )
        
        training_dict = config_dict.get("training", {})
        optimizer = OptimizerConfig(**training_dict.get("optimizer", {}))
        scheduler = SchedulerConfig(**training_dict.get("scheduler", {}))
        loss_weights = LossWeightsConfig(**training_dict.get("loss_weights", {}))
        
        training_config = TrainingConfig(
            **{k: v for k, v in training_dict.items() 
               if k not in ["optimizer", "scheduler", "loss_weights"]},
            optimizer=optimizer,
            scheduler=scheduler,
            loss_weights=loss_weights
        )
        
        logging_config = LoggingConfig(**config_dict.get("logging", {}))
        
        return cls(
            data=data_config,
            model=model_config,
            training=training_config,
            logging=logging_config
        )
    
    def save(self, config_path: str) -> None:
        """
        Save configuration to a YAML file.
        
        Args:
            config_path: Path to save the configuration
        """
        # Convert dataclass to dictionary
        config_dict = self.to_dict()
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
        
        # Save to YAML file
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the configuration
        """
        # This is a simplified implementation
        # A more complete implementation would recursively convert nested dataclasses
        return {
            "data": vars(self.data),
            "model": {
                "text_encoder": vars(self.model.text_encoder),
                "predictors": {
                    "duration_predictor": vars(self.model.predictors.duration_predictor),
                    "f0_predictor": vars(self.model.predictors.f0_predictor)
                },
                "flow_decoder": vars(self.model.flow_decoder),
                "vocoder": vars(self.model.vocoder),
                "discriminator": vars(self.model.discriminator)
            },
            "training": {
                **{k: v for k, v in vars(self.training).items() 
                   if k not in ["optimizer", "scheduler", "loss_weights"]},
                "optimizer": vars(self.training.optimizer),
                "scheduler": vars(self.training.scheduler),
                "loss_weights": vars(self.training.loss_weights)
            },
            "logging": vars(self.logging)
        }