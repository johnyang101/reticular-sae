"""
Utility functions for training Sparse Autoencoders (SAEs) on Protein Language Model (PLM) activations.
"""

from typing import Optional
from dataclasses import dataclass
from omegaconf import OmegaConf, DictConfig


@dataclass
class TrainingParams:
    """Container for computed training parameters."""
    datamodule_batch_size: int
    steps_per_epoch: int
    total_steps: int
    norm_factor: Optional[float] = None


def validate_dict_size(activation_dim: int, expansion_factor: float, dict_size: int) -> None:
    """Validate that the dictionary size matches the expected size."""
    expected_size = int(activation_dim * expansion_factor)
    if dict_size != expected_size:
        raise ValueError(
            f"Dictionary size ({dict_size}) must equal activation_dim ({activation_dim}) "
            f"* expansion_factor ({expansion_factor}) = {expected_size}"
        )


def validate_batch_size(batch_size: int, toks_per_bucket: int) -> None:
    """Validate that the batch size is compatible with tokens per bucket."""
    if batch_size % toks_per_bucket != 0:
        raise ValueError(
            f"batch_size ({batch_size}) must be divisible by toks_per_bucket "
            f"({toks_per_bucket})"
        )


def calculate_training_params(cfg: DictConfig) -> TrainingParams:
    """Calculate training parameters based on configuration."""
    # Validate batch size
    validate_batch_size(cfg.batch_size, cfg.datamodule.toks_per_bucket)
    
    # Calculate effective batch size and training steps
    datamodule_batch_size = cfg.batch_size // cfg.datamodule.toks_per_bucket
    steps_per_epoch = cfg.datamodule.buckets_per_epoch / datamodule_batch_size
    total_steps = cfg.epochs * steps_per_epoch
    
    return TrainingParams(
        datamodule_batch_size=datamodule_batch_size,
        steps_per_epoch=steps_per_epoch,
        total_steps=total_steps
    )


def update_config_with_training_params(cfg: DictConfig, params: TrainingParams) -> DictConfig:
    """Update configuration with computed training parameters."""
    # Create computed parameters dict
    computed_params = {
        'computed': {
            'datamodule_batch_size': params.datamodule_batch_size,
            'steps_per_epoch': params.steps_per_epoch,
            'total_steps': params.total_steps,
        }
    }
    
    if params.norm_factor is not None:
        computed_params['computed']['norm_factor'] = params.norm_factor
    
    # Merge the computed parameters into the config
    cfg['computed'] = computed_params['computed']
    
    return cfg