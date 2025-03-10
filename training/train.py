"""
Script for training Sparse Autoencoders (SAEs) on Protein Language Model (PLM) activations.

This script provides a command-line interface for training SAEs using PyTorch Lightning,
with support for different training strategies, model architectures, and evaluation metrics.
"""

from typing import Dict, Any, Optional, List, Tuple

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from training.datamodule import LitData_PLM_Datamodule as PLMDatamodule
from training.trainer import (
    SAETrainerLM, 
    CustomModelCheckpointCallback_DL, 
    FidelityEvaluationCallback_DL, 
)
from training.utils import (
    validate_dict_size,
    calculate_training_params,
    update_config_with_training_params,
)

# Import dictionary learning models
from dictionary_learning.trainers.standard import StandardTrainer
from dictionary_learning.trainers.top_k import AutoEncoderTopK, TopKTrainer
from dictionary_learning.trainers.matryoshka_batch_top_k import MatryoshkaBatchTopKSAE, MatryoshkaBatchTopKTrainer
from dictionary_learning.dictionary import AutoEncoder

import hydra
from omegaconf import DictConfig


def setup_datamodule(cfg: DictConfig, training_params) -> Tuple[PLMDatamodule, DictConfig]:
    """
    Initialize and set up the data module.
    
    Args:
        cfg: Configuration object containing computed parameters
        training_params: TrainingParams object to update with norm factor
        
    Returns:
        Tuple of (configured PLMDatamodule instance, updated config)
    """
    datamodule = PLMDatamodule(
        optimized_dir=cfg.datamodule.optimized_dir,
        batch_size=cfg.computed.datamodule_batch_size,
        model=cfg.datamodule.model_type,
        layer=cfg.datamodule.layer_number,
        normalize_activations=cfg.datamodule.normalize_activations,
        norm_factor_avging_steps=cfg.datamodule.norm_factor_avging_steps,
        center_activations=cfg.datamodule.center_activations,
        shuffle=cfg.datamodule.shuffle,
    )
    
    datamodule.setup()
    
    if cfg.datamodule.normalize_activations:
        training_params.norm_factor = datamodule.norm_factor
        cfg = update_config_with_training_params(cfg, training_params)
        print(f"Normalization factor: {cfg.computed.norm_factor:.4f}")
    
    return datamodule, cfg


def initialize_model(cfg: DictConfig) -> SAETrainerLM:
    """
    Initialize the SAE model with the appropriate configuration.
    
    Args:
        cfg: Configuration object containing computed parameters
        
    Returns:
        Configured SAETrainerLM instance
    """
    sae_trainer_cfg = format_sae_trainer_cfg(cfg.sae_trainer_cfg)
    sae_trainer_cfg.update({"steps": cfg.computed.total_steps})
    
    return SAETrainerLM(sae_trainer_cfg=sae_trainer_cfg)


def create_callbacks(cfg: DictConfig, datamodule: PLMDatamodule) -> List[pl.Callback]:
    """
    Create a list of callbacks for the trainer.
    
    Args:
        cfg: Configuration object containing computed parameters
        datamodule: Data module instance
        
    Returns:
        List of callbacks
    """
    callbacks = [
        CustomModelCheckpointCallback_DL(
            dirpath=cfg.save_dir,
            filename='sae-step-{step:02d}.ckpt',
            every_n_train_steps=cfg.save_steps,
        ),
        LearningRateMonitor(logging_interval='step'),
    ]
    
    if hasattr(cfg, 'eval_seq_path') and cfg.eval_seq_path:
        callbacks.append(
            FidelityEvaluationCallback_DL(
                eval_seq_path=cfg.eval_seq_path,
                plm_name=datamodule.plm_name,
                layer_idx=cfg.datamodule.layer_number,
                normalize_activations=cfg.datamodule.normalize_activations,
                norm_factor=cfg.computed.norm_factor if hasattr(cfg.computed, 'norm_factor') else None,
                eval_batch_size=cfg.eval_batch_size,
                eval_steps=cfg.eval_steps,
                start_step=cfg.eval_start_step,
                verbose=True
            )
        )
    
    return callbacks


def setup_wandb_logger(cfg: DictConfig, datamodule: PLMDatamodule) -> Optional[WandbLogger]:
    """
    Set up the Weights & Biases logger if configured.
    
    Args:
        cfg: Configuration object containing computed parameters
        datamodule: Data module instance
        
    Returns:
        WandbLogger instance or None if wandb is disabled
    """
    if not cfg.use_wandb:
        return None
        
    logger = WandbLogger(
        name=cfg.wandb_name,
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        log_model=True
    )
    
    # Log hyperparameters
    hyperparams = {
        'layer': datamodule.layer,
        'plm_name': datamodule.plm_name,
        'norm_factor': cfg.computed.norm_factor if hasattr(cfg.computed, 'norm_factor') else None,
        'batch_size': cfg.batch_size,
        'save_dir': cfg.save_dir,
    }
    
    if hasattr(cfg, 'eval_seq_path') and cfg.eval_seq_path:
        hyperparams.update({
            'eval_seq_path': str(cfg.eval_seq_path),
            'eval_steps': cfg.eval_steps,
        })
        
    logger.log_hyperparams(hyperparams)
    return logger


def format_sae_trainer_cfg(sae_trainer_cfg: DictConfig) -> Dict[str, Any]:
    """
    Initialize trainer configuration with appropriate classes.
    
    Args:
        sae_trainer_cfg: Configuration from Hydra
        
    Returns:
        Dictionary with trainer configuration including class references
        
    Raises:
        ValueError: If the trainer class is unknown
    """
    trainer_dict = dict(sae_trainer_cfg)

    class_mappings = {
        "StandardTrainer": (StandardTrainer, AutoEncoder),
        "TopKTrainer": (TopKTrainer, AutoEncoderTopK),
        "MatryoshkaBatchTopKTrainer": (MatryoshkaBatchTopKTrainer, MatryoshkaBatchTopKSAE),
    }
    
    trainer_class = sae_trainer_cfg.trainer_class
    if trainer_class in class_mappings:
        trainer_dict.update({
            "trainer_class": class_mappings[trainer_class][0],
            "dict_class": class_mappings[trainer_class][1],
        })
    else:
        raise ValueError(f"Unknown trainer class: {trainer_class}")

    return trainer_dict


def create_trainer(cfg: DictConfig, callbacks: List[pl.Callback], logger: Optional[WandbLogger]) -> pl.Trainer:
    """
    Create a PyTorch Lightning trainer based on configuration.
    
    Args:
        cfg: Configuration object
        callbacks: List of callbacks
        logger: WandbLogger instance or None
        
    Returns:
        Configured PyTorch Lightning Trainer
        
    Raises:
        ValueError: If neither epochs nor steps is provided
    """
    common_kwargs = {
        'callbacks': callbacks,
        'logger': logger,
        'log_every_n_steps': cfg.log_steps,
        'enable_progress_bar': True,
        'enable_checkpointing': True,
        'fast_dev_run': cfg.fast_dev_run
    }
    
    if cfg.epochs:
        return pl.Trainer(max_epochs=cfg.epochs, **common_kwargs)
    elif cfg.steps:
        return pl.Trainer(max_steps=cfg.steps, **common_kwargs)
    else:
        raise ValueError("Either epochs or steps must be provided in config")

@hydra.main(version_base=None, config_path="configs", config_name="train_std")
def train_SAE_on_PLM_embeds(cfg: DictConfig) -> None:
    """
    Train a Sparse Autoencoder on Protein Language Model embeddings.
    
    Args:
        cfg: Configuration object from Hydra containing all training parameters
    """
    pl.seed_everything(cfg.seed)
    
    validate_dict_size(
        cfg.sae_trainer_cfg.activation_dim,
        cfg.expansion_factor,
        cfg.sae_trainer_cfg.dict_size
    )

    training_params = calculate_training_params(cfg)
    cfg = update_config_with_training_params(cfg, training_params)
    
    # Initialize and set up the data module, updating norm factor if needed
    datamodule, cfg = setup_datamodule(cfg, training_params)

    model = initialize_model(cfg)
    callbacks = create_callbacks(cfg, datamodule)
    logger = setup_wandb_logger(cfg, datamodule)
    trainer = create_trainer(cfg, callbacks, logger)
    
    trainer.fit(model, datamodule=datamodule)
    print(f"Training completed. Model saved to {cfg.save_dir}")


if __name__ == "__main__":
    train_SAE_on_PLM_embeds()