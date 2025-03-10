"""
PyTorch Lightning modules and callbacks for training sparse autoencoders.

This module provides Lightning modules for training various types of 
sparse autoencoders on protein language model (PLM) activations, as well as
callbacks for model checkpointing, fidelity evaluation, and FLOP tracking.
"""

import os
import torch
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import pytorch_lightning as pl

from eval.cross_entropy_eval import get_loss_recovery_fn

class SAETrainerLM(pl.LightningModule):
    """
    Lightning module for Sparse Autoencoder training.
    
    This class wraps SAE trainers from the dictionary_learning package
    to enable training with PyTorch Lightning.
    
    Args:
        sae_trainer_cfg: Configuration dictionary for the SAE trainer
        logging_steps: Frequency of logging training statistics (in steps)
    """

    def __init__(
        self,
        sae_trainer_cfg: Dict[str, Any],
        logging_steps: int = 100,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False  # We handle optimization manually
        self.logging_steps = logging_steps
        self.sae_trainer_cfg = sae_trainer_cfg

        # Initialize the SAE trainer
        sae_trainer_class = self.sae_trainer_cfg.get("trainer_class")
        if not sae_trainer_class:
            raise ValueError("Trainer class not provided in sae_trainer_cfg")
            
        trainer_cfg = dict(self.sae_trainer_cfg)
        del trainer_cfg["trainer_class"]
        self.sae_trainer = sae_trainer_class(**trainer_cfg)
        self.ae = self.sae_trainer.ae

        # Counter for tracking steps (used for logging and scheduling)
        self.lm_step_counter = 0 
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the autoencoder.
        
        Args:
            x: Input tensor of shape [batch_size, d_model]
            
        Returns:
            Reconstructed tensor of same shape
        """
        return self.sae_trainer.ae(x)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        """
        Perform a single training step.
        
        Args:
            batch: Input batch of activations
            batch_idx: Index of the current batch
        """
        self.sae_trainer.update(self.lm_step_counter, batch)
        self.lm_step_counter += 1
    
    def on_train_batch_end(self, outputs: Optional[Any], batch: torch.Tensor, batch_idx: int) -> None:
        """
        Log statistics at the end of each training batch.
        
        Args:
            outputs: Outputs from the training step (unused)
            batch: The input batch
            batch_idx: Index of the current batch
        """
        super().on_train_batch_end(outputs, batch, batch_idx)
        if self.lm_step_counter % self.logging_steps == 0:
            log = self.log_stats(batch)
            for key, value in log.items():
                self.log(f'train/{key}', value, on_step=True, on_epoch=False)
        

    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler._LRScheduler]]:
        """
        Configure optimizers and learning rate schedulers.
        
        Returns:
            Tuple of (optimizers, schedulers)
        """
        return [self.sae_trainer.optimizer], [self.sae_trainer.scheduler]
    
    def log_stats(self, batch: torch.Tensor) -> Dict[str, float]:
        """
        Compute and return training statistics.
        
        Args:
            batch: Current batch of data
            
        Returns:
            Dictionary of statistics to log
        """
        log = {}
        act, act_hat, f, losslog = self.sae_trainer.loss(batch, step=self.lm_step_counter, logging=True)

        # Calculate L0 sparsity (average number of non-zero features)
        l0 = (f != 0).float().sum(dim=-1).mean().item()
        
        # Calculate fraction of variance explained
        total_variance = torch.var(act, dim=0).sum()
        residual_variance = torch.var(act - act_hat, dim=0).sum()
        frac_variance_explained = 1 - residual_variance / total_variance
        log["frac_variance_explained"] = frac_variance_explained.item()
        
        # Log loss components
        log.update({
            f"{k}": v.cpu().item() if isinstance(v, torch.Tensor) else v 
            for k, v in losslog.items()
        })
        log["l0"] = l0
        
        # Get additional logging parameters from trainer
        trainer_log = self.sae_trainer.get_logging_parameters()
        for name, value in trainer_log.items():
            if isinstance(value, torch.Tensor):
                value = value.cpu().item()
            log[f"{name}"] = value

        return log
    
class CustomModelCheckpointCallback_DL(pl.Callback):
    """
    Custom model checkpointing callback that uses pl_module's lm_step_counter.
    
    This callback saves model checkpoints at specified intervals, using the
    model's internal step counter instead of the trainer's global step.
    
    Args:
        dirpath: Directory path to save the checkpoints
        filename: Filename format for the checkpoint (e.g., 'model-{step}.ckpt')
        every_n_train_steps: Save a checkpoint every n training steps
    """
    
    def __init__(self, dirpath: str, filename: str, every_n_train_steps: int):
        super().__init__()
        self.dirpath = dirpath
        self.filename = filename
        self.every_n_train_steps = every_n_train_steps

        # Ensure the save directory exists
        os.makedirs(self.dirpath, exist_ok=True)

    def on_train_batch_end(
        self, 
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule, 
        outputs: Any, 
        batch: Any, 
        batch_idx: int
    ) -> None:
        """
        Save checkpoint at the end of each batch if conditions are met.
        
        Args:
            trainer: PyTorch Lightning trainer
            pl_module: PyTorch Lightning module
            outputs: Outputs from the training step
            batch: The input batch
            batch_idx: Index of the current batch
        """
        # Use the model's internal step counter
        step = pl_module.lm_step_counter

        # Save checkpoint if it's time
        if step > 0 and step % self.every_n_train_steps == 0:
            checkpoint_path = os.path.join(self.dirpath, self.filename.format(step=step))
            trainer.save_checkpoint(checkpoint_path)
            print(f"Checkpoint saved at step {step}: {checkpoint_path}")


class FidelityEvaluationCallback_DL(pl.Callback):
    """
    Callback to evaluate model fidelity using loss recovery.
    
    This callback evaluates the fidelity of the sparse autoencoder at specified
    intervals by measuring how well it recovers the PLM's predictions.
    
    Args:
        eval_seq_path: Path to evaluation sequence file
        plm_name: Name of the protein language model
        layer_idx: Index of the layer to evaluate
        normalize_activations: Whether activations are normalized
        norm_factor: Normalization factor (required if normalize_activations=True)
        eval_batch_size: Batch size for evaluation
        eval_steps: Evaluate every n steps
        start_step: Start evaluation after this many steps
        verbose: Whether to print evaluation results
    """
    
    def __init__(
        self,
        eval_seq_path: Path,
        plm_name: str,
        layer_idx: int,
        normalize_activations: bool,
        norm_factor: Optional[float] = None,
        eval_batch_size: int = 128,
        eval_steps: int = 1000,
        start_step: int = 0,
        verbose: bool = True,
    ):
        super().__init__()
        self.plm_name = plm_name
        self.layer_idx = layer_idx
        self.normalize_activations = normalize_activations
        self.norm_factor = norm_factor
        self.eval_seq_path = eval_seq_path
        self.eval_batch_size = eval_batch_size
        self.eval_steps = eval_steps
        self.start_step = start_step
        self.verbose = verbose
        self.loss_recovery_fn = None
    
    def on_train_start(
        self, 
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """
        Initialize fidelity evaluation function at the start of training.
        
        Args:
            trainer: PyTorch Lightning trainer
            pl_module: PyTorch Lightning module
        """
        self.device = pl_module.device

        print(f"Initializing fidelity evaluation (every {self.eval_steps} steps)")

        self.loss_recovery_fn = get_loss_recovery_fn(
            esm_model_name=self.plm_name,
            layer_idx=self.layer_idx,
            eval_seq_path=self.eval_seq_path,
            device=self.device,
            batch_size=self.eval_batch_size,
        )
    
    def on_train_batch_end(
        self, 
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """
        Run fidelity evaluation at specified intervals.
        
        Args:
            trainer: PyTorch Lightning trainer
            pl_module: PyTorch Lightning module
            outputs: Outputs from the training step
            batch: The input batch
            batch_idx: Index of the current batch
        """
        # Check if it's time to run evaluation
        step = pl_module.lm_step_counter
        if (step >= self.start_step and step % self.eval_steps == 0):
            
            if self.verbose:
                print(f"\nRunning fidelity evaluation at step {step}")
            
            with torch.no_grad():
                # Scale biases if necessary for evaluation
                if self.normalize_activations:
                    assert self.norm_factor is not None, "norm_factor must be provided if normalize_activations is True"
                    pl_module.ae.scale_biases(self.norm_factor)

                # Run evaluation
                recovery_metrics = self.loss_recovery_fn(pl_module.ae)

                # Scale biases back if necessary
                if self.normalize_activations:
                    pl_module.ae.scale_biases(1 / self.norm_factor)
                
                # Log metrics
                pl_module.log('val/pct_loss_recovered', recovery_metrics['pct_loss_recovered'])
                pl_module.log('val/CE_w_sae_patching', recovery_metrics['CE_w_sae_patching'])
                
                if self.verbose:
                    print(f"Loss recovered: {recovery_metrics['pct_loss_recovered']:.2%}")
                    print(f"CE with SAE patching: {recovery_metrics['CE_w_sae_patching']:.4f}")