"""
PyTorch Lightning DataModule for working with Protein Language Model (PLM) activations.

This module provides a DataModule implementation that works with optimized 
protein language model activation data, supporting operations like normalization
and activation centering.
"""

from typing import Optional
import torch
import pytorch_lightning as pl
from litdata import StreamingDataset, StreamingDataLoader
from tqdm import tqdm

MODEL_PARAMS_TO_ESM_MODEL_NAME = {
    "8m": "esm2_t6_8M_UR50D",
    '3b': 'esm2_t36_3B_UR50D',
}

MODEL_TO_D_MODEL = {
    "8m": 320,
    '3b': 2560,
}

class LitData_PLM_Datamodule(pl.LightningDataModule):
    """
    DataModule for protein language model (PLM) activations.
    
    This class handles the loading and preprocessing of PLM activation data,
    including normalization and centering operations that are helpful for 
    training sparse autoencoders on these activations.
    
    Args:
        optimized_dir: Path to the directory containing optimized activation data
        batch_size: Batch size for the data loader
        num_workers: Number of workers for the data loader
        model: Model identifier (e.g., "8m" for ESM-2 8M parameter model)
        layer: Layer number to extract activations from (1-indexed)
        normalize_activations: Whether to normalize activations by a scaling factor
        norm_factor_avging_steps: Number of steps to use for calculating normalization factor
        center_activations: Whether to center activations by subtracting mean
        shuffle: Whether to shuffle the data during training
    """

    def __init__(
        self, 
        optimized_dir: str, 
        batch_size: int, 
        num_workers: int = 4, 
        model: str = "8m", #TODO: Add clarity on `model` and `layer` params
        layer: int = 6, 
        normalize_activations: bool = True, 
        norm_factor_avging_steps: int = 100, 
        center_activations: bool = True, 
        shuffle: bool = False,
        verbose: bool = False
    ):
        super().__init__()
        self.optimized_dir = optimized_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        assert model in MODEL_PARAMS_TO_ESM_MODEL_NAME, f"Model {model} not found in MODEL_PARAMS_TO_ESM_MODEL_NAME"

        self.plm_name = MODEL_PARAMS_TO_ESM_MODEL_NAME[model]
        self.layer = layer - 1  # Convert to 0-indexed
        self.d_model = MODEL_TO_D_MODEL[model]
        if self.d_model is None:
            raise ValueError(f"Model {model} does not have a defined dimension in MODEL_TO_D_MODEL")
        
        self.normalize_activations = normalize_activations
        self.norm_factor_avging_steps = norm_factor_avging_steps
        self.center_activations = center_activations
        self.shuffle = shuffle
        self.norm_factor = None

        self.verbose = verbose
        if self.verbose:
            print(f"DataModule initialized with shuffle={self.shuffle}")

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Set up the dataset and calculate normalization factor if needed.
        
        Args:
            stage: Optional stage (e.g., 'fit', 'validate')
        """
        self.dataset = StreamingDataset(self.optimized_dir, shuffle=False, drop_last=True)
        
        if self.normalize_activations:
            self.norm_factor = self.get_norm_factor(
                self.dataset, 
                self.norm_factor_avging_steps
            )

    def collate_fn(self, batch: list) -> torch.Tensor:
        """
        Collate function for the data loader that applies normalization and centering.
        
        Args:
            batch: List of tensors to collate
            
        Returns:
            Processed batch tensor
        """
        batch = torch.cat(batch, dim=0)

        if self.normalize_activations and self.norm_factor is not None:
            batch = batch / self.norm_factor

        if self.center_activations:
            d_model_mean = torch.mean(batch, dim=-1, keepdim=True)
            batch = batch - d_model_mean

        return batch

    def train_dataloader(self) -> StreamingDataLoader:
        """
        Create a data loader for training.
        
        Returns:
            StreamingDataLoader for training
        """
        return StreamingDataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )
    
    def get_norm_factor(self, data, steps: int) -> float:
        """
        Calculate a fixed scalar normalization factor for activation vectors.
        
        Per Section 3.1 in https://arxiv.org/pdf/2408.05147, we find a 
        fixed scalar factor so activation vectors have unit mean squared norm.
        This is very helpful for hyperparameter transfer between different 
        layers and models.
        
        Args:
            data: Dataset to calculate normalization factor from
            steps: Number of steps to average over (more steps = more accurate)
            
        Returns:
            Normalization factor (float)
            
        Notes:
            If experiencing troubles with hyperparameter transfer between models,
            it may be worth instead normalizing to the square root of d_model.
            See https://transformer-circuits.pub/2024/april-update/index.html#training-saes
        """
        if hasattr(self, 'norm_factor') and self.norm_factor is not None:
            return self.norm_factor

        total_mean_squared_norm = 0.0
        count = 0

        for step, act_BD in enumerate(tqdm(data, total=steps, desc="Calculating norm factor")):
            if step >= steps:
                break

            count += 1
            mean_squared_norm = torch.mean(torch.sum(act_BD ** 2, dim=1))
            total_mean_squared_norm += mean_squared_norm

        average_mean_squared_norm = total_mean_squared_norm / count
        norm_factor = torch.sqrt(average_mean_squared_norm).item()

        if self.verbose:
            print(f"Average mean squared norm: {average_mean_squared_norm:.4f}")
            print(f"Normalization factor: {norm_factor:.4f}")
        
        return norm_factor