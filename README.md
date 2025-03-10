# Reticular SAE

Official repo for `Towards Interpretable Protein Structure Prediction with Sparse Autoencoders` accepted to ICLR 2025 workshops.

## Overview

This repository contains code for training and evaluating Sparse Autoencoders, with a focus on protein structure prediction tasks. The project includes components for:

- Training SAE models with configurable parameters
- Evaluating RMSD (Root Mean Square Deviation) for protein structure predictions
- Integration with ESMF (ESM Fold) predictions

## Project Structure

```
reticular-sae/
├── dictionary_learning/  # Submodule with SAE trainers from Marks et al. 2024
├── training/           # Training scripts and configurations
│   ├── train.py     # Main training script
│   └── configs/     # Training configurations for Matryoshka, TopK, etc.
└── eval/
    ├── cross_entropy_eval.py  # Cross entropy evaluation metrics
    └── rmsd_eval.py           # RMSD evaluation for protein structures
```

## Setup

1. Clone the repository
2. Install dependencies
3. Sign into AWS CLI to access ESM2 embeddings stored in public S3 bucket.

## Usage

### Training

To train a Matryoshka SAE model, the `test_training_command.sh` script will call `training/train.py` and should run out of the box.

### Evaluation

To evaluate RMSD for protein structure predictions, first generate and save PDB files using ESMFold and run `rmsd_eval.py`.

## Configuration

The project uses Hydra for configurations. Key configuration options include:

- `dict_size`: Size of the dictionary
- `expansion_factor`: Expansion factor for the model
- `k`: Number of active features
- `lr`: Learning rate
- `warmup_steps`: Number of warmup steps

## License

This project is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for details.
