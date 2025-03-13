# SAEFold by Reticular (YC F24)

## Introduction

SAEFold enables mechanistic interpretability on ESMFold, a protein structure prediction model, for the first time using sparse autoencoders (SAEs) trained on ESM2-3B. 

For more details, see our paper published at the ICLR 2025 GEM workshop linked below. All code and pretrained models are provided under the Apache 2.0 license, making them freely available for both academic and commercial use.

📄 **Paper**: [ArXiv Link](https://arxiv.org/abs/2503.08764) \
🔍 **Interactive Visualizations**: [sae.reticular.ai](https://sae.reticular.ai) \
🏢 **More about Reticular**: [reticular.ai](https://reticular.ai)

## Overview

This repository contains code for training and evaluating SAEs, with a focus on protein structure prediction tasks. The project includes components for:

- Training your own protein SAEs out-of-the-box on embeddings from 100K sequences hosted on our public S3 bucket.
- Evaluating SAEFold's protein structure prediction recovery on CASP14.

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

## Installation

1. Clone the repository.

2. Install dependencies.
   ```bash
   # First, install dependencies for the dictionary learning submodule
   cd dictionary_learning
   pip install -r requirements.txt
   
   # Then, install the main package
   cd ../
   pip install -e .
   ```

3. Sign into AWS CLI to access ESM2 embeddings stored in public S3 bucket.
   ```bash
   aws configure
   ```

4. (Optional) Initialize Weights and Biases (WandB) \

    Run
    ```bash
    wandb init
    ```
    and set `use_wandb` to True in `training/configs/train_default.yaml`

## Usage

### Training

To train a Matryoshka SAE model, the `test_training_command.sh` script will call `training/train.py` and should run out of the box.

### Load Pretrained Models
We provide pretrained Matryoshka SAE models in the `pretrained_models` directory. These can be loaded using the following code:
```python
from dictionary_learning.trainers.matryoshka_batch_top_k import MatryoshkaBatchTopKSAE
pretrained_model = MatryoshkaBatchTopKSAE.from_pretrained(ckpt_path)
#Checkpoints must be unnormalized to use for downstream tasks.
pretrained_model.scale_biases(norm_factor)
```

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

## Citation 📚

If you use SAEFold in your research, please cite our paper:

```bibtex
@inproceedings{parsan2025interpretableproteinstructureprediction,
    title     = {Towards Interpretable Protein Structure Prediction with Sparse Autoencoders},
    author    = {Parsan, Nithin and Yang, David J. and Yang, John J.},
    booktitle = {ICLR 2025 Workshop on Generative and Experimental Perspectives for Biomolecular Design (GEM)},
    year      = {2025},
    eprint    = {2503.08764},
    archivePrefix = {arXiv},
    primaryClass = {q-bio.BM},
    url       = {https://arxiv.org/abs/2503.08764}
}
```
