#!/bin/bash
python reticular-sae/training/train.py \
  -cn train_matry \
  datamodule=8M_L3_1k_seqs_opt \
  epochs=4 \
  sae_trainer_cfg.dict_size=2560 \
  expansion_factor=8 \
  sae_trainer_cfg.k=32 \
  sae_trainer_cfg.lr=1e-7 \
  wandb_name=test_command \
  eval_start_step=1 \
  eval_steps=1 \
  log_steps=1 \
  save_steps=1 \
  sae_trainer_cfg.warmup_steps=100