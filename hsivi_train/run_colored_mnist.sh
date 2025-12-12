#!/bin/bash
# HSIVI Training Script for Colored MNIST
# Adapted from https://github.com/longinYu/HSIVI

export CUDA_VISIBLE_DEVICES="2"

# Default configuration
WORKDIR="./work_dir/hsivi_5_lr_8e-5_8e-5"
DATA_DIR="./data"
PRETRAINED_MODEL="./ckpts/model-5.pt"  # Your pretrained diffusion model

# Training parameters
N_TRAIN_ITERS=100000
BATCH_SIZE=256
N_DISCRETE_STEPS=6  # NFE + 1 = 10 + 1

SAVE_EVERY=10
SAMPLE_EVERY=10
LOG_EVERY=5
NUM_SAMPLES=16

FID_EVERY=10
FID_NUM_SAMPLES=1000
SAMPLING_BATCH_SIZE=$BATCH_SIZE

PHI_LR=0.00008
F_LR=0.00008
F_LEARNING_TIMES=20
SKIP_TYPE="quad"

# Model parameters
PHI_BASE_DIM=64
F_BASE_DIM=64

echo "============================================="
echo "HSIVI Training for Colored MNIST"
echo "============================================="
echo "Working directory: ${WORKDIR}"
echo "Data directory: ${DATA_DIR}"
echo "Pretrained model: ${PRETRAINED_MODEL}"
echo "Number of function evaluations: $((N_DISCRETE_STEPS - 1))"
echo "============================================="

# Run training
python -m hsivi_train.train \
    --config default \
    --data_dir ${DATA_DIR} \
    --workdir ${WORKDIR} \
    --pretrained_model ${PRETRAINED_MODEL} \
    --n_train_iters ${N_TRAIN_ITERS} \
    --training_batch_size ${BATCH_SIZE} \
    --n_discrete_steps ${N_DISCRETE_STEPS} \
    --save_every ${SAVE_EVERY} \
    --sample_every ${SAMPLE_EVERY} \
    --log_every ${LOG_EVERY} \
    --num_samples ${NUM_SAMPLES} \
    --fid_every ${FID_EVERY} \
    --fid_num_samples ${FID_NUM_SAMPLES} \
    --sampling_batch_size ${SAMPLING_BATCH_SIZE} \
    --phi_learning_rate ${PHI_LR} \
    --f_learning_rate ${F_LR} \
    --f_learning_times ${F_LEARNING_TIMES} \
    --phi_base_dim ${PHI_BASE_DIM} \
    --f_base_dim ${F_BASE_DIM} \
    --skip_type ${SKIP_TYPE} \
    --image_gamma \
    --seed 42

