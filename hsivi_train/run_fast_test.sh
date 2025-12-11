#!/bin/bash
# Fast test training script for HSIVI on Colored MNIST
# Use this to quickly verify the setup works

WORKDIR="./work_dir/hsivi_test"
DATA_DIR="./data"
PRETRAINED_MODEL="./ckpts/model-5.pt"

echo "============================================="
echo "HSIVI Fast Test Training"
echo "============================================="

python -m hsivi_train.train \
    --config fast \
    --data_dir ${DATA_DIR} \
    --workdir ${WORKDIR} \
    --pretrained_model ${PRETRAINED_MODEL} \
    --n_train_iters 1000 \
    --training_batch_size 32 \
    --n_discrete_steps 6 \
    --seed 42

echo "============================================="
echo "Fast test complete!"
echo "Check samples in ${WORKDIR}/samples/"
echo "============================================="

