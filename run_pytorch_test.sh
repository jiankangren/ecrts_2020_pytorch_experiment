#!/bin/bash
#
# This script runs the pytorch tests needed for the paper.

RESULT_DIR="./results"
EPOCHS=40

python mnist_pytorch.py --seed 1337 --epochs $EPOCHS --time-output $RESULT_DIR/pytorch_times_gpu.json

python mnist_pytorch.py --stream-cu-mask ffff --seed 1337 --epochs $EPOCHS --time-output $RESULT_DIR/pytorch_times_gpu_16_CUs.json

python mnist_pytorch.py --stream-cu-mask ff --seed 1337 --epochs $EPOCHS --time-output $RESULT_DIR/pytorch_times_gpu_8_CUs.json

python mnist_pytorch.py --stream-cu-mask f --seed 1337 --epochs $EPOCHS --time-output $RESULT_DIR/pytorch_times_gpu_4_CUs.json

python mnist_pytorch.py --stream-cu-mask 3 --seed 1337 --epochs $EPOCHS --time-output $RESULT_DIR/pytorch_times_gpu_2_CUs.json

python mnist_pytorch.py --stream-cu-mask 1 --seed 1337 --epochs $EPOCHS --time-output $RESULT_DIR/pytorch_times_gpu_1_CU.json

python mnist_pytorch.py --no-cuda --seed 1337 --epochs $EPOCHS --time-output $RESULT_DIR/pytorch_times_cpu.json

python mnist_pytorch.py --seed 1337 --epochs $EPOCHS --enable-competitor --time-output $RESULT_DIR/pytorch_times_gpu_full_shared.json

python mnist_pytorch.py --stream-cu-mask ffff --seed 1337 --epochs $EPOCHS --enable-competitor --competitor-cu-mask ffff0000 --time-output $RESULT_DIR/pytorch_times_gpu_partitioned_16_CUs.json


