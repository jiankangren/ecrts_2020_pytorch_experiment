#!/bin/bash
#
# This script runs the pytorch tests needed for the paper.

RESULT_DIR="./results"
EPOCHS=30
EPOCHS_WITH_COMPETITOR=6

# Both competitor and pytorch share the full GPU (the competitor is configured
# to run indefinitely and not write any output).
#python mnist_pytorch.py --seed 1337 --epochs 1000 >/dev/null &
#../../hip_plugin_framework/bin/runner ./plugin_framework_competitor_config.json >/dev/null
#COMPETITOR_PID=$!
#echo "Giving the competitor time to start..."
#sleep 4
#echo "Competitor started with PID $COMPETITOR_PID"
#python mnist_pytorch.py --seed 1337 --epochs $EPOCHS_WITH_COMPETITOR --time-output $RESULT_DIR/pytorch_times_gpu_full_shared.json
#kill -9 $COMPETITOR_PID

# The competitor and pytorch are isolated on half of the GPU
#../../hip_plugin_framework/bin/runner ./plugin_framework_competitor_config.json >/dev/null
#COMPETITOR_PID=$!
#echo "Giving the competitor time to start..."
#sleep 4
#echo "Competitor started with PID $COMPETITOR_PID"
#HSA_DEFAULT_CU_MASK=ffff python mnist_pytorch.py --seed 1337 --epochs $EPOCHS_WITH_COMPETITOR --time-output $RESULT_DIR/pytorch_times_gpu_partitioned_16_CUs.json
#kill -9 $COMPETITOR_PID

python mnist_pytorch.py --seed 1337 --epochs $EPOCHS --time-output $RESULT_DIR/pytorch_times_gpu.json

HSA_DEFAULT_CU_MASK=ffff python mnist_pytorch.py --seed 1337 --epochs $EPOCHS --time-output $RESULT_DIR/pytorch_times_gpu_16_CUs.json

HSA_DEFAULT_CU_MASK=ff python mnist_pytorch.py --seed 1337 --epochs $EPOCHS --time-output $RESULT_DIR/pytorch_times_gpu_8_CUs.json

HSA_DEFAULT_CU_MASK=f python mnist_pytorch.py --seed 1337 --epochs $EPOCHS --time-output $RESULT_DIR/pytorch_times_gpu_4_CUs.json

HSA_DEFAULT_CU_MASK=3 python mnist_pytorch.py --seed 1337 --epochs $EPOCHS --time-output $RESULT_DIR/pytorch_times_gpu_2_CUs.json

HSA_DEFAULT_CU_MASK=1 python mnist_pytorch.py --seed 1337 --epochs $EPOCHS --time-output $RESULT_DIR/pytorch_times_gpu_1_CU.json

python mnist_pytorch.py --seed 1337 --epochs $EPOCHS --no-cuda --time-output $RESULT_DIR/pytorch_times_cpu.json

