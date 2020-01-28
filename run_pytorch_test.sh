#!/bin/bash
#
# This script runs the pytorch tests needed for the paper. It takes care of
# setting up and destroying the ramdisk used for the training data, using
# the needed CU masks, and setting the output names.
sudo true
if [ $? -ne 0 ];
then
	echo "sudo is needed for setting up a ramdisk temp FS."
	exit 1
fi

RESULT_DIR="./results"

EPOCHS=20

EPOCHS_WITH_COMPETITOR=3

TMP_DIR=`pwd`/temp_data
mkdir $TMP_DIR
sudo mount -t tmpfs -o size=1024M tmpfs $TMP_DIR

#python mnist_pytorch.py --seed 1337 --epochs $EPOCHS --time-output $RESULT_DIR/pytorch_times_gpu.json
#
#HSA_DEFAULT_CU_MASK=ffff python mnist_pytorch.py --seed 1337 --epochs $EPOCHS --time-output $RESULT_DIR/pytorch_times_gpu_16_CUs.json
#
#HSA_DEFAULT_CU_MASK=ff python mnist_pytorch.py --seed 1337 --epochs $EPOCHS --time-output $RESULT_DIR/pytorch_times_gpu_8_CUs.json
#
#HSA_DEFAULT_CU_MASK=f python mnist_pytorch.py --seed 1337 --epochs $EPOCHS --time-output $RESULT_DIR/pytorch_times_gpu_4_CUs.json
#
#HSA_DEFAULT_CU_MASK=3 python mnist_pytorch.py --seed 1337 --epochs $EPOCHS --time-output $RESULT_DIR/pytorch_times_gpu_2_CUs.json
#
#HSA_DEFAULT_CU_MASK=1 python mnist_pytorch.py --seed 1337 --epochs $EPOCHS --time-output $RESULT_DIR/pytorch_times_gpu_1_CU.json
#
#python mnist_pytorch.py --seed 1337 --epochs $EPOCHS --no-cuda --time-output $RESULT_DIR/pytorch_times_cpu.json

# Both competitor and pytorch share the full GPU (the competitor is configured
# to run indefinitely and not write any output).
python mnist_pytorch.py --seed 1337 --epochs 1000 >/dev/null &
COMPETITOR_PID=$!
echo "Giving the competitor time to start..."
sleep 20
echo "Competitor started with PID $COMPETITOR_PID"
python mnist_pytorch.py --seed 1337 --epochs $EPOCHS_WITH_COMPETITOR --time-output $RESULT_DIR/pytorch_times_gpu_full_shared.json
kill -9 $COMPETITOR_PID

# The competitor and pytorch are isolated on half of the GPU
HSA_DEFAULT_CU_MASK=ffff0000 python mnist_pytorch.py --seed 1337 --epochs 1000 >/dev/null &
COMPETITOR_PID=$!
echo "Giving the competitor time to start..."
sleep 20
echo "Competitor started with PID $COMPETITOR_PID"
HSA_DEFAULT_CU_MASK=ffff python mnist_pytorch.py --seed 1337 --epochs $EPOCHS_WITH_COMPETITOR --time-output $RESULT_DIR/pytorch_times_gpu_partitioned_16_CUs.json
kill -9 $COMPETITOR_PID

sudo umount $TMP_DIR
rmdir $TMP_DIR

