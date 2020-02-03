Scripts for evaluating PyTorch's MNIST Example on AMD GPUs
==========================================================

This small repository contains a script for running a pre-trained version of
PyTorch's MNIST example, and plotting results.

About the script
----------------

The main script in this repository is `mnist_pytorch.py`. It (obviously)
requires that PyTorch is installed. TorchVision is also required for obtaining
the dataset. `mnist_pytorch.py` was derived from PyTorch's
[MNIST example](https://github.com/pytorch/examples/blob/master/mnist/main.py).
After running the example once in order to save the pretrained network in
`mnist_cnn.pt`, the script was heavily modified to perform inference only, and
to save iteration times in JSON files.

More importantly, the script was modified to add an optional competitor thread,
and to set compute-unit masks when requested (it will still work on non-AMD
GPUs or CPUs, so long as a compute-unit mask isn't set). The competitor thread,
when enabled, runs a network similar to the MNIST network but with a larger
number of layers and larger layer sizes.  It processes random data with the
same "shape" as the MNIST data.

Run the script with the `--help` command-line option for more information, or
`run_pytorch_test.sh` for specific invocation examples of how to run it.

Installing and Modifying PyTorch
--------------------------------

Follow these instructions, after installing all of the needed ROCm
prerequisites, to set up PyTorch on a system with an AMD GPU.

The modified version of PyTorch can be set up as follows (this assumes you're
using a `conda` environment, as recommended by the official PyTorch
instructions):

 1. Install the PyTorch prerequisites:
    `conda install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi`

 2. Obtain the PyTorch source code:
    `git clone --recursive https://github.com/pytorch/pytorch`

 3. Run the script to convert PyTorch code to HIP:
    ```
    cd pytorch
    python tools/amd_build/build_amd.py
    ```

 4. Apply the patch contained in this repository to enable the `set_cu_mask`
    method on PyTorch's `Stream` objects:
    ```
    cd pytorch
    git apply path/to/this/repo/enable_pytorch_cu_mask.patch
    ```

 5. (This may not be necessary, but I had to do it to get PyTorch to compile on
    my system.) Fix some code that doesn't want to compile on AMD GPUs:
      1. Open `caffe2/operators/hip/relu_op.hip`
      2. Find the line `__floats2half2_rn(xx.x > 0.0f ? xx.x : 0.0f, xx.y > 0.0f ? xx.y : 0.0f);`
      3. Add a `(float)` cast before every `xx.x` and `xx.y` in that line.
      4. Find the line `__floats2half2_rn(yy.x > 0.0f ? dy.x : 0.0f, yy.y > 0.0f ? dy.y : 0.0f);`
      5. Add a `(float)` cast before `yy.x`, `dy.x`, `yy.y` and `dy.y`.
      6. Save the changes.

 6. Build and install PyTorch (this may take several hours--if it fails make
    sure that you have all of the necessary ROCm packages installed):
    ```
    export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
    python setup.py install
    ```

 7. Assuming installation succeeded, install `torchvision` from source as well
    (you can't install it using `conda` without installing `conda`'s version of
    PyTorch, which would overwrite our version of PyTorch):
    ```
    cd <some/new/directory>
    git clone https://github.com/pytorch/vision
    cd vision
    export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
    python setup.py install
    ```

