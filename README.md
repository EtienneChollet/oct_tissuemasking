# oct_tissuemasking
This package contains a basic 3D UNet and patching scripts to create training data.

# 1 Installation

## 1.1 Create a new mamba environment

Create a new mamba environment called oct_tissuemasking with python 3.9.

```
mamba create -n oct_tissuemasking python=3.9
mamba activate oct_tissuemasking
```

## 1.2 Install dependencies

We will need to install synthspline for vasculature synthesis.

```
pip install git+https://github.com/balbasty/synthspline.git#f78ba23
```

## 1.3 Set cuda parameters

We need to identify and set our cuda version to make sure we install the right prebuilt wheel for cupy.

```
export CUDA_VERSION=<cuda-version>
```

## 1.4 Install oct_tissuemasking

This will take a while so we will set the timeout for a 20,000 seconds!

```
pip install oct_tissuemasking --default-timeout=20000
```