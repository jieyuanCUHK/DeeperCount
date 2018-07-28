# DeeperCount: implementation of UNet in Keras with image deformation and complex image augmentation.

Users can use DeeperCount and their labeled images to easily train a UNet.

---
## System requirements

The code of DeeperCount runs under Linux (i.e., Centos, https://www.centos.org/) on a 64-bit machine with at least two GPUs. It requires Python 2.7, [pip](https://bootstrap.pypa.io/get-pip.py) and several python packages including numpy, scipy, sklearn, matplotlib, cv2, [Keras](https://github.com/keras-team/keras), [Tensorflow](https://github.com/tensorflow/tensorflow) and [imgaug](https://github.com/aleju/imgaug). Before running DeeperCount code, these packages should be installed manually. To utilize GPUs, [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) should also be installed for NVIDIA GPUs.

We have tested these code in CentOS Linux release 7.3.1611, Keras 2.1.2, Tensorflow 1.4.0 and CUDA 8.0.61.

## Using DeeperCount

### Get DeeperCount
```
git clone https://github.com/jieyuanCUHK/DeeperCount.git
```

### Run DeeperCount

We recommend run DeeperCount code under an isolated python environment, e.g. a virtual python environment created using following commands via [virtualenv](https://virtualenv.pypa.io/en/stable/):

```
cd ./DeeperCount

# clean PYTHONPATH
export PYTHONPATH=

# Create an isolated python environment
virtualenv --no-site-packages env

# Activate the isolated python environment
source env/bin/activate
```

After all the required python packages are installed in the created virtual environment manually by the user using pip command, DeeperCount can be executed via:

```

```

## Please cite
