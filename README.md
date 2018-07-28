# DeeperCount: implementation of UNet in Keras with image deformation and complex image augmentation.

Users can use DeeperCount and their labeled images to easily train a UNet.

---
## System requirements

The code of DeeperCount runs under Linux (i.e., Centos, https://www.centos.org/) on a 64-bit machine with at least two GPUs. It requires Python 2.7, [pip](https://bootstrap.pypa.io/get-pip.py) and several python packages including numpy, scipy, sklearn, matplotlib, cv2, [Keras](https://github.com/keras-team/keras), [Tensorflow](https://github.com/tensorflow/tensorflow) and [imgaug](https://github.com/aleju/imgaug). Before running DeeperCount code, these packages should be installed manually. To utilize GPUs, [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) should also be installed for NVIDIA GPUs.

We have tested these code in CentOS Linux release 7.3.1611, Keras 2.1.2, Tensorflow 1.4.0 and CUDA 8.0.61.

## Using DeeperCount

### Get DeeperCount
```console
git clone https://github.com/jieyuanCUHK/DeeperCount.git
```

### Run DeeperCount

We recommend run DeeperCount code under an isolated python environment, e.g. a virtual python environment created using following commands via [virtualenv](https://virtualenv.pypa.io/en/stable/):

```console
cd ./DeeperCount

# clean PYTHONPATH
export PYTHONPATH=

# Create an isolated python environment
virtualenv --no-site-packages env

# Activate the isolated python environment
source env/bin/activate
```

After all the required python packages are installed in the created virtual environment manually by the user using pip command, DeeperCount can be executed following commands in run.sh:

* download parameter needed for network training:
```console
wget -P ./01_model_parameters http://sunlab.cpy.cuhk.edu.hk/DeeperCount/parameter.tar.gz
tar -xzvf ./01_model_parameters/parameter.tar.gz -C ./01_model_parameters
```

* first split large images to 512*512 tif images:
```console
python ./02_source_code_for_training/0_split_image.py image_format
# images should be put into ./03_image_directory/Train_image and ./03_image_directory/Label_image
# supported image format: jpg/png/tif
# splited images will be stored into ./03_image_directory/Split_train_image and ./03_image_directory/Split_label_image
```

* then conduct image deformation;
```console
python ./02_source_code/1_image_deformation.py
# merged image and image after deformation will be stored in ./03_image_directory/Merged_after_deform
```

* image augmentation:

  * using implementation in Keras, simple augmentation:
```console
python ./02_source_code_for_training/2_original_image_augmentation.py number_of_images_after_augmentation
# results will be stored in ./03_image_directory/After_augmentation_primary
```

  * or using imgaug, complexed image augmentation:
```console
python ./02_source_code_for_training/3_enhanced_image_augmentation.py batch_num batch_size
# results will be stored in ./03_image_directory/After_augmentation_primary
```

* train the model, please change the number of GPU/number of epoch/batch size:
```console
python ./02_source_code_for_training/5_training_DeeperCount.py network_parameter  
# need to specify the model parameter you want to use: unet_refined.hdf5|unet.hdf5
```

* do prediction:

  * process images that need to predict:
```console
python ./04_source_code_for_predicting/0_get_ready_for_predicting.py image_format
# supported image format: jpg/png/tif
# the images that need to predict: are stored in ./03_image_directory/Predict_image
```

  * final image prediction:
```console
python ./04_source_code_for_predicting/1_predicting_using_DeeperCount.py network_parameter
# need to specify the parameter you want to use: unet_user_new.hdf5|unet.hdf5|unet_refined.hdf5
# the predicted images are stored in ./03_image_directory/Predict_image in jpg format 
```

## Please cite
