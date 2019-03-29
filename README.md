# DeeperCount: implementation of U-Net-Mu model in Keras with image deformation and enhanced image augmentation.
![](https://github.com/jieyuanCUHK/DeeperCount/blob/master/Logo.jpg)
Users can use DeeperCount and their labeled images to easily train a U-Net.

---
## System requirements

The code of DeeperCount runs under Linux (i.e., Centos, https://www.centos.org/) on a 64-bit machine with at least two GPUs. It requires Python 2.7, [pip](https://bootstrap.pypa.io/get-pip.py) and several python packages including numpy, scipy, sklearn, matplotlib, cv2, [Keras](https://github.com/keras-team/keras), [Tensorflow](https://github.com/tensorflow/tensorflow) and [imgaug](https://github.com/aleju/imgaug). Before running DeeperCount code, these packages should be installed manually. To utilize GPUs, [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) and [cuDNN library](https://developer.nvidia.com/cudnn) should also be installed for NVIDIA GPUs.

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

#### 1. download parameter needed for network training:
```console
wget -P ./01_model_parameters http://sunlab.cpy.cuhk.edu.hk/DeeperCount/parameter.tar.gz
tar -xzvf ./01_model_parameters/parameter.tar.gz -C ./01_model_parameters
```

#### 2. split large images to 512*512 tif images:
```console
python ./02_source_code_for_training/0_split_image.py image_format
# images should be put into ./03_image_directory/Train_image and ./03_image_directory/Label_image, the corresponding image and label should have the same name.
# supported image format: jpg/png/tif
# splited images will be stored into ./03_image_directory/Split_train_image and ./03_image_directory/Split_label_image
```

#### 3. conduct image deformation;
```console
python ./02_source_code/1_image_deformation.py
# merged image (merging image and label into different channels of one image) and image after deformation will be stored in ./03_image_directory/Merged_after_deform. Images with "train" in name beginning are the deformed images.
```

#### 4. conduct image augmentation:

##### 4.1 using implementation in Keras, simple augmentation:
```console
python ./02_source_code_for_training/2_original_image_augmentation.py number_of_images_after_augmentation
# results will be stored in ./03_image_directory/After_augmentation_primary. "number_of_images_after_augmentation": the number of augmented image outputs for one input image. 
```

##### 4.2 or using imgaug, complexed image augmentation:
```console
python ./02_source_code_for_training/3_enhanced_image_augmentation.py batch_num batch_size
# results will be stored in ./03_image_directory/After_augmentation_primary. The number of augmented image will be batch_num*batch_size*(original image+deformed image)
```
#### 5. split the merged image, getting ready for training:
```console
python 02_source_code_for_training/4_get_ready_for_training.py 
# the channel-splited images will be stored in ./03_image_directory/Final_label and ./03_image_directory/Final_train.
```

#### 6. train the model, please change the number of GPU/number of epoch/batch size:
```console
python ./02_source_code_for_training/5_training_DeeperCount.py network_parameter  
# need to specify the model parameter you want to use: unet_refined.hdf5|unet.hdf5
# please pay attention that the unet_refined.hdf5 parameter was trained using multi-gpus, while unet.hdf5 was not. 
# please change the critical parameters including learning rate, GPU number, optimization algorithm, epoch number, batch size, etc. according to your experiment.
# the parameter after transfer learning is ./01_model_parameters/unet_user_new.hdf5.
```

#### 7. do prediction:

##### 7.1 process images that need to predict:
```console
python ./04_source_code_for_predicting/0_get_ready_for_predicting.py image_format
# supported image format: jpg/png/tif
# the images that need to predict: are stored in ./03_image_directory/Predict_image
```

##### 7.2 final image prediction:
```console
python ./04_source_code_for_predicting/1_predicting_using_DeeperCount.py network_parameter
# need to specify the parameter you want to use: unet_user_new.hdf5|unet.hdf5|unet_refined.hdf5
# the predicted images are stored in ./03_image_directory/Prediction_results in jpg format 
```
##### 7.3 further merge the predicted image to a whole image (if your image is not in 512*512):
```console
python 04_source_code_for_predicting/2_merge_small_images.py image_format
# the images will be in ./03_image_directory/Prediction_results/Merged.
```
## Please cite
