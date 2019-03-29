#0. download parameter needed for network training:
wget -P ./01_model_parameters http://sunlab.cpy.cuhk.edu.hk/DeeperCount/parameter.tar.gz
tar -xzvf ./01_model_parameters/parameter.tar.gz -C ./01_model_parameters

#1. first split large images to 512*512 tif images:
python ./02_source_code_for_training/0_split_image.py image_format
#images should be put into ./03_image_directory/Train_image and ./03_image_directory/Label_image,  the corresponding image and label should have the same name.
#supported image format: jpg/png/tif
#splited images will be stored into ./03_image_directory/Split_train_image and ./03_image_directory/Split_label_image

#2. then conduct image deformation;
python ./02_source_code/1_image_deformation.py
#merged image (merging image and label into different channels of one image) and image after deformation will be stored in ./03_image_directory/Merged_after_deform. Images with "train" in name beginning are the deformed images.

#3. image augmentation:
#(1) using implementation in Keras, simple augmentation:
python ./02_source_code_for_training/2_original_image_augmentation.py number_of_images_after_augmentation
#results will be stored in ./03_image_directory/After_augmentation_primary.  "number_of_images_after_augmentation": the number of augmented image outputs for one input image. 

#(2) or using imgaug, complexed image augmentation:
python ./02_source_code_for_training/3_enhanced_image_augmentation.py batch_num batch_size
#results will be stored in ./03_image_directory/After_augmentation_primary. The number of augmented image will be batch_num*batch_size*(original image+deformed image).

#4. split the merged image, getting ready for training:
python 02_source_code_for_training/4_get_ready_for_training.py 
# the channel-splited images will be stored in ./03_image_directory/Final_label and ./03_image_directory/Final_train.

#5. train the model, please change the number of GPU/number of epoch/batch size:
python ./02_source_code_for_training/5_training_DeeperCount.py network_parameter  
#need to specify the model parameter you want to use: unet_refined.hdf5|unet.hdf5
#please pay attention that the unet_refined.hdf5 parameter was trained using multi-gpus, while unet.hdf5 was not. 

#6. do prediction:
#(1) process images that need to predict:
python ./04_source_code_for_predicting/0_get_ready_for_predicting.py image_format
#supported image format: jpg/png/tif
#the images that need to predict: are stored in ./03_image_directory/Predict_image

#(2) final image prediction:
python ./04_source_code_for_predicting/1_predicting_using_DeeperCount.py network_parameter
#need to specify the parameter you want to use: unet_user_new.hdf5|unet.hdf5|unet_refined.hdf5
#the predicted images are stored in ./03_image_directory/Predict_image in jpg format 
