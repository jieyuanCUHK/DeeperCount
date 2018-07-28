#0. download parameter needed for network training:
wget -P ./01_model_parameters http://sunlab.cpy.cuhk.edu.hk/DeeperCount/parameter.tar.gz
tar -xzvf ./01_model_parameters/parameter.tar.gz

#1. first split large images to 512*512 tif images:
python ./02_source_code_for_training/0_split_image.py image_format
#supported image format: jpg/png/tif
#splited images will be stored into ./03_image_directory/Split_train_image and ./03_image_directory/Split_label_image

#2. then conduct image deformation;
python ./02_source_code/1_image_deformation.py
#merged image and image after deformation will be stored in ./03_image_directory/Merged_after_deform

#3. image augmentation:
#(1) using implementation in Keras, simple augmentation:
python ./02_source_code_for_training/2_original_image_augmentation.py number_of_images_after_augmentation
#results will be stored in ./03_image_directory/After_augmentation_primary

#(2) or using imgaug, complexed image augmentation:
python ./02_source_code_for_training/3_enhanced_image_augmentation.py batch_num batch_size
#results will be stored in ./03_image_directory/After_augmentation_primary

#4. train the model, please change the number of GPU/number of Epoch/Batch size:
python ./02_source_code_for_training/5_training_DeeperCount.py network_parameter  
#need to specify the model parameter you want to use: unet_refined.hdf5|unet.hdf5

#5. do prediction:
#(1) process images that need to predict:
python ./04_source_code_for_predicting/0_get_ready_for_predicting.py image_format
#supported image format: jpg/png/tif
#the images that need to predict: are stored in ./03_image_directory/Predict_image

#(2) final image prediction:
python ./04_source_code_for_predicting/1_predicting_using_DeeperCount.py network_parameter
#need to specify the parameter you want to use: unet_user_new.hdf5|unet.hdf5|unet_refined.hdf5
#the predicted images are stored in ./03_image_directory/Predict_image in jpg format 