import os 
import sys
#coding=utf-8  
  
os.environ["CUDA_VISIBLE_DEVICES"] = "1"	#default: using one single GPU
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils import multi_gpu_model
import numpy as np
from keras.models import *
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras import backend as keras
from keras import backend as K
import tensorflow as tf

class dataProcess(object):
	def __init__(self, out_rows, out_cols, test_path = "./03_image_directory/Predict_image", npy_path = "./03_image_directory/Predict_image/", img_type = "tif"):

		self.out_rows = out_rows
		self.out_cols = out_cols
		self.img_type = img_type
		self.test_path = test_path
		self.npy_path = npy_path

	def load_test_data(self):
		print('Load images')
		print('-'*30)
		imgs_test = np.load(self.npy_path+"/imgs_test.npy")
		imgs_test = imgs_test.astype('float32')
		imgs_test /= 255
		#mean = imgs_test.mean(axis = 0)
		#imgs_test -= mean	
		return imgs_test,self.npy_path

class myUnet(object):

	def __init__(self, img_rows = 512, img_cols = 512):

		self.img_rows = img_rows
		self.img_cols = img_cols

	def load_data(self):
		#Load data from not augmentated directories.
		mydata = dataProcess(self.img_rows, self.img_cols)
		imgs_train, imgs_mask_train = mydata.load_train_data()
		#print imgs_train.ndim, imgs_mask_train.ndim
		imgs_test = mydata.load_test_data()
		return imgs_train, imgs_mask_train, imgs_test

	def load_data_test(self):
		mydata = dataProcess(self.img_rows, self.img_cols)
		imgs_test,directory = mydata.load_test_data()
		return imgs_test,directory

	def save_predicted_img(self,temp):

		print("Save predicted images")
		print('-'*30)
		imgs = np.load(temp+'/imgs_result_predict.npy')
		names = np.load(temp+'/test_names.npy')

		for i in range(imgs.shape[0]):
			img = imgs[i]
			img = array_to_img(img)
			img.save(temp+'/prediction_results_'+str(names[i])+".jpg")
			#temp is: ./03_image_directory/Predict_image/

	def predict(self, model, option):
		print('Predict images')
		print('-'*30)
		imgs_test, direct=self.load_data_test()
		imgs_mask_test = model.predict(imgs_test, batch_size=5, verbose=1)
		np.save(direct+'/imgs_result_predict.npy', imgs_mask_test)			
		return direct

	def create_model(self):

		inputs = Input((512, 512,1))
		conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
		conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
		pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
		

		conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
		conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
		pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)


		conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
		conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
		pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)



		conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
		conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
		drop4 = Dropout(0.5)(conv4)
		pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

		conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
		conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
		drop5 = Dropout(0.5)(conv5)

		up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
		merge6 = merge([drop4,up6], mode = 'concat', concat_axis = 3)
		conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
		conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

		up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
		merge7 = merge([conv3,up7], mode = 'concat', concat_axis = 3)
		conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
		conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

		up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
		merge8 = merge([conv2,up8], mode = 'concat', concat_axis = 3)
		conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
		conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

		up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
		merge9 = merge([conv1,up9], mode = 'concat', concat_axis = 3)
		conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
		conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
		conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
		conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

		model = Model(input = inputs, output = conv10)

		return model

if __name__ == '__main__':
	myunet = myUnet()
	model_to_use=sys.argv[1]
	if sys.argv[1]=="unet.hdf5":
		model=myunet.create_model()

		#do not use multi GPU by default
		#model = multi_gpu_model(model, gpus=2)
		model.load_weights("./01_model_parameters/unet.hdf5")
		temp=myunet.predict(model,option="aug")
		myunet.save_predicted_img(temp)

	elif sys.argv[1]=="unet_refined.hdf5":
		model=myunet.create_model()
		
		#do not use multi GPU by default
		#model = multi_gpu_model(model, gpus=2)
		model.load_weights("./01_model_parameters/unet_refined.hdf5")
		temp=myunet.predict(model,option="aug")
		myunet.save_predicted_img(temp)

	elif sys.argv[1]=="unet_user_new.hdf5":
		model=myunet.create_model()

		#do not use multi GPU by default
		#model = multi_gpu_model(model, gpus=2)
		model.load_weights("./01_model_parameters/unet_user_new.hdf5")
		temp=myunet.predict(model,option="aug")
		myunet.save_predicted_img(temp)


	else:
	#raise error message and quit;
		print 'DeeperCount U-Net predicting code, please specify the model parameter you want to use.'
        	print ''
	        sys.exit(2)
