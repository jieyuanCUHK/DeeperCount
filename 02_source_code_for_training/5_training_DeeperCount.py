import os 
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"	#default: using two GPUs, the parameters were trained on 5 GPUs
from keras.utils import multi_gpu_model
import numpy as np
from keras.models import *
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras import backend as keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras import backend as K
import tensorflow as tf

def usage():
        print 'DeeperCount Training Code'
        print 'Author: jieyuan@link.cuhk.edu.hk'
        print 'Usage: python 5_training_DeeperCount.py [unet.hdf5|unet_refined.hdf5]'
        sys.exit(2)

# check args
if len(sys.argv) < 1:
        usage()



class dataProcess(object):

	def __init__(self, out_rows, out_cols, data_path, label_path, npy_path = "./", img_type = "tif"):

		self.out_rows = out_rows
		self.out_cols = out_cols
		self.data_path = data_path
		self.label_path = label_path
		self.img_type = img_type
		self.npy_path = npy_path


	def load_train_data_aug(self):
		print('Loading images')
		print('-'*30)
		imgs_train = np.load(self.npy_path+"/aug_imgs_train.npy")
		imgs_mask_train = np.load(self.npy_path+"/aug_imgs_mask_train.npy")
		imgs_train = imgs_train.astype('float32')
		imgs_mask_train = imgs_mask_train.astype('float32')
		imgs_train /= 255
		#mean = imgs_train.mean(axis = 0)
		#imgs_train -= mean	
		imgs_mask_train /= 255
		imgs_mask_train[imgs_mask_train > 0.5] = 1
		imgs_mask_train[imgs_mask_train <= 0.5] = 0
		return imgs_train,imgs_mask_train

class myUnet(object):

	def __init__(self, img_rows = 512, img_cols = 512):

		self.img_rows = img_rows
		self.img_cols = img_cols

	def load_data_aug_new(self):
		mydata = dataProcess(self.img_rows, self.img_cols,data_path="./03_image_directory",label_path="./03_image_directory",npy_path="./03_image_directory")
		imgs_train, imgs_mask_train = mydata.load_train_data_aug()
		return imgs_train, imgs_mask_train

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

import random


if __name__ == '__main__':
	myunet = myUnet()
	model_to_use=sys.argv[1]
	if sys.argv[1]=="unet.hdf5":
	#want to fine tune the parameters learned on EM images
		model=myunet.create_model()
	
		#multi-gpu, default is 2;
		model=multi_gpu_model(model, gpus=2)
		model.load_weights("./01_model_parameters/unet.hdf5")

		#load data
		imgs_train,imgs_mask_train=myunet.load_data_aug_new()
		train_x, val_x, train_y, val_y = train_test_split(imgs_train, imgs_mask_train, test_size=0.1)


		model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
		#use early stopper	
		earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
		model_checkpoint = ModelCheckpoint('./01_model_parameters/unet_user_new.hdf5', monitor='loss',verbose=1, save_best_only=True)

		history=model.fit(train_x, train_y, batch_size=1, nb_epoch=1, verbose=1,shuffle=True, validation_data=(val_x,val_y),callbacks=[model_checkpoint,earlystopper])

		#draw performance figures;
		'''
		plt.plot(history.history['acc'])
		plt.plot(history.history['val_acc'])
		plt.title('Model Accuracy Using User Images')
		plt.ylabel('Accuracy')
		plt.xlabel('Epoch')
		plt.legend(['Train', 'Validation'], loc='upper left')
		plt.savefig("accuracy_perfor.png")

		plt.plot(history.history['loss'])
		plt.plot(history.history['val_loss'])
		plt.title('Model Loss Using User Images')
		plt.ylabel('Loss')
		plt.xlabel('Epoch')
		plt.legend(['train', 'test'], loc='upper left')
		plt.savefig("loss_perfor.png")
		'''
	elif sys.argv[1]=="unet_refined.hdf5":
	#want to fine tune the parameters learned on fiber staining images
		model=myunet.create_model()
		#multi GPU option, 2 by default
		model=multi_gpu_model(model, gpus=2)
		model.load_weights("./01_model_parameters/unet_refined.hdf5")

		#load data
		imgs_train,imgs_mask_train=myunet.load_data_aug_new()
		train_x, val_x, train_y, val_y = train_test_split(imgs_train, imgs_mask_train, test_size=0.1)

		model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
		#use early stopper	
		earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
		model_checkpoint = ModelCheckpoint('unet_user_new.hdf5', monitor='loss',verbose=1, save_best_only=True)

		history=model.fit(train_x, train_y, batch_size=1, nb_epoch=1, verbose=1,shuffle=True, validation_data=(val_x,val_y),callbacks=[model_checkpoint,earlystopper])

		#draw performance figures:
		'''
		plt.plot(history.history['acc'])
		plt.plot(history.history['val_acc'])
		plt.title('Model Accuracy Using User Images')
		plt.ylabel('Accuracy')
		plt.xlabel('Epoch')
		plt.legend(['Train', 'Validation'], loc='upper left')
		plt.savefig("accuracy_perfor.png")

		plt.plot(history.history['loss'])
		plt.plot(history.history['val_loss'])
		plt.title('Model Loss Using User Images')
		plt.ylabel('Loss')
		plt.xlabel('Epoch')
		plt.legend(['train', 'test'], loc='upper left')
		plt.savefig("loss_perfor.png")
		'''
	else:
	#error, give a message and quit
		print 'DeeperCount U-Net training code, please specify the model parameter you want to use.'
        print ''
        sys.exit(2)
