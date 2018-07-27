#generating the npy files, using the provided tif files.

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np 
import os
import glob
import cv2
import cv2
import sys
#from libtiff import TIFF

class myAugmentation(object):
	
	def __init__(self, aug_merge_path="./03_image_directory/After_augmentation_primary", aug_train_path="./03_image_directory/Final_train", aug_label_path="./03_image_directory/Final_label",  img_type="tif"):
		
		self.img_type = img_type
		self.aug_merge_path = aug_merge_path
		self.aug_train_path = aug_train_path
		self.aug_label_path = aug_label_path
		self.aug_imgs=glob.glob(aug_merge_path+"/*."+img_type)
		self.slices = len(self.aug_imgs)
	def splitMerge(self):

		"""
		split merged image (after image augmentation) apart
		"""
		path_merge = self.aug_merge_path
		path_train = self.aug_train_path
		path_label = self.aug_label_path
		path = path_merge + "/"
		train_imgs = glob.glob(path+"/*."+self.img_type)
		savedir = path_train + "/"
		if not os.path.lexists(savedir):
			os.mkdir(savedir)
		savedir = path_label + "/"
		if not os.path.lexists(savedir):
			os.mkdir(savedir)
		count=0
		for imgname in train_imgs:
			midname = imgname[imgname.rindex("/")+1:imgname.rindex("."+self.img_type)]
			img = cv2.imread(imgname)
			img_train = img[:,:,2]#cv2 read image rgb->bgr
			img_label = img[:,:,0]
			cv2.imwrite(path_train+"/"+midname+"_train"+"."+self.img_type,img_train)
			cv2.imwrite(path_label+"/"+midname+"_label"+"."+self.img_type,img_label)
			count=count+1
			if count==len(train_imgs):
				break
				

class dataProcess(object):

	def __init__(self, out_rows, out_cols, data_path, label_path, npy_path = "./", img_type = "tif"):

		self.out_rows = out_rows
		self.out_cols = out_cols
		self.data_path = data_path
		self.label_path = label_path
		self.img_type = img_type
		self.npy_path = npy_path


	def create_train_data_aug(self):
		i = 0
		print('Creating training images')
		print('-'*30)

		imgs = glob.glob(self.data_path+"/*."+self.img_type)
		corr_labels= glob.glob(self.label_path+"/*."+self.img_type)
		imgs.sort()
		corr_labels.sort()

		imgdatas = np.ndarray((len(imgs),self.out_rows,self.out_cols,1), dtype=np.uint8)
		imglabels = np.ndarray((len(imgs),self.out_rows,self.out_cols,1), dtype=np.uint8)
		for itering in zip(imgs,corr_labels):
			imgname=itering[0]
			labelname=itering[1]
			midname = imgname[imgname.rindex("/")+1:]
			midname_label= labelname[labelname.rindex("/")+1:]

			img = load_img(self.data_path + "/" + midname,grayscale = True)
			label = load_img(self.label_path + "/" + midname_label,grayscale = True)
			img = img_to_array(img)
			label = img_to_array(label)
			#img = cv2.imread(self.data_path + "/" + midname,cv2.IMREAD_GRAYSCALE)
			#label = cv2.imread(self.label_path + "/" + midname,cv2.IMREAD_GRAYSCALE)
			#img = np.array([img])
			#label = np.array([label])
			imgdatas[i] = img
			imglabels[i] = label
			if i % 100 == 0:
				print('Done: {0}/{1} images'.format(i, len(imgs)))
			i += 1
		print('Creation finished')
		print('-'*30)
		np.save(self.npy_path + '/aug_imgs_train.npy', imgdatas)
		np.save(self.npy_path + '/aug_imgs_mask_train.npy', imglabels)
		print('Saving to .npy files completed')


if __name__ == "__main__":
	au=myAugmentation()
    	au.splitMerge()   
	mydata_t= dataProcess(512,512,data_path="./03_image_directory/Final_train",label_path="./03_image_directory/Final_label",npy_path="./03_image_directory/")
	mydata_t.create_train_data_aug()

