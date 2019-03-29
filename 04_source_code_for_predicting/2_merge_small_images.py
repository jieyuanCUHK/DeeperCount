#!/usr/bin/env python
# -*- coding: utf-8 -*- 


#code used to merge the single images into a big one
#format: python 2_merge_small_images.py suyang_20180627
 
import numpy as np
import os
import glob
import re
import sys
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import shutil


test_r_path="./03_image_directory/Predict_image"
test_path="./03_image_directory/Prediction_results/"

#test_r_path: the original image that used for prediction
#test_path: the folder of the predicted results

img_type="jpg"
imgs = glob.glob(test_r_path+"/*."+img_type)
test_imgs=glob.glob(test_r_path+"/*."+sys.argv[1])	#test_imgs: denotes the original shape of the image;

save_path=test_path+'Merged/'

if not os.path.lexists(save_path):
    os.mkdir(save_path)
    os.system('chmod 777 '+save_path)

#the original dimension of the image:

height=img_to_array(load_img(test_imgs[0])).shape[0]
width=img_to_array(load_img(test_imgs[0])).shape[1]


#then you need to process all the images in imgs array:
for image_names in imgs:

	#then initialize a np.array:
	imgdatas=np.ndarray((height,width,3))
	#print imgdatas.shape

	#initialized by all elements of zero/ negative 1
	imgdatas.fill(-1)
	#print imgdatas

	#then begin to replace the values in imgdatas to actual values
	#pay attention that in newer training situation, the image width is 512:

	if height > 512 and width>512:

		for row in range(0,height-512,30):
			for col in range (0,width-512,30):
				print col,row
				if height-row-512> 30 and width-col-512>30:
					#print re.split('[_]',image_names[image_names.rindex("/")+1:image_names.rindex(".")])[0:2] 
					readimage=img_to_array(load_img(test_path+image_names[image_names.rindex("/")+1:image_names.rindex(".")]+"_"+str(row)+"_"+str(col)+"_.jpg"))
			
					#see whether the region has been assigned values before, if it has, just ignore; else, assign values to it	
					for i in range(0,512):
						for j in range(0,512):
				#			#print imgdatas[i,j,:]
							tmp=imgdatas[row+i,col+j,:]==-1
							if tmp[0] and tmp[1] and tmp[2]:
								imgdatas[row+i,col+j,:]=readimage[i,j,:]
				if width-col-512<30 and height-row-512>30:
					readimage=img_to_array(load_img(test_path+image_names[image_names.rindex("/")+1:image_names.rindex(".")]+"_"+str(row)+"_"+str(width)+"_.jpg"))	
					for i in range(0,512):
                                        	for j in range(0,512):
							tmp=imgdatas[row+i,width-512+j,:]==-1
							if tmp[0] and tmp[1] and tmp[2]:
								imgdatas[row+i,width-512+j,:]=readimage[i,j,:]
				if height-row-512<30 and width-col-512>30:
					readimage=img_to_array(load_img(test_path+image_names[image_names.rindex("/")+1:image_names.rindex(".")]+"_"+str(height)+"_"+str(col)+"_.jpg"))
                                        for i in range(0,512):
                                        	for j in range(0,512):
			 
							tmp=imgdatas[height-512+i,col+j,:]==-1
							if tmp[0] and tmp[1] and tmp[2]:
								imgdatas[height-512+i,col+j,:]=readimage[i,j,:]

				if height-row-512<30 and width-col-512<30:
					readimage=img_to_array(load_img(test_path+image_names[image_names.rindex("/")+1:image_names.rindex(".")]+"_"+str(height)+"_"+str(width)+"_.jpg"))
					for i in range(0,512):
						for j in range(0,512):
							tmp=imgdatas[height-512+i,width-512+j,:]==-1
							if tmp[0] and tmp[1] and tmp[2]:
								imgdatas[height-512+i,width-512+j,:]=readimage[i,j,:]

														 
				#imgdatas[row:row+512,col:col+512]=readimage
	elif height==512 and width>512:
	#then the image width is 512, need to modify:
		row=0
		for col in range (0,width-512,30):
			readimage=img_to_array(load_img(test_path+image_names[image_names.rindex("/")+1:image_names.rindex(".")]+"_"+str(row)+"_"+str(col)+"_.jpg"))
			
				#see whether the region has been assigned values before, if it has, just ignore; else, assign values to it
			for i in range(0,512):
				for j in range(0,512):
				#		#print imgdatas[i,j,:]
					tmp=imgdatas[row+i,col+j,:]==-1
					if tmp[0] and tmp[1] and tmp[2]:
						imgdatas[row+i,col+j,:]=readimage[i,j,:]
	
	

	if height==512 and width>512:
		determinrow=511
		count_col=0
		col_arr=[]

		for columns in np.transpose(imgdatas[:,:,0]):		#for columns
			print columns.shape
			if sum(columns)==-1*columns.shape[0]:
					col_arr.append(count_col)
			count_col=count_col+1

		determincol=min(col_arr)
		print determincol
	
	if height>512 and width>512:
		determinrow=height-1
		determincol=width-1

	if height==512 and width==512:
	#the image is exactly in 512*512 size
		determincol=0
		determinrow=0
		
	#actually the original image has some boundary that is lost during the sliding window procedure, so the boundaries should be eliminated.
	#so use the boundaries that are determined above:
	if determincol !=0 and determinrow!=0:
		imgdatas=np.uint8(imgdatas)
		imgdatas=imgdatas[0:determinrow+1,0:determincol+1,:]
		imgdatas = array_to_img(imgdatas)
		imgdatas.save(save_path+image_names[image_names.rindex("/")+1:image_names.rindex(".")]+"_Final_Prediction_"+".jpg")
