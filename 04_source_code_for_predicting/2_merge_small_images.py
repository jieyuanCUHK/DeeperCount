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

#test_r_path="../0_images_input/"+sys.argv[1]+'/'
test_r_path="./"
#test_path="../0_images_input/"+sys.argv[1]+'/'
test_path="./"

img_type="jpg"
imgs = glob.glob(test_r_path+"/*."+img_type)
test_imgs=glob.glob(test_path+"/*.tif")
save_path=test_path+'/deep_results'

if not os.path.lexists(save_path):
    os.mkdir(save_path)
    os.system('chmod 777 '+save_path)

#the original dimension of the image:

height=img_to_array(load_img(test_imgs[0])).shape[0]
width=img_to_array(load_img(test_imgs[0])).shape[1]


#then you need to process all the images in imgs array:
for image_names in test_imgs:

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
				if height-row> 30 and width-col>30:
					readimage=img_to_array(load_img(test_path+image_names[image_names.rindex("/")+1:image_names.rindex(".")]+str(row)+"_"+str(col)+".jpg"))
			
					#see whether the region has been assigned values before, if it has, just ignore; else, assign values to it	
					for i in range(0,512):
						for j in range(0,512):
				#			#print imgdatas[i,j,:]
							tmp=imgdatas[row+i,col+j,:]==-1
							if tmp[0] and tmp[1] and tmp[2]:
								imgdatas[row+i,col+j,:]=readimage[i,j,:]
				elif width-col<30 and height-row>30:
					readimage=img_to_array(load_img(test_path+image_names[image_names.rindex("/")+1:image_names.rindex(".")]+str(row)+"_"+str(width)+".jpg"))	
					for i in range(0,512):
                                        	for j in range(0,512):
							tmp=imgdatas[row+i,width-512+j,:]==-1
							if tmp[0] and tmp[1] and tmp[2]:
								imgdatas[row+i,width-512+j,:]=readimage[i,j,:]
				if height-row<30 and width-col>30:
					readimage=img_to_array(load_img(test_path+image_names[image_names.rindex("/")+1:image_names.rindex(".")]+str(height)+"_"+str(col)+".jpg"))
                                        for i in range(0,512):
                                        	for j in range(0,512):
			 
							tmp=imgdatas[height-512+i,col+j,:]==-1
							if tmp[0] and tmp[1] and tmp[2]:
								imgdatas[height-512+i,col+j,:]=readimage[i,j,:]

				if height-row<30 and width-col<30:
					readimage=img_to_array(load_img(test_path+image_names[image_names.rindex("/")+1:image_names.rindex(".")]+str(height)+"_"+str(width)+".jpg"))
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
			readimage=img_to_array(load_img(test_path+image_names[image_names.rindex("/")+1:image_names.rindex(".")]+str(row)+"_"+str(col)+".jpg"))
			
				#see whether the region has been assigned values before, if it has, just ignore; else, assign values to it
			for i in range(0,512):
				for j in range(0,512):
				#		#print imgdatas[i,j,:]
					tmp=imgdatas[row+i,col+j,:]==-1
					if tmp[0] and tmp[1] and tmp[2]:
						imgdatas[row+i,col+j,:]=readimage[i,j,:]
	
	if height > 512 and width>512:

		count_row=0
		row_arr=[]
		
		for items in imgdatas[:,:,0][:,]:		#for rows, 1*1000
			#print items.shape
			if sum(items)==-1*items.shape[0]:
				row_arr.append(count_row)
			count_row=count_row+1
		
		
		determinrow=min(row_arr)		#gives the row boundary


		count_col=0
		col_arr=[]

		for columns in np.transpose(imgdatas[:,:,0]):		#for columns
			print columns.shape
			if sum(columns)==-1*columns.shape[0]:
					col_arr.append(count_col)
			count_col=count_col+1

		determincol=min(col_arr)
		print determincol

	elif height==512 and width>512:
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
	
	else:
	#the image is exactly in 512*512 size
		determincol=0
		determinrow=0
	#then read in the image and save them
		original_image=img_to_array(load_img(image_names))
		savename= image_names[image_names.rindex("/")+1:]
		original_image=array_to_img(original_image)
		original_image.save(save_path+'/'+savename+"_adjusted_"+".tif")
		predicted_image=img_to_array(load_img(image_names[:-4]+".jpg"))
		predicted_image=array_to_img(predicted_image)
		predicted_image.save(save_path+'/'+savename+"_label_"+".tif")

	#actually the original image has some boundary that is lost during the sliding window procedure, so the boundaries should be eliminated.
	#so use the boundaries that are determined above:
	if determincol !=0 and determinrow!=0:
		original_image=img_to_array(load_img(image_names))
		original_image=original_image[0:determinrow+1,0:determincol+1,:]
		original_image= array_to_img(original_image)
		savename= image_names[image_names.rindex("/")+1:]

		original_image.save(save_path+"/"+savename+"_adjusted_"+".tif")
		imgdatas=np.uint8(imgdatas)
		imgdatas=imgdatas[0:determinrow+1,0:determincol+1,:]
		imgdatas = array_to_img(imgdatas)
		imgdatas.save(save_path+'/'+savename+"_label_"+".tif")
