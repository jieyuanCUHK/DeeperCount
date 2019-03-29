#generating the npy files, using the provided tif files.

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np 
import os
import glob
import sys
#from libtiff import TIFF

def usage():
        print 'Spliting Image Code'
        print 'Author: jieyuan@link.cuhk.edu.hk'
        print 'Usage: python 0_split_image.py image_format'
	print ''
        sys.exit(2)

# check args
if len(sys.argv) < 1:
        usage()



def splitimages(path,path_save,img_type):
	imgs = glob.glob(path+"/*."+img_type)
	imgs.sort()
	print imgs
	namearr=[]
	img_temp=img_to_array(load_img(path+"/"+imgs[0][imgs[0].rindex("/")+1:],grayscale=True)) 

	#make sure that path_save is really existing:
	if not os.path.lexists(path_save):
		os.mkdir(path_save)
	if img_temp.shape[0]> 512 and img_temp.shape[1]> 512:

		imgdatas = np.ndarray((512,512,1), dtype=np.uint8)

		#imgs has an array of stored images;
		for imgname_temp in imgs:
		#then use imgname_temp to get the name of the image:

			temp=imgname_temp[imgname_temp.rindex("/")+1:]
			midname=temp
			print midname
			#then you need to iterate the matrix that read into python, so that many images can be generated. By sliding window on matrixs.
			img = load_img(path + "/" + midname,grayscale = True)	#the input is converted to (row,column,1)
			img = img_to_array(img)
			print img.shape

			for row in range(0,img.shape[0]-512,30):		#step length: 30
				for col in range (0,img.shape[1]-512,30):

					#if you do not want to lose information:
					if img.shape[1]-col-512>30 and img.shape[0]-row-512>30:
						imgdatas[:,:,0]=img[row:row+512,col:col+512,0]
						new=array_to_img(imgdatas)
						new.save(path_save+"/"+midname+"_"+str(col)+"_"+str(row)+".tif")
					if img.shape[1]-col-512<30 and img.shape[0]-row-512>30:
						imgdatas[:,:,0]=img[row:row+512,img.shape[1]-512:img.shape[1],0]
						new=array_to_img(imgdatas)
						new.save(path_save+"/"+midname+"_"+str(img.shape[1])+"_"+str(row)+".tif")
					if img.shape[0]-row-512<30 and img.shape[1]-col-512>30:
						imgdatas[:,:,0]=img[img.shape[0]-512:img.shape[0],col:col+512,0]
						new=array_to_img(imgdatas)
						new.save(path_save+"/"+midname+"_"+str(col)+"_"+str(img.shape[0])+".tif")
						
					if img.shape[0]-row-512<30 and img.shape[1]-col-512<30:
						imgdatas[:,:,0]=img[img.shape[0]-512:img.shape[0],img.shape[1]-512:img.shape[1],0]	
						new=array_to_img(imgdatas)
						new.save(path_save+"/"+midname+"_"+str(img.shape[1])+"_"+str(img.shape[0])+".tif")
					
	elif img_temp.shape[0]==512 and img_temp.shape[1]>512:	#by default: the width of the image is 512px.
		imgdatas = np.ndarray((512,512,1), dtype=np.uint8)

		#imgs has an array of stored images;
		for imgname_temp in imgs:
			#then use imgname_temp to get the name of the image:
			temp=imgname_temp[imgname_temp.rindex("/")+1:]
			midname=temp
			#then you need to iterate the matrix that read into python, so that many images can be generated. By sliding window on matrixs.
			img = load_img(path + "/" + midname,grayscale = True)
			img = img_to_array(img)
					
			row=0
							
			for col in range (0,img.shape[1]-512,30):		#step length: 30
				imgdatas[:,:,0]=img[row:row+512,col:col+512,0]
				new=array_to_img(imgdatas)
				new.save(path_save+"/"+midname+"_"+"0"+"_"+str(col)+".tif")
	else:	#shape is exactly 512*512
		imgdatas = np.ndarray((512,512,1), dtype=np.uint8)
		for imgname_temp in imgs:
			temp=imgname_temp[imgname_temp.rindex("/")+1:]
			midname=temp
			img = load_img(path + "/" + midname,grayscale = True)
			imgdatas = img_to_array(img)
			imga=array_to_img(imgdatas)
			imga.save(path_save+"/"+midname+" "+".tif")
	
if __name__ == "__main__":

	splitimages("./03_image_directory/Train_image","./03_image_directory/Split_train_image",sys.argv[1])	#split images, the input parameter is the format of the image
	splitimages("./03_image_directory/Label_image","./03_image_directory/Split_label_image",sys.argv[1])	#split labels, the input parameter is the format of the image
	
