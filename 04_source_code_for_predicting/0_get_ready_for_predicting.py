#generating the npy files, using the provided tif files.

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np 
import os
import glob
import math
import sys
#from libtiff import TIFF

class dataProcess(object):

	def __init__(self, out_rows, out_cols, test_path = "./03_image_directory/Predict_image", npy_path = "./", img_type = sys.argv[1]):

		self.out_rows = out_rows
		self.out_cols = out_cols
		self.img_type = img_type
		self.test_path = test_path
		self.npy_path = npy_path

	def create_test_data(self,option):
		i = 0

		if option != "not_original_size":
			imgs = glob.glob(self.test_path+"/*."+self.img_type)
			imgs.sort()
			namearr=[]
			imgdatas = np.ndarray((len(imgs),self.out_rows,self.out_cols,1), dtype=np.uint8)
			for imgname in imgs:
				midname = imgname[imgname.rindex("/")+1:]
				namearr.append(midname[:midname.rindex(".")])
				img = load_img(self.test_path + "/" + midname,grayscale = True)
				img = img_to_array(img)
				#img = cv2.imread(self.test_path + "/" + midname,cv2.IMREAD_GRAYSCALE)
				#img = np.array([img])
				imgdatas[i] = img
				i += 1
			np.save(self.npy_path + '/imgs_test.npy', imgdatas)
			np.save(self.npy_path+ '/test_names.npy',np.asarray(namearr))
			print('Saving to imgs_test.npy files completed')
		else:
			imgs = glob.glob(self.test_path+"/*."+self.img_type)
			imgs.sort()
			namearr=[]
			print self.test_path+"/"+imgs[0][imgs[0].rindex("/")+1:]
			img_temp=img_to_array(load_img(self.test_path+"/"+imgs[0][imgs[0].rindex("/")+1:],grayscale=True))
			#re-define the length of imgdatas, using a loaded sample image img_temp:
			#30 is the step of iteration in cutting the image

			#has situation that the width and height of the image is of 512 pixels:

			if img_temp.shape[0]> 512 and img_temp.shape[1]> 512:

				imgdatas = np.ndarray((int(math.ceil(float(img_temp.shape[0]-512+1)/30)*math.ceil(float(img_temp.shape[1]-512+1)/30))*len(imgs),self.out_rows,self.out_cols,1), dtype=np.uint8)

				#imgs has an array of stored images;
				for imgname_temp in imgs:

				#then use imgname_temp to get the name of the image:

					temp=imgname_temp[imgname_temp.rindex("/")+1:]
					midname=temp
				#then you need to iterate the matrix that read into python, so that many images can be generated. By sliding window on matrixs.
					img = load_img(self.test_path + "/" + midname,grayscale = True)
					img = img_to_array(img)
					for row in range(0,img.shape[0]-512,30):
						for col in range (0,img.shape[1]-512,30):
#							imgdatas[i]=img[row:row+512,col:col+512]
#							namearr.append(midname[:midname.rindex(".")]+str(row)+"_"+str(col))
#							i +=1
							
							#if you do not want to lose information:
		                                        if img.shape[1]-col-512>30 and img.shape[0]-row-512>30:
                		                                imgdatas[i]=img[row:row+512,col:col+512]
                                		                namearr.append(midname[:midname.rindex(".")]+str(row)+"_"+str(col))
								i +=1
                                                		
		                                        if img.shape[1]-col-512<30 and img.shape[0]-row-512>30:
		                                                imgdatas[i]=img[row:row+512,img.shape[1]-512:img.shape[1]]
                                        			namearr.append(midname[:midname.rindex(".")]+str(row)+"_"+str(img.shape[1]))
								i +=1
							if img.shape[0]-row-512<30 and img.shape[1]-col-512>30:

		                                                imgdatas[i]=img[img.shape[0]-512:img.shape[0],col:col+512]
								namearr.append(midname[:midname.rindex(".")]+str(img.shape[0])+"_"+str(col))
								i +=1
							
        		                                if img.shape[0]-row-512<30 and img.shape[1]-col-512<30:
	                	                                imgdatas[i]=img[img.shape[0]-512:img.shape[0],img.shape[1]-512:img.shape[1]]
								namearr.append(midname[:midname.rindex(".")]+str(img.shape[0])+"_"+str(img.shape[1]))
								i +=1


			elif img_temp.shape[0]==512 and img_temp.shape[1]>512: 	#meaning the image size is less than 512, basially the height is 512
				imgdatas = np.ndarray((int(math.ceil(float(img_temp.shape[1]-512+1)/30)*len(imgs)),self.out_rows,self.out_cols,1), dtype=np.uint8)



				#imgs has an array of stored images;
				for imgname_temp in imgs:

				#then use imgname_temp to get the name of the image:

					temp=imgname_temp[imgname_temp.rindex("/")+1:]
					midname=temp
				#then you need to iterate the matrix that read into python, so that many images can be generated. By sliding window on matrixs.
					img = load_img(self.test_path + "/" + midname,grayscale = True)
					img = img_to_array(img)	
					row=0
						
					for col in range (0,img.shape[1]-512,30):
						imgdatas[i]=img[row:row+512,col:col+512]
						namearr.append(midname[:midname.rindex(".")]+str(row)+"_"+str(col))
						i +=1
			else:
				imgdatas=np.ndarray((len(imgs),512,512,1), dtype=np.uint8)
				for imgname_temp in imgs:
					temp=imgname_temp[imgname_temp.rindex("/")+1:]
					midname=temp
		                        img = load_img(self.test_path + "/" + midname,grayscale = True)
                		        img = img_to_array(img)
                        		imgdatas[i]=img
					namearr.append(midname[:midname.rindex(".")])

			#name is consisting name+row value+column value
			np.save(self.npy_path + '/imgs_test.npy', imgdatas)
			np.save(self.npy_path+ '/test_names.npy',np.asarray(namearr))
			print('Saving to imgs_test.npy files completed')

if __name__ == "__main__":
	#if you want to do the prediction on your data, then the images should be 512*512 in size, or use the following code:
	mydata =  dataProcess(512,512,test_path = "./03_image_directory/Predict_image/", npy_path = "./03_image_directory/Predict_image/")

	#then the npy file will be generated.
	mydata.create_test_data(option="not_original_size")
