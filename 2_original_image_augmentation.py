from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np 
import os
import glob
import matplotlib.pyplot as plt
import sys

def usage():
        print 'Image Augmentation Code'
        print 'Author: jieyuan@link.cuhk.edu.hk'
        print 'Usage: python 2_original_image_augmentation.py augmentated_image_number'
        sys.exit(2)

# check args
if len(sys.argv) < 1:
        usage()

class myAugmentation(object):
	
	"""
	Image augmentation using Keras ImageDataGenerator.
	"""

	def __init__(self, merge_path="./03_image_directory/Merged_after_deform", aug_merge_path="./03_image_directory/After_augmentation_primary", img_type="tif"):
		
		# merge_path: the path of merging the training image and label.
		# aug_merge_path: the path of the merged image with augmentation
		"""
		Using glob to get all .img_type form path
		"""
		self.merged_imgs=glob.glob(merge_path+"/*."+img_type)
		self.merge_path = merge_path
		self.img_type = img_type
		self.aug_merge_path = aug_merge_path
		#needs data agumentation
		self.datagen = ImageDataGenerator(
							        rotation_range=20,
							        width_shift_range=0.2,
							        height_shift_range=0.2,
							        shear_range=0.1,
							        zoom_range=0.1,
							        horizontal_flip=True,
							        fill_mode='reflect')
								
	def Augmentation(self):

		"""
		Start augmentation.....
		"""
		trains = self.merged_imgs
		path_merge = self.merge_path
		imgtype = self.img_type
		path_aug_merge = self.aug_merge_path


		j=0
		#then read for images, which already been deformed.
		for imgname in trains:
			j=j+1
			midname = imgname[imgname.rindex("/")+1:]
			
			img_t = load_img(path_merge+"/"+midname)
			x_t = img_to_array(img_t)
			
			#img is the merged image after deformation
			img = x_t

			img = img.reshape((1,) + img.shape) 
			#image is (1, 512, 512) in dimension
			savedir = path_aug_merge + "/"
			if not os.path.lexists(savedir):
				os.mkdir(savedir)
			if j<=len(trains):			
				self.doAugmentate(img, savedir, midname)

	def doAugmentate(self, img, save_to_dir, save_prefix, batch_size=5, save_format='tif', imgnum=sys.argv[1]):
		datagen = self.datagen
		i = 0
		for batch in datagen.flow(img,
                          batch_size=batch_size,
                          save_to_dir=save_to_dir,
                          save_prefix=save_prefix,
                          save_format=save_format):
			  i += 1
		    	  if i > int(imgnum)-1:
		          	break


if __name__ == "__main__":

	
	aug = myAugmentation()
	aug.Augmentation()
