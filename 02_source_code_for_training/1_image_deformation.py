#the code for image elastic transform
import sys
import os
import numpy as np
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import glob
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]
    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
    print
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)

def draw_grid(im, grid_size):
    # Draw grid lines
    for i in range(0, im.shape[1], grid_size):
        cv2.line(im, (i, 0), (i, im.shape[0]), color=(255,))
    for j in range(0, im.shape[0], grid_size):
        cv2.line(im, (0, j), (im.shape[1], j), color=(255,))

path_train="./03_image_directory/Split_train_image"
path_label="./03_image_directory/Split_label_image"
path_merge="./03_image_directory/Merged_after_deform"	#this path stores the image that finished image deformation.
img_type="tif"
train_imgs = glob.glob(path_train+"/*."+img_type)
label_imgs= glob.glob(path_label+"/*."+img_type)
print len(train_imgs)
print len(label_imgs)
if len(train_imgs) != len(label_imgs) or len(train_imgs) == 0 or len(train_imgs) == 0:
	print "Please Make Sure the Number of Training Image Should Equal to the Number of Label Image."
	sys.exit(2)

i=0



#then read for the images:
for imgname in train_imgs:
	i=i+1
	midname = imgname[imgname.rindex("/")+1:]
	img_t = load_img(path_train+"/"+midname)	
	img_l = load_img(path_label+"/"+midname)
	x_t = img_to_array(img_t)
	x_l = img_to_array(img_l)

	savedir = path_merge + "/"
	if not os.path.lexists(savedir):
		os.mkdir(savedir)
			
#put the label into one channnel of the original image, so that augmentation can be done simutaneously.
	x_t[:,:,2] = x_l[:,:,0]
	img_tmp = array_to_img(x_t)
	img_tmp.save(path_merge+"/"+midname)    #saved image that only conducted merging.
	
	#img is the merged original image
	img = x_t	

	#draw_grid(img, int(img.shape[1]*0.125))
	trans = elastic_transform(img, img.shape[1]*2, img.shape[1]*0.08, img.shape[1]*0.05)

	img_fi = array_to_img(trans)
	img_fi.save(path_merge+"/train."+midname)
	

