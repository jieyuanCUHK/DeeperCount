#Code for enhanced image augmentation
import os
from imgaug import augmenters as iaa
import imgaug as ia
import glob
from scipy import misc
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import sys

def usage():
        print '\nEnhanced Image Augmentation Code\n'
        print 'Author: jieyuan@link.cuhk.edu.hk\n'
        print 'Usage: python 3_enhanced_image_augmentation.py number_of_batches batch_size'
        print ''
        sys.exit(2)

# check args
if len(sys.argv) < 2:
        usage()


sometimes = lambda aug: iaa.Sometimes(0.5, aug)
seq = iaa.Sequential([
    	# apply the following augmenters to most images
        iaa.Fliplr(0.5), # horizontally flip 50% of all images
        iaa.Flipud(0.2), # vertically flip 20% of all images
        sometimes(iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
            rotate=(-45, 45), # rotate by -45 to +45 degrees
            shear=(-16, 16), # shear by -16 to +16 degrees
            order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
            mode="reflect" # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        )),
        # execute 0 to 5 of the following (less important) augmenters per image
        # don't execute all of them, as that would often be way too strong
        iaa.SomeOf((0, 5),
            [
                iaa.OneOf([
                    iaa.GaussianBlur((0, 1.0)), # blur images with a sigma between 0 and 3.0
                    
                ]),
                iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                
                # search either for all edges or for directed edges,
                # blend the result with the original image using a blobby mask
	        #iaa.SimplexNoiseAlpha(iaa.OneOf([
                    #iaa.EdgeDetect(alpha=(0.5, 1.0)),
                #])),
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.08*255)), # add gaussian noise to images, for 50% of the overall images.
                iaa.OneOf([
                    iaa.Dropout((0.01, 0.1)), # randomly remove up to 10% of the pixels
                ]),
                
                iaa.Add((-10, 10)), # change brightness of images (by -10 to 10 of original value)
                iaa.AddToHueAndSaturation((-10, 10)), # change hue and saturation
                # either change the brightness of the whole image (sometimes per channel) or change the brightness of subareas
                iaa.ContrastNormalization((0.5, 2.0)), # improve or worsen the contrast
                sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
            ],
            random_order=True
        )
    ],
    random_order=True

)
#conduct image augmentation on both the original merged images and the deformed images.
path_merge="./03_image_directory/Merged_after_deform"
img_type="tif"
imgs=glob.glob(path_merge+"/*."+img_type)
savedir="./03_image_directory/After_augmentation_primary"
if not os.path.lexists(savedir):
    os.mkdir(savedir)

batches=[]

nb_batches=int(sys.argv[1])
batch_size=int(sys.argv[2])
print nb_batches
print batch_size

for imgname in imgs:
    
    midname = imgname[imgname.rindex("/")+1:]
    img_t = load_img(path_merge+"/"+midname)
    x_t = img_to_array(img_t)
    #print int(nb_batches)
    for _ in range(nb_batches):
	batches.append(
	    np.array([x_t for _ in range (batch_size)], dtype=np.uint8)
	)

#print len(batches)
i=0
for images_aug in seq.augment_batches(batches):
    print images_aug.shape
    for j in range(batch_size):
    	misc.imsave(savedir+"/"+"image_%06d.tif" % (i,), images_aug[j])
    	i=i+1

