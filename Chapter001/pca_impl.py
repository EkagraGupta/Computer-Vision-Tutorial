from PIL import Image
import numpy as np
import pylab as pl
import pca
import zipfile

from imtools import get_imlist

# imlist = get_imlist('/home/ekagra/personal/projects/ComputerVision/data')

# print(imlist)
im_path = "/home/ekagra/personal/projects/ComputerVision/data/empire_test_image.jpg"
im = np.array(Image.open(im_path).convert("L"))  # Open one image to get size
m, n = im.shape[0:2]  # Get the size of images
im_num = 1  # Get the number of images

# Create a matrix to store all flattened images
im_matrix = np.array([im.flatten()], "f")

# Perform PCA
V, S, im_mean = pca.pca(im_matrix)

# Show some images (mean)
pl.figure()
pl.gray()
pl.imshow(im_mean.reshape(m, n))
pl.show()
