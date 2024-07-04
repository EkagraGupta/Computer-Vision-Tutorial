from scipy import ndimage
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

im_path = '/home/ekagra/personal/projects/ComputerVision/data/empire_test_image.jpg'
im = np.array(Image.open(im_path).convert('L'))
H = np.array([
    [1.4, .05, -100],
    [.05, 1.5, -100],
    [0, 0, 1]
])
im2 = ndimage.affine_transform(im, H[:2, :2], (H[0, 2], H[1, 2]))

plt.figure()
plt.gray()
plt.imshow(im2)
plt.show()