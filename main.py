from Chapter002 import sift
# import matplotlib.pyplot as plt
# from PIL import Image
# import numpy as np

# imname = '/home/ekagra/personal/projects/ComputerVision/data/empire_test_image.jpg'
# im1 = np.array(Image.open(imname).convert('L'))
# sift.process_image(imname,'/home/ekagra/personal/projects/ComputerVision/data/empire.sift')
# l1,d1 = sift.read_features_from_file('empire.sift')
# plt.figure()
# plt.gray()
# sift.plot_features(im1,l1,circle=True)
# plt.show()

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

def display_image(image, title=None):
    image = image / 2 + 0.5  # unnormalize
    np_image = image.numpy()
    plt.imshow(np.transpose(np_image, (1, 2, 0)))
    if title:
        plt.title(title)
    plt.show()

if __name__ == '__main__':
    im_path = '/home/ekagra/personal/projects/ComputerVision/data/empire_test_image.jpg'
    out_path = '/home/ekagra/personal/projects/ComputerVision/data/empire.sift'
    
    # Convert image to grayscale and process
    im1 = np.array(Image.open(im_path).convert('L'))
    sift.process_image(im_name=im_path, result_name=out_path)
    
    if not os.path.exists(out_path):
        raise FileNotFoundError(f"{out_path} not found.")

    l1, d1 = sift.read_features_from_file(out_path)

    plt.figure()
    plt.gray()
    sift.plot_features(im1, l1, circle=True)
    plt.show()
