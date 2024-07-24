import homography, sift2
import camera
import numpy as np
from PIL import Image

# load images
im1_path = '/home/ekagra/Documents/GitHub/Computer-Vision-Tutorial/data/empire_test_image.jpg'
im2_path = '/home/ekagra/Documents/GitHub/Computer-Vision-Tutorial/data/empire_test_image_blurred.jpg'
im1 = np.array(Image.open(im1_path).convert('L'))
im2 = np.array(Image.open(im2_path).convert('L'))

# compute features
l0, d0 = sift2.process_image(im1_path)
l1, d1 = sift2.process_image(im2_path)

# match features and estimate homography
matches = sift2.match_twosided(desc1=d0, desc2=d1)
ndx = matches.nonzero()[0]
fp = homography.make_homog(l0[ndx, :2].T)
ndx2 = [int(matches[i]) for i in ndx]
tp = homography.make_homog(l1[ndx2, :2].T)
model = homography.RansacModel()
print(f'fp: {fp.shape}\ttp: {tp.shape}')
H = homography.H_from_ransac(fp, tp, model)

def cube_points(c, wid):
    """Creates a list of points for plotting a cube with plot.
    (The first 5 points are the bottom square, some sides repeated)
    """
    p = []
    # bottom
    p.append([c[0] - wid, c[1] - wid, c[2] - wid])
    p.append([c[0] - wid, c[1] + wid, c[2] - wid])
    p.append([c[0] + wid, c[1] + wid, c[2] - wid])
    p.append([c[0] + wid, c[1] - wid, c[2] - wid])
    p.append([c[0] - wid, c[1] - wid, c[2] - wid])
    p.append([c[0] - wid, c[1] - wid, c[2] - wid])

    # top
    p.append([c[0] - wid], c[1] - wid, c[2] + wid)
    p.append([c[0] - wid])