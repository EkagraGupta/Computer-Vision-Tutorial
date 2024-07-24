import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def process_image(im_path):
    im = np.array(Image.open(im_path).convert('L'))
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(im, None)
    locs = []
    for kp in keypoints:
        pts = kp.pt
        scale = kp.size
        rot = kp.angle
        locs_tup = (pts[0], pts[1], scale, rot)
        locs.append(locs_tup)
    return np.array(locs), descriptors

def plot_features(im, locs, circle=False):
    """Show image with featrures.

    Args:
        im (array): input image as array
        locs (array): position, size and orientation of each feature
        circle (bool, optional): _description_. Defaults to False.
    """

    def draw_circle(c, r):
        t = np.arange(0, 1.01, .01) * 2 * np.pi
        x = r * np.cos(t) + c[0]
        y = r * np.sin(t) + c[1]
        print(t)
        plt.plot(x, y, 'r', linewidth=2)
    
    plt.imshow(im)
    
    if circle:
        for p in locs:
            draw_circle(p[:2], p[2])
    else:
        plt.plot(locs[:, 0], locs[:, 1], 'ob')
    plt.axis('off')

def match(desc1, desc2):
    """FOr each descriptor in the first image, select its match in the second image.

    Args:
        desc1 (_type_): _description_
        desc2 (_type_): _description_
    """
    desc1 = np.array([d / np.linalg.norm(d) for d in desc1])
    desc2 = np.array([d / np.linalg.norm(d) for d in desc2])
    
    dist_ratio = 0.6
    desc1_size = desc1.shape

    match_scores = np.zeros((desc1_size[0], 1), 'int')

    for i in range(desc1_size[0]):
        dot_prods = np.dot(desc1[i, :], desc2.T)
        dot_prods = 0.9999 * dot_prods

        # inverse cosine and sort, return index for features in second image
        indx = np.argsort(np.arccos(dot_prods))

        # check if nearest neighbor has angle less than dist_ratio times 2nd
        if np.arccos(dot_prods)[indx[0]] < dist_ratio * np.arccos(dot_prods)[indx[1]]:
            match_scores[i] = int(indx[0])
    
    return match_scores

def match_twosided(desc1, desc2):
    """Two-sided symmetric version of match()

    Args:
        desc1 (_type_): _description_
        desc2 (_type_): _description_
    """

    matches_12 = match(desc1, desc2)
    matches_21 = match(desc2, desc1)

    ndx_12 = matches_12.nonzero()[0]

    # remove matches that are not symmetric
    for n in ndx_12:
        if matches_21[int(matches_12[n])] != n:
            matches_12[n] = 0
    
    return matches_12

def appendimages(im1,im2):
    """ Return a new image that appends the two images side-by-side. """

    # select the image with the fewest rows and fill in enough empty rows
    rows1 = im1.shape[0]
    rows2 = im2.shape[0]

    if rows1 < rows2:
        im1 = np.concatenate((im1,np.zeros((rows2-rows1,im1.shape[1]))),axis=0)
    elif rows1 > rows2:
        im2 = np.concatenate((im2,np.zeros((rows1-rows2,im2.shape[1]))),axis=0)

    # if none of these cases they are equal, no filling needed.
    return np.concatenate((im1,im2), axis=1)

def plot_matches(im1,im2,locs1,locs2,matchscores,show_below=True):
    """ Show a figure with lines joining the accepted matches
    input: im1,im2 (images as arrays), locs1,locs2 (feature locations),
    matchscores (as output from match()),
    show_below (if images should be shown below matches). """
    im3 = appendimages(im1,im2)
    
    if show_below:
        im3 = np.vstack((im3,im3))
    plt.imshow(im3)
    cols1 = im1.shape[1]
    
    for i,m in enumerate(matchscores):
        if m>0:
            x = [locs1[i][1], locs2[m][0][1]+cols1]
            y = [locs1[i][0],locs2[m][0][0]]
            plt.plot(x, y, 'c')
            # plt.axis('off')
    plt.show()

if __name__=='__main__':
    im2_path = '/home/ekagra/Documents/GitHub/Computer-Vision-Tutorial/data/empire_test_image.jpg'
    im1_path = '/home/ekagra/Documents/GitHub/Computer-Vision-Tutorial/data/empire_test_image_blurred.jpg'
    im1 = np.array(Image.open(im1_path).convert('L'))
    im2 = np.array(Image.open(im2_path).convert('L'))

    l1, d1 = process_image(im_path=im1_path)
    l2, d2 = process_image(im_path=im2_path)
    
    match = match_twosided(desc1=d1, desc2=d2)
    plot_matches(im1=im1, im2=im2, locs1=l1, locs2=l2, matchscores=match, show_below=False)