from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt


def sift(im):
    sift = cv2.SIFT_create()
    keyp, desc = sift.detectAndCompute(im, None)
    locs = np.array([[kp.pt[0], kp.pt[1], kp.size, kp.angle] for kp in keyp])
    return locs, desc


def plot_matches(im1, im2, keypoints1, keypoints2, matches):
    im_matches = cv2.drawMatches(
        im1,
        keypoints1,
        im2,
        keypoints2,
        matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    plt.imshow(im_matches)
    plt.show()


def match(desc1, desc2, threshold=0.75):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=2)

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < threshold * n.distance:
            good_matches.append(m)

    return good_matches


def match_twosided(desc1, desc2, threshold=0.75):
    matches_12 = match(desc1, desc2, threshold)
    matches_21 = match(desc2, desc1, threshold)
    matches_12_dict = {m.queryIdx: m for m in matches_12}
    matches_21_dict = {m.queryIdx: m for m in matches_21}

    # Ensure matches are symmetric
    symmetric_matches = []
    for m in matches_12:
        if (
            m.trainIdx in matches_21_dict
            and matches_21_dict[m.trainIdx].trainIdx == m.queryIdx
        ):
            symmetric_matches.append(m)

    return np.array(symmetric_matches)


if __name__ == "__main__":
    im1_path = "/home/ekagra/Documents/GitHub/Computer-Vision-Tutorial/data/empire_test_image_blurred.jpg"
    im2_path = "/home/ekagra/Documents/GitHub/Computer-Vision-Tutorial/data/empire_test_image.jpg"
    im1 = np.array(Image.open(im1_path).convert("L"))
    im2 = np.array(Image.open(im2_path).convert("L"))
    locs1, descriptors1 = sift(im1)
    locs2, descriptors2 = sift(im2)
    matches = match_twosided(desc1=descriptors1, desc2=descriptors2)
    ndx = matches.nonzero()[0]
    # Extracting matched indices
    # ndx = [m.queryIdx for m in matches]
    ndx2 = [m.trainIdx for m in matches]
    import homography
    fp = homography.make_homog(locs1[ndx, :2].T)
    # ndx2 = [int(matches[i]) for i in ndx]
    tp = homography.make_homog(locs2[ndx2, :2].T)
    model = homography.RansacModel()
    H = homography.H_from_ransac(fp, tp, model)
    # Plot matches
    # plot_matches(im1, im2, keypoints1, keypoints2, matches)
    