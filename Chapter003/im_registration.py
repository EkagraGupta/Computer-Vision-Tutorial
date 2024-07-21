from xml.dom import minidom
import numpy as np
from scipy import linalg, ndimage
import os
from PIL import Image
import matplotlib.pyplot as plt


def read_points_from_xml(xmlFileName):
    """Reads control points for face alignment

    Args:
        xmlFileName (_type_): _description_
    """

    xml_doc = minidom.parse(xmlFileName)
    face_list = xml_doc.getElementsByTagName("face")
    faces = {}

    for xml_face in face_list:
        file_name = xml_face.attributes["file"].value
        # xf, yf coordinates of the leftmost eye
        xf = int(xml_face.attributes["xf"].value)
        yf = int(xml_face.attributes["yf"].value)
        # xs, ys coordinates of the rightmost eye
        xs = int(xml_face.attributes["xs"].value)
        ys = int(xml_face.attributes["ys"].value)
        # xm, ym coordinates of the mouth
        xm = int(xml_face.attributes["xm"].value)
        ym = int(xml_face.attributes["ym"].value)

        faces[file_name] = np.array([xf, yf, xs, ys, xm, ym])
    return faces


def compute_rigid_transform(ref_points, points):
    """Computes rotation, scale and trranslation for aligning points to ref_points

    Args:
        ref_points (_type_): _description_
        points (_type_): _description_
    """
    A = np.array(
        [
            [points[0], -points[1], 1, 0],
            [points[1], points[0], 0, 1],
            [points[2], -points[3], 1, 0],
            [points[3], points[2], 0, 1],
            [points[4], -points[5], 1, 0],
            [points[5], points[4], 0, 1],
        ]
    )

    y = np.array(
        [
            ref_points[0],
            ref_points[1],
            ref_points[2],
            ref_points[3],
            ref_points[4],
            ref_points[5],
        ]
    )

    # least sq sol to minimize ||Ax - y||
    a, b, tx, ty = linalg.lstsq(A, y)[0]
    R = np.array([[a, -b], [b, a]])

    return R, tx, ty  # rotated array as well as translation in x and y directions


def rigid_alignment(faces, path, plot=False):
    """Align images rigidly and save as new images.
    "path" determines where the aligned images are saved.
    Set plot=True to plot the images.

    Args:
        faces (_type_): _description_
        path (_type_): _description_
        plot (bool, optional): _description_. Defaults to False.
    """
    # take the points in first image as reference points
    ref_points = list(faces.values())[0]

    # warp each image using affine transform
    for face in faces:
        points = faces[face]

        R, tx, ty = compute_rigid_transform(ref_points=ref_points, points=points)
        T = np.array([[R[1][1], R[1][0]], [R[0][1], R[0][0]]])

        im = np.array(Image.open(os.path.join(path, face)))
        im2 = np.zeros(im.shape, "uint8")

        # warp each color channel
        for i in range(len(im.shape)):
            im2[:, :, i] = ndimage.affine_transform(
                im[:, :, i], linalg.inv(T), offset=[-ty, -tx]
            )

        if plot:
            plt.imshow(im2)
            plt.show()

        # crop away border and save aligned images
        h, w = im2.shape[:2]
        border = int((w + h) / 20)

        # crop away border
        # imsave(os.path.join(path, 'aligned/' + face), im2[border: h - border, border:w - border, :])
        align_path = os.path.join(path, "aligned")
        if not os.path.exists(align_path):
            os.makedirs(align_path)
        save_path = os.path.join(align_path, face)
        im2 = im2[border : h - border, border : w - border, :]
        im2_pil = Image.fromarray(im2)
        im2_pil.save(save_path)


if __name__ == "__main__":
    xml_path = "/home/ekagra/personal/projects/ComputerVision/data/jkfaces.xml"
    out_path = "/home/ekagra/personal/projects/ComputerVision/data/jkfaces"
    points = read_points_from_xml(xmlFileName=xml_path)
    print(list(points)[0])
    # register
    rigid_alignment(points, out_path)
