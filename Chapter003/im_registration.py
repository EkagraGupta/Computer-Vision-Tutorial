from xml.dom import minidom
import numpy as np

def read_points_from_xml(xmlFileName):
    """Reads control points for face alignment

    Args:
        xmlFileName (_type_): _description_
    """

    xml_doc = minidom.parse(xmlFileName)
    face_list = xml_doc.getElementsByTagName('face')
    faces = {}

    for xml_face in face_list:
        file_name = xml_face.attributes['file'].value
        # xf, yf coordinates of the leftmost eye
        xf = int(xml_face.attributes['xf'].value) 
        yf = int(xml_face.attributes['yf'].value)
        # xs, ys coordinates of the rightmost eye
        xs = int(xml_face.attributes['xs'].value)
        ys = int(xml_face.attributes['ys'].value)
        # xm, ym coordinates of the mouth
        xm = int(xml_face.attributes['xm'].value)
        ym = int(xml_face.attributes['ym'].value)

        faces[file_name] = np.array([xf, yf, xs, ys, xm, ym])
    return faces


if __name__=='__main__':
    xml_path = '/home/ekagra/personal/projects/ComputerVision/data/jkfaces.xml'
    faces = read_points_from_xml(xmlFileName=xml_path)
    print(list(faces.items())[:1])