import numpy as np
from skimage.draw import polygon2mask
from xml.etree import ElementTree as ET
from xml.dom import minidom



def contour2mask(contour, shape):
    """
    takes in a contour (like the one in dicom files with PSg modality) and returns
    a mask of the contour of the given shape
    :param contour:
    :param shape:
    :return:
    """
    xs, ys = np.round(contour[::2]).astype('int'), np.round(contour[1::2]).astype('int')
    polygon = np.stack((ys, xs), axis=1)
    mask = polygon2mask(shape, polygon)
    return mask


def prettify(xmlStr):
    """
    Return a pretty-printed XML string for the Element.
    :param xmlStr:
    :return:
    """
    INDENT = "    "
    rough_string = ET.tostring(xmlStr, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent=INDENT)

