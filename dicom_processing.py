import tempfile
import nibabel as nib
import numpy as np
import torch
import os
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
from typing import Optional, Any, Mapping, Hashable, List

import monai
from monai.config import print_config
from monai.utils import first
from monai.config import KeysCollection
from monai.data import Dataset, ArrayDataset, create_test_image_3d, DataLoader
from monai.transforms import (
    Transform,
    MapTransform,
    Randomizable,
    # AddChannel,
    # AddChanneld,
    Compose,
    LoadImage,
    LoadImaged,
    Lambda,
    Lambdad,
    RandSpatialCrop,
    RandSpatialCropd,
    ToTensor,
    ToTensord,
    Orientation,
    Rotate
)
from glob import glob
from monai.transforms import (
    Transform,
    MapTransform,
)
from typing import Optional, Any, Mapping, Hashable, List
from monai.config import KeysCollection
import numpy as np
from omegaconf import DictConfig, OmegaConf
import hydra
import matplotlib.pyplot as plt
import pydicom
from utils import contour2mask
from xml.etree import ElementTree as ET
from utils import prettify

def load_dicom(dicom_properties,  dicom_dir, image_only=False):
    """
    Supports only MR for scan, and PR for contours, modalities
    Scan will be of shape (Time, Zs, Height, Width). The dicom series_description should be subtraction
    :param dicom_properties:
    :param dicom_dir:
    :param image_only:
    :return:
    """
    data_out = dict()
    MR_data = dict()
    PR_data = dict()
    data = dict()

    # for path in glob(dir_path+'/*.dcm'):
    paths = glob(f"{dicom_dir}/*.dcm")

    for path in paths:
        # dcm_file_dataset = pydicom.read_file(path)
        dcm_file_dataset = pydicom.dcmread(path)
        series_description = dcm_file_dataset.SeriesDescription
        if dcm_file_dataset.Modality == dicom_properties.modalities.mr:#if file is mr. mr is the regular scan and pr is the scan with contour
            # Save a table of the dicom data for extracting metadata from it later
            if not series_description in data:
                MR_data[series_description] = dict()

                data[series_description] = {
                    dicom_properties.my_keys.dcm_data_key: dcm_file_dataset,
                    dicom_properties.fields.acquisition_time: set(),
                    dicom_properties.fields.slice_location: set()
                }
            data[series_description][dicom_properties.fields.acquisition_time].add(dcm_file_dataset.AcquisitionTime)
            data[series_description][dicom_properties.fields.slice_location].add(dcm_file_dataset.SliceLocation)
            MR_data[series_description][dcm_file_dataset[dicom_properties.fields.SOP_instance_UID].value] = {
                dicom_properties.fields.acquisition_time: dcm_file_dataset.AcquisitionTime,
                dicom_properties.fields.slice_location: dcm_file_dataset.SliceLocation,
                dicom_properties.fields.pixel_array: dcm_file_dataset.pixel_array
            }
        elif dcm_file_dataset.Modality == dicom_properties.modalities.pr:# if the dcm file has contour inside of it (pr)(psg file)
            if not series_description in PR_data:#series_description is not already a key in the dict PR_data, create it (initialize an empty nested dict).
                PR_data[series_description] = dict()
            if (0x0070, 0x0001) in dcm_file_dataset:
                PR_data[series_description][dcm_file_dataset[(0x0008, 0x1115)][0][(0x0008, 0x1140)][0][(0x0008, 0x1155)].value] = \
                    dcm_file_dataset[(0x0070, 0x0001)][0][(0x0070, 0x0009)][0][(0x0070, 0x0022)].value
            else:
                PR_data[series_description][dcm_file_dataset[(0x0008, 0x1115)][0][(0x0008, 0x1140)][0][(0x0008, 0x1155)].value] = None

    for series_description in data:
        slice_locations = sorted(list(data[series_description][dicom_properties.fields.slice_location]))
        acquisition_times = sorted(list(data[series_description][dicom_properties.fields.acquisition_time]))
        height = data[series_description][dicom_properties.my_keys.dcm_data_key][dicom_properties.fields.rows].value
        width = data[series_description][dicom_properties.my_keys.dcm_data_key][dicom_properties.fields.columns].value
        scan = np.empty((len(acquisition_times), len(slice_locations), height, width), dtype=np.int16)
        seg = np.empty((len(acquisition_times), len(slice_locations), height, width), dtype=np.bool)
        for key in MR_data[series_description]:
            slice_location = slice_locations.index(MR_data[series_description][key][dicom_properties.fields.slice_location])
            acquisition_time = acquisition_times.index(MR_data[series_description][key][dicom_properties.fields.acquisition_time])
            scan[acquisition_time, slice_location,:,:] = MR_data[series_description][key][dicom_properties.fields.pixel_array]
            if key not in PR_data[series_description] or PR_data[series_description][key] is None:
                seg[acquisition_time, slice_location, :, :] = np.zeros((height, width), dtype=np.bool)
            else:
                contour = PR_data[series_description][key]
                seg[acquisition_time, slice_location, :, :] = contour2mask(contour, (height, width))
        data_out[dicom_properties.my_keys.scan_key] = scan
        if np.sum(seg) != 0:
            data_out[dicom_properties.my_keys.seg_key] = seg
        if not image_only:
            metadata = dict()

            for field in dicom_properties.metadata_fields.keys():
                key = dicom_properties.metadata_fields[field]
                metadata[field] = data[series_description][dicom_properties.my_keys.dcm_data_key][key].value
            data_out[dicom_properties.my_keys.metadata_key] = metadata
    return data_out


def write_xml(xml_properties, xml_dst_path, accession, label, bndbox, metadata):
    """
    writes an  xml file containing all the information needed for preprocess4model.py.
    :param xml_properties: according to conf/xml
    :param xml_dst_path:
    :param accession:
    :param label: tumor or benign
    :param bndbox: list of [dim{i}_min, dim{i}_max] for dim in len(seg.shape)] representing the roi
    :param metadata:
    :return:
    """
    # name of root tag is data
    root = ET.Element(xml_properties.my_keys.root_key)

    # Adding a subtag named `Opening`
    # inside our root tag
    accession_element = ET.SubElement(root, xml_properties.my_keys.accession_key)
    accession_element.text = str(accession)


    bndbox_element = ET.SubElement(root, xml_properties.my_keys.bndbox_key)
    # The reason that we have the label inside of the bndbox element is because an accession might have
    # multiple bndboxes with different labels. We want our xml file to support that functionality
    # though that functionality isn't available and should be used
    label_element = ET.SubElement(bndbox_element, xml_properties.my_keys.label_key)
    label_element.text = label
    dims_element = ET.SubElement(bndbox_element, xml_properties.my_keys.dims_key)
    for dim, [dim_min, dim_max] in enumerate(bndbox):
        dim_min_element = ET.SubElement(dims_element, f"dim{dim}_min")
        dim_max_element = ET.SubElement(dims_element, f"dim{dim}_max")
        dim_min_element.text = str(dim_min)
        dim_max_element.text = str(dim_max)

    # metadata_
    metadata_element = ET.SubElement(root, xml_properties.my_keys.metadata_key)
    for data in metadata:
        data_element = ET.SubElement(metadata_element, data)
        data_element.text = str(metadata[data])

    prettified_xmlStr = prettify(root)
    output_file = open(xml_dst_path, "w")
    output_file.write(prettified_xmlStr)
    output_file.close()

#
def get_bndbox(seg):
    """
    :param seg: array that represent the roi as a mask
    :return: return list of [dim{i}_min, dim{i}_max] for dim in len(seg.shape)]
    """
    vs = np.indices(seg.shape)
    bndbox = []
    for v in vs:
        v *= seg
        v[seg==False] = -1
        bndbox.append([np.min(v[v != -1]), np.max(v[v != -1])])
    return bndbox


@hydra.main(config_path='conf', config_name='config')
def main(cfg: DictConfig) -> None:
    for accession in cfg.dicom.data.benign.accessions: #from dicom_info.yaml
        print('accession: ', accession)
        # load all relevant data from the dicoms files.
        data = load_dicom(cfg.dicom, f"{cfg.dicom.data.benign.path}/{accession}")
        scan = data[cfg.dicom.my_keys.scan_key]
        seg = data[cfg.dicom.my_keys.seg_key]
        metadata = data[cfg.dicom.my_keys.metadata_key]
        label = cfg.dicom.data.benign.label
        bndbox = get_bndbox(seg)
        xml_dst_path = f"{cfg.xml.data.path}/{accession}.xml"

        write_xml(cfg.xml, xml_dst_path, accession, label, bndbox, metadata)
        np.save(f"{cfg.numpy.data.path}/{accession}.npy", scan)

    for accession in cfg.dicom.data.tumor.accessions:
        print('accession: ', accession)
        # load all relevant data from the dicoms files.
        data = load_dicom(cfg.dicom, f"{cfg.dicom.data.tumor.path}/{accession}")
        scan = data[cfg.dicom.my_keys.scan_key]
        seg = data[cfg.dicom.my_keys.seg_key]
        metadata = data[cfg.dicom.my_keys.metadata_key]
        label = cfg.dicom.data.tumor.label
        bndbox = get_bndbox(seg)
        xml_dst_path = f"{cfg.xml.data.path}/{accession}.xml"

        write_xml(cfg.xml, xml_dst_path, accession, label, bndbox, metadata)
        np.save(f"{cfg.numpy.data.path}/{accession}.npy", scan)


if __name__=="__main__":
    main()
    print('end')


