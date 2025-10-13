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

def find_nii(dicom_dir):
    """ מחפש קובץ .nii או .nii.gz בנתיב הנתון. מחזיר את הנתיב הראשון שמוצא או None. """
    cands = glob(os.path.join(dicom_dir, "*.nii")) + glob(os.path.join(dicom_dir, "*.nii.gz"))
    return cands[0] if cands else None


def is_dicom_file(path: str) -> bool:
    """Return True if file at path looks like a DICOM (with or without extension)."""
    try:
        # Quick header check: 'DICM' at byte 128–131 (many DICOMs have it)
        with open(path, "rb") as f:
            preamble = f.read(132)
            if len(preamble) >= 132 and preamble[128:132] == b"DICM":
                return True
        # Some valid DICOMs don't have the 'DICM' marker → try reading the header
        pydicom.dcmread(path, stop_before_pixels=True, force=True)
        return True
    except Exception:
        return False
def create_paths_from_new_data(dicom_dir):
    if not os.path.exists(dicom_dir):
        dicom_dir= dicom_dir.removesuffix("_A")
    target_folders = ["t1_fl3d_spair_tra_x 4 POST_SUB","t1_fl3d_dixon_tra_p2 x 4 POST_W_SUB"]
    search_paths= []
    # print (f"target_folder is {target_folder}")
    for folder in target_folders:
        # full path to that folder
        target_path = os.path.join(dicom_dir, folder)
        if os.path.exists(target_path):
            print(f"Found specific target folder: {folder}")
            search_paths = [target_path]

    # print (f"traget_path is {target_path}")
    if not search_paths: #if there is not t1_fl3d_spair_tra_x 4 POST_SUB folder but there are folders with "_SUB_Series00" in their name
        search_paths = []
        for root, dirs, _ in os.walk(dicom_dir):
            for d in dirs:
                if "_SUB_Series00" in d and "dyn" not in d:#If there are folders with "_SUB_Series00" in their name
                    search_paths.append(os.path.join(root, d))
                elif ")-(" in d and ".." not in d:
                    if ("_Series0500" in d or "_Series0600" in d): #(16124_5_117)-(16124_5_1)_Series0500 or (837_6_117)-(837_6_1)_Series0600 
                        search_paths.append(os.path.join(root, d))

                        
                    
        if not search_paths:
            print("❌ No folder found with target name or '_SUB_Series00'")
            return []
    
    dicom_paths = []
    # Add all dicom files
    for spath in search_paths:
        print(f"Scanning folder: {spath}")
        for root, _, files in os.walk(spath):
            for fn in files:
                full = os.path.join(root, fn)
                if is_dicom_file(full) and "Zone.Identifier" not in full:
                    dicom_paths.append(full)

    print(len(dicom_paths), "DICOM files found (by content)")
    return dicom_paths

def process_paths(paths, dicom_properties, data, MR_data, PR_data, dicom_dir, rotate_new_data=False):
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
                
                #Get pixel array and rotate it if needed for new data
                pixel_array = dcm_file_dataset.pixel_array
                if rotate_new_data:
                    pixel_array = np.rot90(pixel_array, k=2)  # 180 degree rotation
                
                
                MR_data[series_description][dcm_file_dataset[dicom_properties.fields.SOP_instance_UID].value] = {
                    dicom_properties.fields.acquisition_time: dcm_file_dataset.AcquisitionTime,
                    dicom_properties.fields.slice_location: dcm_file_dataset.SliceLocation,
                    # dicom_properties.fields.pixel_array: dcm_file_dataset.pixel_array
                    dicom_properties.fields.pixel_array: pixel_array

                }
            elif dcm_file_dataset.Modality == dicom_properties.modalities.pr:# if the dcm file has contour inside of it (pr)(psg file)
                if not series_description in PR_data:#series_description is not already a key in the dict PR_data, create it (initialize an empty nested dict).
                    PR_data[series_description] = dict()
                if (0x0070, 0x0001) in dcm_file_dataset:
                    PR_data[series_description][dcm_file_dataset[(0x0008, 0x1115)][0][(0x0008, 0x1140)][0][(0x0008, 0x1155)].value] = \
                        dcm_file_dataset[(0x0070, 0x0001)][0][(0x0070, 0x0009)][0][(0x0070, 0x0022)].value
                else:
                    PR_data[series_description][dcm_file_dataset[(0x0008, 0x1115)][0][(0x0008, 0x1140)][0][(0x0008, 0x1155)].value] = None
    pass
                
def data_new_format(dicom_properties, dicom_dir, data, MR_data, nii_path, image_only=False, rotated_nii_data=None):
    """
    Handle new data format with .nii segmentation
    """
    data_out = dict()
    
    for series_description in data:
        slice_locations = sorted(list(data[series_description][dicom_properties.fields.slice_location]))
        acquisition_times = sorted(list(data[series_description][dicom_properties.fields.acquisition_time]))
        height = data[series_description][dicom_properties.my_keys.dcm_data_key][dicom_properties.fields.rows].value
        width = data[series_description][dicom_properties.my_keys.dcm_data_key][dicom_properties.fields.columns].value
        
        # Create empty 4d numpy arrays
        scan = np.empty((len(acquisition_times), len(slice_locations), height, width), dtype=np.int16)
        seg = np.empty((len(acquisition_times), len(slice_locations), height, width), dtype=np.bool)
        
        # Fill scan array from MR data
        for key in MR_data[series_description]:
            slice_location = slice_locations.index(MR_data[series_description][key][dicom_properties.fields.slice_location])
            acquisition_time = acquisition_times.index(MR_data[series_description][key][dicom_properties.fields.acquisition_time])
            scan[acquisition_time, slice_location,:,:] = MR_data[series_description][key][dicom_properties.fields.pixel_array]
        
        # Load .nii segmentation
        try:
            nii_img = nib.load(nii_path)
            nii_data = nii_img.get_fdata()
            # Rotate .nii data 180 degrees to match DICOM rotation
            nii_data = np.rot90(nii_data, k=2, axes=(0, 1))
            nii_seg = (nii_data > 0).astype(np.bool)
            
            print(f"NII shape: {nii_seg.shape}, Scan shape: {scan.shape}")
            
            # Handle 3D .nii segmentation
            if nii_seg.ndim == 3:
                # Try different axis permutations to match scan dimensions
                # scan shape is (T, Z, H, W), nii might be (H, W, Z) or (Z, H, W) etc.
                target_3d_shape = scan.shape[1:]  # (Z, H, W)
                
                if nii_seg.shape == target_3d_shape:
                    # Direct match - use for all time points
                    for t in range(len(acquisition_times)):
                        seg[t] = nii_seg
                elif nii_seg.shape == (target_3d_shape[2], target_3d_shape[1], target_3d_shape[0]):  # (W, H, Z)
                    nii_seg = nii_seg.transpose(2, 1, 0)  # -> (Z, H, W)
                    for t in range(len(acquisition_times)):
                        seg[t] = nii_seg
                elif nii_seg.shape == (target_3d_shape[1], target_3d_shape[2], target_3d_shape[0]):  # (H, W, Z)
                    nii_seg = nii_seg.transpose(2, 0, 1)  # -> (Z, H, W)
                    for t in range(len(acquisition_times)):
                        seg[t] = nii_seg
                else:
                    print(f"Warning: Cannot match .nii shape {nii_seg.shape} to scan 3D shape {target_3d_shape}")
                    print("Available permutations didn't work. Using zeros.")
                    seg = np.zeros_like(scan, dtype=np.bool)
            elif nii_seg.shape == scan.shape:
                # 4D match
                seg = nii_seg
            else:
                print(f"Warning: .nii shape {nii_seg.shape} doesn't match scan shape {scan.shape}")
                seg = np.zeros_like(scan, dtype=np.bool)
                
        except Exception as e:
            print(f"Error loading .nii file {nii_path}: {e}")
            seg = np.zeros_like(scan, dtype=np.bool)
        data_out[dicom_properties.my_keys.scan_key] = scan
        if np.sum(seg) != 0:
            data_out[dicom_properties.my_keys.seg_key] = seg
            
        if not image_only:
            metadata = dict()
            dcm_dataset = data[series_description][dicom_properties.my_keys.dcm_data_key]
             # Debug: print all available tags
            # print(f"Available DICOM tags in {series_description}:")
            # for tag in dcm_dataset:
            #     try:
            #         print(f"  {tag}: {dcm_dataset[tag].keyword} = {dcm_dataset[tag].value}")
            #     except:
            #         print(f"  {tag}: (unable to read)")
            for field in dicom_properties.metadata_fields.keys():
                key = dicom_properties.metadata_fields[field]
                # metadata[field] = data[series_description][dicom_properties.my_keys.dcm_data_key][key].value
                if key in dcm_dataset:
                    metadata[field] = dcm_dataset[key].value
                elif (field == "SpacingBetweenSlices" or field == "spacing_between_slices") and (0x0018, 0x0050) in dcm_dataset:
                    # Use slice thickness as fallback for new data (they're the same value in old data)
                    metadata[field] = dcm_dataset[(0x0018, 0x0050)].value
                    print(f"Using Slice Thickness as fallback for Spacing Between Slices: {metadata[field]}")
                else:
                    print(f"Warning: DICOM tag {key} not found in {series_description}, setting to None")
                    metadata[field] = None
            data_out[dicom_properties.my_keys.metadata_key] = metadata
    
    return data_out
def rotate_new_data(MR_data, nii_path, dicom_properties):
    """
    Rotate new data (breast up) to match old data orientation (breast down)
    Apply 180-degree rotation to both DICOM pixel arrays and .nii segmentation
    """
    # Rotate all DICOM pixel arrays in MR_data
    for series_description in MR_data:
        for key in MR_data[series_description]:
            # Rotate pixel array 180 degrees (flip both axes)
            MR_data[series_description][key][dicom_properties.fields.pixel_array] = np.rot90(
                MR_data[series_description][key][dicom_properties.fields.pixel_array], k=2
            )
    
    # Rotate .nii file if it exists
    rotated_nii_data = None
    if nii_path:
        try:
            nii_img = nib.load(nii_path)
            nii_data = nii_img.get_fdata()
            # Rotate the .nii data 180 degrees on the appropriate axes
            # Assuming the .nii is in (H, W, Z) format, rotate H and W axes
            rotated_nii_data = np.rot90(nii_data, k=2, axes=(0, 1))
        except Exception as e:
            print(f"Error rotating .nii file: {e}")
            rotated_nii_data = None
    
    return MR_data, rotated_nii_data

def load_dicom(dicom_properties, accession, nii_paths,dicom_dir, image_only=False):
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
    if paths: #if paths not empty,  i.e if its david old data
        process_paths(paths, dicom_properties, data, MR_data, PR_data,dicom_dir,rotate_new_data=False)
    else: #if paths is empty, so this is new data (with subfolders)
        paths= create_paths_from_new_data(dicom_dir)
        process_paths(paths, dicom_properties, data, MR_data, PR_data, dicom_dir,rotate_new_data=True)

    # Initialize nii_path
    # nii_path = None
    # # If there is a .nii file we will 
    # if (glob(os.path.join(dicom_dir, "*.nii"))): # if the file is new (with .nii segmentation)
    #     nii_path = find_nii(dicom_dir)
    # elif:
    nii_path= find_accession_in_nii_paths(accession, nii_paths)
    if nii_path is not None:
        return data_new_format(dicom_properties, dicom_dir, data, MR_data, nii_path, image_only)

    for series_description in data:
        slice_locations = sorted(list(data[series_description][dicom_properties.fields.slice_location]))
        acquisition_times = sorted(list(data[series_description][dicom_properties.fields.acquisition_time]))
        height = data[series_description][dicom_properties.my_keys.dcm_data_key][dicom_properties.fields.rows].value
        width = data[series_description][dicom_properties.my_keys.dcm_data_key][dicom_properties.fields.columns].value

        # Create empty 4d numpy array. 4D: T × Slice × H × W.
        scan = np.empty((len(acquisition_times), len(slice_locations), height, width), dtype=np.int16)
        seg = np.empty((len(acquisition_times), len(slice_locations), height, width), dtype=np.bool) #ROI 
        #PROBLEMATIC CODE FOR NII##############################
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
def debug_old_data_metadata(dicom_properties, old_data_path):
    """
    Debug function to check metadata values in old data
    """
    print("=== DEBUGGING OLD DATA METADATA ===")
    
    paths = glob(f"{old_data_path}/*.dcm")
    if not paths:
        print("No .dcm files found in old data path")
        return
    
    # Process just a few files to see the metadata
    for i, path in enumerate(paths[:5]):  # Check first 5 files
        try:
            dcm_dataset = pydicom.dcmread(path)
            if dcm_dataset.Modality == dicom_properties.modalities.mr:
                print(f"\nFile {i+1}: {os.path.basename(path)}")
                print(f"Series Description: {dcm_dataset.SeriesDescription}")
                
                # Check for spacing between slices
                if (0x0018, 0x0088) in dcm_dataset:
                    spacing_value = dcm_dataset[(0x0018, 0x0088)].value
                    print(f"  Spacing Between Slices (0018,0088): {spacing_value}")
                else:
                    print("  Spacing Between Slices (0018,0088): NOT FOUND")
                
                # Check for slice thickness
                if (0x0018, 0x0050) in dcm_dataset:
                    thickness_value = dcm_dataset[(0x0018, 0x0050)].value
                    print(f"  Slice Thickness (0018,0050): {thickness_value}")
                else:
                    print("  Slice Thickness (0018,0050): NOT FOUND")
                
                break  # Just check first MR file
        except Exception as e:
            print(f"Error reading {path}: {e}")
def create_nii_paths_list(nii_path):
    """
    Function gets a directory path containing NIfTI files (.nii / .nii.gz)
    and returns a list of all file paths.

    Parameters
    ----------
    nii_path : str
        Path to directory containing NIfTI files.

    Returns
    -------
    list of str
        List with absolute paths to all NIfTI files found.
    """
    if not os.path.isdir(nii_path):
        raise NotADirectoryError(f"Path does not exist or is not a directory: {nii_path}")

    # מציאת קבצים עם הסיומות המתאימות
    nii_files = glob(os.path.join(nii_path, "*.nii")) + glob(os.path.join(nii_path, "*.nii.gz"))

    # החזרת נתיבים מלאים (absolute)
    nii_files = [os.path.abspath(f) for f in nii_files]

    print(f"Found {len(nii_files)} NIfTI files in {nii_path}")
    return nii_files

def find_accession_in_nii_paths(accession, nii_paths):
    """
    Check if the given accession appears in any of the provided NIfTI paths.

    Parameters
    ----------
    accession : str or int
        The accession number to search for.
    nii_paths : list of str
        List of file paths to NIfTI files.

    Returns
    -------
    str or None
        The first matching NIfTI path if found, otherwise None.
    """
    accession_str = str(accession).removesuffix("_A")

    for path in nii_paths:
        if accession_str in os.path.basename(path):  # check filename
            return path

    return None
@hydra.main(config_path='conf', config_name='config')
def main(cfg: DictConfig) -> None:
    nii_paths= create_nii_paths_list("/media/breacil/Results/Early_Detection/DCE segmentations_3DSlicer")
     # iterate over all groups in your yaml
    groups = [
        cfg.dicom.data.benign_old,
        cfg.dicom.data.benign_new,
        cfg.dicom.data.tumor_old,
        cfg.dicom.data.tumor_new,
    ]

    for group in groups:
        if not getattr(group, "accessions", None):
            continue
        for accession in group.accessions: #from dicom_info.yaml
            print('accession: ', accession)
            # accession= accession+'_A'
            if group in [cfg.dicom.data.benign_new, cfg.dicom.data.tumor_new]:
                accession = str(accession) + "_A"

            # Check first old accession
            # first_old_accession = accession  # replace with actual ID
            # old_accession_path = f"{cfg.dicom.data.benign.path}/{first_old_accession}"
            # debug_old_data_metadata(cfg.dicom, old_accession_path) #to check if the Spacing Between Slices (0018,0088) and Slice Thickness (0018,0050) are the same 
            # load all relevant data from the dicoms files.
            data = load_dicom(cfg.dicom, accession,nii_paths, f"{group.path}/{accession}")
            print ("finished to load dicom, start to compute scan")
            scan = data[cfg.dicom.my_keys.scan_key]
            print ("finished to compute scan, start to compute segmentation")
            seg = data[cfg.dicom.my_keys.seg_key]
            print ("finished to compute segmentation, start to load metadate")
            metadata = data[cfg.dicom.my_keys.metadata_key]
            print ("finished to load metadate, take label")
            label = group.label
            print ("finished to load label, take bndbox")
            bndbox = get_bndbox(seg)
            print ("finished to bndbox label, take xml_dst_path")
            xml_dst_path = f"{cfg.xml.data.path}/{accession}.xml"

            write_xml(cfg.xml, xml_dst_path, accession, label, bndbox, metadata)
            print ("xml exported")
            np.save(f"{cfg.numpy.data.path}/{accession}.npy", scan)
            print("npy exported to:", cfg.numpy.data.path)

            print ("npy exported")


    # for accession in cfg.dicom.data.tumor.accessions:
    #     print('accession: ', accession)
    #     accession = str(accession) + "_A"

    #     # load all relevant data from the dicoms files.
    #     data = load_dicom(cfg.dicom, accession,nii_paths, f"{cfg.dicom.data.tumor.path}/{accession}")
    #     scan = data[cfg.dicom.my_keys.scan_key]
    #     seg = data[cfg.dicom.my_keys.seg_key]
    #     metadata = data[cfg.dicom.my_keys.metadata_key]
    #     label = cfg.dicom.data.tumor.label
    #     bndbox = get_bndbox(seg)
    #     xml_dst_path = f"{cfg.xml.data.path}/{accession}.xml"

    #     write_xml(cfg.xml, xml_dst_path, accession, label, bndbox, metadata)
    #     np.save(f"{cfg.numpy.data.path}/{accession}.npy", scan)


if __name__=="__main__":
    main()
    print('end')


