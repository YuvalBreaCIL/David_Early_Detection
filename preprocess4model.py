import hydra
from omegaconf import DictConfig, OmegaConf
from glob import glob
import xml.etree.ElementTree as ET
import numpy as np
from monai.transforms import Compose
import os
from utils import prettify
from imageio import imwrite
import matplotlib.pyplot as plt
def load_bndboxes(xml_path, xml):
    """
    This function will need to be custom built by the user in any case of change of the xml file structure, it assumes
    this structure example (it only cares about the 'bndbox' tag:
    <?xml version="1.0" ?>
    <annotation>
        <accession>4015008588643</accession>
        <bndbox>
            <label>benign</label>
            <dims>
                <dim0_min>1</dim0_min>
                <dim0_max>1</dim0_max>
                <dim1_min>69</dim1_min>
                <dim1_max>71</dim1_max>
                <dim2_min>284</dim2_min>
                <dim2_max>330</dim2_max>
                <dim3_min>130</dim3_min>
                <dim3_max>137</dim3_max>
            </dims>
        </bndbox>
        <metadata>
            <modalities_in_study>['MR', 'SR', 'PR']</modalities_in_study>
            <study_description>MRI BREASTS</study_description>
            <series_description>sub</series_description>
            <slice_thickness>2</slice_thickness>
            <spacing_between_slices>2</spacing_between_slices>
            <patient_position>FFP</patient_position>
            <pixel_spacing>[0.6641, 0.6641]</pixel_spacing>
        </metadata>
    </annotation>

    :param xml_path:
    :param xml: xml info from config dir xml
    :return: list of bndbox that contain the bndbox dims and the label
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    bndboxes = []
    for bndbox in root.findall(xml.my_keys.bndbox_key):
        print(bndbox)
        label = bndbox.find(xml.my_keys.label_key).text
        dims = [int(dim.text) for dim in bndbox.find(xml.my_keys.dims_key)]
        bndboxes.append([dims, label])

    return bndboxes



def choose_z_range_by_signal(arr, zmin, zmax, ymin, ymax, xmin, xmax, t=1):
    Z = arr.shape[1]
    zmin_f = (Z - 1) - zmax
    zmax_f = (Z - 1) - zmin

    patch_orig = arr[t, zmin:zmax+1, ymin:ymax+1, xmin:xmax+1]
    patch_flip = arr[t, zmin_f:zmax_f+1, ymin:ymax+1, xmin:xmax+1]

    # score - אפשר לשנות ל-percentile אם יש רעש
    s_orig = np.percentile(patch_orig, 99)
    s_flip = np.percentile(patch_flip, 99)

    return (zmin, zmax) if s_orig >= s_flip else (zmin_f, zmax_f)
def bbox_from_seg(seg_t: np.ndarray, margin_yx: int = 0):
    """
    seg_t shape: (Z, Y, X) boolean
    returns: zmin,zmax,ymin,ymax,xmin,xmax
    """
    zs, ys, xs = np.where(seg_t > 0)
    if zs.size == 0:
        return None

    zmin, zmax = int(zs.min()), int(zs.max())
    ymin, ymax = int(ys.min()), int(ys.max())
    xmin, xmax = int(xs.min()), int(xs.max())

    # optional margin in XY
    ymin = max(0, ymin - margin_yx)
    xmin = max(0, xmin - margin_yx)
    ymax = min(seg_t.shape[1] - 1, ymax + margin_yx)
    xmax = min(seg_t.shape[2] - 1, xmax + margin_yx)

    return zmin, zmax, ymin, ymax, xmin, xmax
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Union

def load_bboxes_from_excel( xlsx_path: Union[str, Path],
    accession: Union[str, int],
) -> Tuple[int, int, int, int]:
    """
    Returns a dict:
      { accession_str : (bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max) }

    Excel columns expected:
      accessions, bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max
    """
    df = pd.read_excel(xlsx_path)

    # normalize accession column
    df["accessions"] = (
        df["accessions"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    )
    acc = str(accession).strip().replace(".0", "")

    # make bbox columns numeric (NaN stays NaN)
    bbox_cols = ["bbox_x_min", "bbox_y_min", "bbox_x_max", "bbox_y_max"]
    for c in bbox_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # keep only rows where ALL bbox values exist
    df = df.dropna(subset=bbox_cols)

    # now safe to cast to int
    df[bbox_cols] = df[bbox_cols].astype(int)

    # filter to the wanted accession
    row = df.loc[df["accessions"] == acc]
    if row.empty:
        raise KeyError(f"Accession {acc} not found (or missing bbox values) in {xlsx_path}")

    x_min = int(row.iloc[0]["bbox_x_min"])
    y_min = int(row.iloc[0]["bbox_y_min"])
    x_max = int(row.iloc[0]["bbox_x_max"])
    y_max = int(row.iloc[0]["bbox_y_max"])
    print (f"bnb box from excel is (x_min,y_min,x_max,y_max):{x_min,y_min,x_max,y_max}")
    return x_min, y_min, x_max, y_max


def get_bbox_for_accession(
    xlsx_path: Union[str, Path],
    accession: Union[str, int],
) -> Tuple[int, int, int, int]:
    """
    Returns (bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max) for the given accession.
    """
    bboxes = load_bboxes_from_excel(xlsx_path,accession )
    acc = str(accession).strip().replace(".0", "")
    return bboxes





def get_micro_macro(arr, dims, micro_macro,cfg,group, accession):
    """

    :param arr: the scan from which we'll crop the micro (roi) and macro (area around the roi)
    :param dims: [tmin, tmax, zmin, zmax, ymin, ymax, xmin, xmax] representing the roi
    :param micro_macro: configurations according to conf/micro_macro
    :return:
    """
    # print("micro pre-transform shape:", micro.shape)
    # print("macro pre-transform shape:", macro.shape)
    pad_width = np.max(micro_macro.array_info.macro_shape)
    padded_arr = np.pad(arr, pad_width=((0, 0), (0, 0), (pad_width, pad_width), (pad_width, pad_width)))
    tmin, tmax, zmin, zmax, ymin, ymax, xmin, xmax = dims
    if group in {"tumor_subcm", "benign_subcm"}:
        Z = arr.shape[1]  # 160
        zmin_fixed = (Z - 1) - zmax
        zmax_fixed = (Z - 1) - zmin
        # החלפה
        zmin, zmax = zmin_fixed, zmax_fixed
        #load xsl and extract values for bnd box 
        xlsx_path=cfg.dicom.bnd_box_subcm_cases.xslx_path
        bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max = get_bbox_for_accession(xlsx_path, accession)
        print(accession, "->", bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max)
        xmin = bbox_x_min
        ymin = bbox_y_min
        xmax = bbox_x_max
        ymax = bbox_y_max
        
        
    fig, ax = plt.subplots()
    zmid = (zmin + zmax) // 2
    ax.imshow(arr[1, zmid], cmap="gray")
    ax.set_title(f"z={zmid}")

    from matplotlib.patches import Rectangle
    rect = Rectangle((xmin, ymin),
                    xmax-xmin,
                    ymax-ymin,
                    linewidth=2,
                    edgecolor="red",
                    facecolor="none")
    ax.add_patch(rect)
    plt.show()
    
    # ymin=142
    # xmin=185
    # # zmin=
    # ymax=151
    # xmax=196
    # # zmax=
    
    micro = arr[:+1, zmin: zmax+1, ymin: ymax+1, xmin: xmax+1]
    

    # get macro array:
    macro_height, macro_width = micro_macro.array_info.macro_shape
    macro_ymin = ((ymin + ymax) // 2 + pad_width) - macro_height // 2
    macro_ymax = macro_ymin + macro_height
    macro_xmin = ((xmin + xmax) // 2 + pad_width) - macro_width // 2
    macro_xmax = macro_xmin + macro_width
    macro = padded_arr[:, zmin: zmax+1, macro_ymin: macro_ymax, macro_xmin: macro_xmax]
    print("micro pre-transform shape:", micro.shape)
    print("macro pre-transform shape:", macro.shape)
    micro_transforms = Compose([hydra.utils.instantiate(conf) for _, conf in micro_macro.transforms.micro.items()])
    macro_transforms = Compose([hydra.utils.instantiate(conf) for _, conf in micro_macro.transforms.micro.items()])
    micro_macro_subcm_transforms = Compose([hydra.utils.instantiate(conf) for _, conf in micro_macro.transforms.micro_macro_subcm.items()])

    if micro.shape[0] == 1: #if we have only one timepoint (post sub 11 for example)
        # replace "SliceFromArray(dim=0,slice_num=1)" effect with slice 0
        micro = micro_macro_subcm_transforms(micro)
        macro = micro_macro_subcm_transforms(macro)
    else:
        micro = micro_transforms(micro)
        macro = macro_transforms(macro)

    #DEBUG
    print("Z size:", arr.shape[1])
    print("zmin,zmax from XML:", zmin, zmax)
    print("flipped should be:", (arr.shape[1]-1)-zmax, (arr.shape[1]-1)-zmin)
    print("dims:", dims)
    print("micro shape raw:", micro.shape)
    print("macro shape raw:", macro.shape)
    print("macro center (y,x):", (ymin+ymax)//2, (xmin+xmax)//2)
    print("macro ranges y:", macro_ymin, macro_ymax, "x:", macro_xmin, macro_xmax)
    # assumes pad_width = 50 like your run
    micro_y0_in_macro = (ymin + pad_width) - macro_ymin
    micro_y1_in_macro = (ymax + pad_width) - macro_ymin
    micro_x0_in_macro = (xmin + pad_width) - macro_xmin
    micro_x1_in_macro = (xmax + pad_width) - macro_xmin

    print(micro_y0_in_macro, micro_y1_in_macro, micro_x0_in_macro, micro_x1_in_macro)
    return micro, macro



def build_dir(path):
    """
    builds a directory of the path, if exists it won't build
    :param path:
    :return:
    """
    if os.path.isdir(path):
        return
    os.mkdir(path)


def get_accession_group(cfg, accession: str) -> str:
    def acc_list(x):
        return [str(a) for a in (x or [])]

    if accession in acc_list(cfg.dicom.data.tumor_subcm.accessions):
        return "tumor_subcm"
    if accession in acc_list(cfg.dicom.data.benign_subcm.accessions):
        return "benign_subcm"
    if accession in acc_list(cfg.dicom.data.tumor_new.accessions):
        return "tumor_new"
    if accession in acc_list(cfg.dicom.data.tumor_old.accessions):
        return "tumor_old"
    if accession in acc_list(cfg.dicom.data.benign_new.accessions):
        return "benign_new"
    if accession in acc_list(cfg.dicom.data.benign_old.accessions):
        return "benign_old"
    return "unknown"

def to_2d(img: np.ndarray) -> np.ndarray:
    # If (Z,H,W) -> MIP over Z
    if img.ndim == 3:
        return img.max(axis=0)
    # If already (H,W) -> keep
    if img.ndim == 2:
        return img
    raise ValueError(f"Unexpected image ndim={img.ndim}, shape={img.shape}")

@hydra.main(config_path='conf', config_name='config')
def main(cfg: DictConfig) -> None:
    print(cfg)
    xml_dir = cfg.xml.data.path
    numpy_dir = cfg.numpy.data.path
    accessions = [str(accession) for accession in cfg.micro_macro.data.accessions]
    # benign_dir = f"{cfg.micro_macro.data.path}/{cfg.dicom.data.benign.label}"
    # tumor_dir = f"{cfg.micro_macro.data.path}/{cfg.dicom.data.tumor.label}"
    # build_dir(benign_dir)
    # build_dir(tumor_dir)
    # Build both benign dirs and both tumor dirs
    benign_dirs = [cfg.dicom.data.benign_old.label, cfg.dicom.data.benign_new.label,cfg.dicom.data.benign_subcm.label ]
    tumor_dirs  = [cfg.dicom.data.tumor_old.label, cfg.dicom.data.tumor_new.label, cfg.dicom.data.tumor_subcm.label]

    for b in benign_dirs:
        build_dir(f"{cfg.micro_macro.data.path}/{b}")
    for t in tumor_dirs:
        build_dir(f"{cfg.micro_macro.data.path}/{t}")
        
    for accession in accessions:
        # xml_path = f"{xml_dir}/{accession}.xml"
        # # The new data is: Accession_A so i will try to catch it too.
        # if not os.path.exists(xml_path):
        #     xml_path = f"{xml_dir}/{accession}_A.xml"
        # numpy_path = f"{numpy_dir}/{accession}.npy"
        # # The new data is: Accession_A so i will try to catch it too.
        # if not os.path.exists(numpy_path):
        #     numpy_path = f"{numpy_dir}/{accession}_A.npy"
        
         # Possible suffixes to try (in order)
        suffixes = ["", "_A", "_icrf"]

        xml_path = None
        numpy_path = None

        # ---- Find XML ----
        for suf in suffixes:
            candidate = f"{xml_dir}/{accession}{suf}.xml"
            if os.path.exists(candidate):
                xml_path = candidate
                break

        # ---- Find NPY ----
        for suf in suffixes:
            candidate = f"{numpy_dir}/{accession}{suf}.npy"
            if os.path.exists(candidate):
                numpy_path = candidate
                break

        # ---- Optional: handle missing files ----
        if xml_path is None:
            print(f"❌ XML not found for accession {accession}")
            continue

        if numpy_path is None:
            print(f"❌ NPY not found for accession {accession}")
            continue

        # Now you safely have:
        # xml_path
        # numpy_path
        print("Using:")
        print("  XML :", xml_path)
        print("  NPY :", numpy_path)
        arr = np.load(numpy_path)
        bndboxes = load_bndboxes(xml_path, cfg.xml)
        print(f"bndboxes:{bndboxes}")
        #TODO: maybe add the case of multiple bndboxes on the same accession, right now we don't support it
        bndbox = bndboxes[0]
        print(f"bndboxes[0]:{bndboxes[0]}")
        dims, label = bndbox
        print (f"dims, label:{dims, label}")
        # Extract the manufactor from the XML
        tree = ET.parse(xml_path)
        root = tree.getroot()
        machine_type = root.findtext("metadata/machine_type") or "Unknown"
        group = get_accession_group(cfg, accession)
        print(f"Accession {accession} belongs to group: {group}")
        micro, macro = get_micro_macro(arr, dims, cfg.micro_macro,cfg, group,accession)
        micro_macro_dir = f"{cfg.micro_macro.data.path}/{accession}"
        # Build paths
        build_dir(micro_macro_dir)
        micro_path = f"{micro_macro_dir}/{cfg.micro_macro.data.micro_name}.npy"
        macro_path = f"{micro_macro_dir}/{cfg.micro_macro.data.macro_name}.npy"
        label_path = f"{micro_macro_dir}/{cfg.xml.my_keys.label_key}.xml"
        manufacturer_path = f"{micro_macro_dir}/{cfg.xml.my_keys.manufacturer_key}.xml"

        if cfg.micro_macro.visualize.path != "" and not cfg.micro_macro.visualize.path.isspace():
            visualize_micro_path = f"{cfg.micro_macro.visualize.path}/{label}/{accession}_{cfg.micro_macro.data.micro_name}.png"
            visualize_macro_path = f"{cfg.micro_macro.visualize.path}/{label}/{accession}_{cfg.micro_macro.data.macro_name}.png"
            # imwrite(visualize_micro_path, np.interp(micro, (micro.min(), micro.max()), (0, 255)).astype(np.uint8))
            # imwrite(visualize_macro_path, np.interp(macro, (macro.min(), macro.max()), (0, 255)).astype(np.uint8))
            micro_2d = to_2d(micro)
            macro_2d = to_2d(macro)

            imwrite(visualize_micro_path, np.interp(micro_2d, (micro_2d.min(), micro_2d.max()), (0, 255)).astype(np.uint8))
            imwrite(visualize_macro_path, np.interp(macro_2d, (macro_2d.min(), macro_2d.max()), (0, 255)).astype(np.uint8))
        # Create label.xml
        label_element = ET.Element(cfg.xml.my_keys.label_key)
        label_element.text = label
        prettified_xmlStr = prettify(label_element)
        output_file = open(label_path, "w")
        output_file.write(prettified_xmlStr)
        output_file.close()
        
        # Add manufacturer as xml file
        manufacturer_element = ET.Element(cfg.xml.my_keys.manufacturer_key)
        manufacturer_element.text = machine_type
        prettified_xmlStr = prettify(manufacturer_element)
        output_file = open(manufacturer_path, "w")
        output_file.write(prettified_xmlStr)
        output_file.close()
        np.save(micro_path, micro)
        np.save(macro_path, macro)



if __name__=="__main__":
    main()
    print('end')