import hydra
from omegaconf import DictConfig, OmegaConf
from glob import glob
import xml.etree.ElementTree as ET
import numpy as np
from monai.transforms import Compose
import os
from utils import prettify
from imageio import imwrite

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




def get_micro_macro(arr, dims, micro_macro):
    """

    :param arr: the scan from which we'll crop the micro (roi) and macro (area around the roi)
    :param dims: [tmin, tmax, zmin, zmax, ymin, ymax, xmin, xmax] representing the roi
    :param micro_macro: configurations according to conf/micro_macro
    :return:
    """

    pad_width = np.max(micro_macro.array_info.macro_shape)
    padded_arr = np.pad(arr, pad_width=((0, 0), (0, 0), (pad_width, pad_width), (pad_width, pad_width)))
    tmin, tmax, zmin, zmax, ymin, ymax, xmin, xmax = dims
    micro = arr[:, zmin: zmax+1, ymin: ymax+1, xmin: xmax+1]

    # get macro array:
    macro_height, macro_width = micro_macro.array_info.macro_shape
    macro_ymin = ((ymin + ymax) // 2 + pad_width) - macro_height // 2
    macro_ymax = macro_ymin + macro_height
    macro_xmin = ((xmin + xmax) // 2 + pad_width) - macro_width // 2
    macro_xmax = macro_xmin + macro_width
    macro = padded_arr[:, zmin: zmax+1, macro_ymin: macro_ymax, macro_xmin: macro_xmax]

    micro_transforms = Compose([hydra.utils.instantiate(conf) for _, conf in micro_macro.transforms.micro.items()])
    macro_transforms = Compose([hydra.utils.instantiate(conf) for _, conf in micro_macro.transforms.micro.items()])

    micro = micro_transforms(micro)
    macro = macro_transforms(macro)

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
    benign_dirs = [cfg.dicom.data.benign_old.label, cfg.dicom.data.benign_new.label]
    tumor_dirs  = [cfg.dicom.data.tumor_old.label, cfg.dicom.data.tumor_new.label]

    for b in benign_dirs:
        build_dir(f"{cfg.micro_macro.data.path}/{b}")
    for t in tumor_dirs:
        build_dir(f"{cfg.micro_macro.data.path}/{t}")
        
    for accession in accessions:
        xml_path = f"{xml_dir}/{accession}.xml"
        # The new data is: Accession_A so i will try to catch it too.
        if not os.path.exists(xml_path):
            xml_path = f"{xml_dir}/{accession}_A.xml"
        numpy_path = f"{numpy_dir}/{accession}.npy"
        # The new data is: Accession_A so i will try to catch it too.
        if not os.path.exists(numpy_path):
            numpy_path = f"{numpy_dir}/{accession}_A.npy"
        arr = np.load(numpy_path)
        bndboxes = load_bndboxes(xml_path, cfg.xml)
        #TODO: maybe add the case of multiple bndboxes on the same accession, right now we don't support it
        bndbox = bndboxes[0]
        dims, label = bndbox
        # Extract the manufactor from the XML
        tree = ET.parse(xml_path)
        root = tree.getroot()
        machine_type = root.findtext("metadata/machine_type") or "Unknown"
        
        micro, macro = get_micro_macro(arr, dims, cfg.micro_macro)
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
            imwrite(visualize_micro_path, np.interp(micro, (micro.min(), micro.max()), (0, 255)).astype(np.uint8))
            imwrite(visualize_macro_path, np.interp(macro, (macro.min(), macro.max()), (0, 255)).astype(np.uint8))
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