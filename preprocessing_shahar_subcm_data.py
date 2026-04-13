"""In this file, The input is 
- Lesion shachar marked on subcm cases 
- Original scans of those accessions (from pacs).

The pipeline is:
1. take the post_sub folder and extract the t2 from it (if already extracted, like: t1_fl3d_spair_tra_x 4 POST_SUB_Series0012 then we will take the extracted.)
2. padd and transform the dcm files in this folder (to make shahar lesions fit the scan accurate)



as a start:
1. open new folders accession_icrf in this path:
/media/breacil/Data/NEW/Early Detection/data_with_segmentation
and copied there the original data scan post sub (or post sub t2 if exist).

2. from each scan folder i will create new folder that do the transform and rotation to be as the lesions that shahar marked (after registration).


"""

INPUT_NII_FOLDER  =  "/mnt/breacil/Results/Early_Detection/segmentations_icrf/test_bad_cases" #"/media/breacil/Results/Early_Detection/segmentations_icrf"
OUTPUT_PROCESSED_NII_FOLDER = "/mnt/breacil/Results/Early_Detection/segmentations_icrf/test_results_bad_cases" #"/media/breacil/Results/Early_Detection/segmentations_icrf_processed"

INPUT_FOLDER_original_dcm = "/mnt/breacil/Data/NEW/Early Detection/data_with_segmentation/try_again/try_tast_8" #"/media/breacil/Data/NEW/Early Detection/data_with_segmentation/try_again" #"/media/breacil/Data/NEW/Early Detection/data_with_segmentation" #The transformed dcm also saved here with _process addition
import os
import pydicom
import numpy as np
import nibabel as nib


TARGET_SIZE = (512, 512)
ROTATE_K = 3   # rotate 90° CW


def pad_to_size(img, target_size):
    h, w = img.shape
    target_h, target_w = target_size

    pad_h = target_h - h
    pad_w = target_w - w

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    return np.pad(
        img,
        [(pad_top, pad_bottom), (pad_left, pad_right)],
        mode="constant",
        constant_values=0,
    )


def process_dicom(in_path, out_path):
    ds = pydicom.dcmread(in_path)
    arr = ds.pixel_array

    # ---- PRINT ORIGINAL SIZE ----
    print(f"\nDICOM: {os.path.basename(in_path)}")
    print("  Original size:", arr.shape)
    
    # 1) Rotate
    arr = np.rot90(arr, k=ROTATE_K)

    # 2) Pad to 512x512
    arr = pad_to_size(arr, TARGET_SIZE).astype(arr.dtype)

    # 3) Flip vertically
    arr = np.flip(arr, axis=0)
    # ---- PRINT FINAL SIZE ----
    print("  Final size   :", arr.shape)
    ds.Rows, ds.Columns = arr.shape
    ds.PixelData = arr.tobytes()

    ds.save_as(out_path)
    print(f"✔ Saved processed: {out_path}")


def process_all_icrf_folders(root_folder):
    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)

        # Only folders that contain _icrf
        if not os.path.isdir(folder_path):
            continue
        if "_icrf" not in folder_name:
            continue

        print(f"\n=== Searching inside {folder_path} ===")

        # Each subfolder inside the _icrf folder is a sequence (e.g. MRI series)
        for seq_name in os.listdir(folder_path):
            seq_path = os.path.join(folder_path, seq_name)

            if not os.path.isdir(seq_path):
                continue

            # Output folder next to original folder
            out_folder = seq_path + "_processed"
            os.makedirs(out_folder, exist_ok=True)
            print(f"→ Processing sequence: {seq_name}")
            print(f"→ Output folder: {out_folder}")

            # Process all DICOMs inside seq folder
            for fname in os.listdir(seq_path):
                if fname.lower().endswith(".dcm"):
                    in_path = os.path.join(seq_path, fname)
                    out_path = os.path.join(out_folder, fname)
                    try:
                        process_dicom(in_path, out_path)
                    except Exception as e:
                        print(f"❌ Error processing {in_path}: {e}")

def transform_nii(in_path, out_path):
    # Load original NIfTI
    nii = nib.load(in_path)
    data = nii.get_fdata()
    data = data.astype(nii.get_data_dtype())  # preserve dtype

    print(f"Processing: {os.path.basename(in_path)}")
    print("Original shape:", data.shape)

    # 1) Flip vertically (axis 0)
    data_flipped = data[::-1, :, :]

    # 2) Rotate 90° CCW in (0,1) plane
    data_rot = np.rot90(data_flipped, k=1, axes=(0, 1))

    print("Transformed shape:", data_rot.shape)

    # Build new NIfTI object
    new_header = nii.header.copy()
    new_header.set_data_shape(data_rot.shape)

    new_nii = nib.Nifti1Image(data_rot, nii.affine, header=new_header)

    # Save
    nib.save(new_nii, out_path)
    print(f"✔ Saved transformed: {out_path}\n")



# Run pad and transform all dcm in scan
process_all_icrf_folders(INPUT_FOLDER_original_dcm)
# flip and rotate all lesions in this path: 
# /media/breacil/Results/Early_Detection/segmentations_icrf
#part of the lesion were manually fixed and saved in this path on t2:
#/media/breacil/Results/Early_Detection/post_sub_with_seperation_of_t
# --------------------------
# Process each NIfTI in folder
# --------------------------
for fname in os.listdir(INPUT_NII_FOLDER):
    if fname.lower().endswith(".nii") or fname.lower().endswith(".nii.gz"):
        # if "4015011883147" in fname:
        in_path  = os.path.join(INPUT_NII_FOLDER, fname)
        out_path = os.path.join(OUTPUT_PROCESSED_NII_FOLDER, fname)  # same name in processed folder
    
        try:
            transform_nii(in_path, out_path)
        except Exception as e:
            print(f"❌ Error processing {fname}: {e}")
            
            
# DICOM_FOLDER = "/media/breacil/Data/NEW/Early Detection/data_with_segmentation/4015011264327_icrf/t1_fl3d_dixon_tra_p2_x_4_POST_W_SUB_14_processed"
# SEG_NII_PATH = "/media/breacil/Results/Early_Detection/segmentations_icrf_processed/4015011264327.nii"
