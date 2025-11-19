import os
import SimpleITK as sitk
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Paths
ROOT = Path("Downloads/3Dircadb1")   # Raw data from 3D-IRCAD-b1
OUT = Path("3Dircadb1")
OUT.mkdir(exist_ok=True)

def read_dicom_series(path: Path):
    """Reads a DICOM series (folder) as a SimpleITK image."""
    reader = sitk.ImageSeriesReader()
    files = reader.GetGDCMSeriesFileNames(str(path))
    reader.SetFileNames(files)
    return reader.Execute()

def merge_tumors(mask_dir: Path):
    """Combine all tumor masks in MASKS_DICOM into one binary volume."""
    tumor_volume = None
    for sub in os.listdir(mask_dir):
        if "tumor" in sub.lower():
            tumor_path = mask_dir / sub
            if tumor_path.is_dir():
                mask = read_dicom_series(tumor_path)
                tumor_arr = sitk.GetArrayFromImage(mask) > 0
                if tumor_volume is None:
                    tumor_volume = tumor_arr
                else:
                    tumor_volume = np.logical_or(tumor_volume, tumor_arr)
    return tumor_volume

def convert_patient(patient_dir: Path, idx: int):
    """Convert one 3Dircadb1 case to volume + segmentation NIfTI files."""
    pid = f"{idx:02d}"
    ct_dir = patient_dir / "PATIENT_DICOM"
    mask_dir = patient_dir / "MASKS_DICOM"
    liver_dir = mask_dir / "liver"

    # --- CT volume ---
    ct_img = read_dicom_series(ct_dir)
    ct_out = OUT / f"volume-{pid}.nii"
    sitk.WriteImage(ct_img, str(ct_out))

    # --- Liver mask ---
    liver_img = read_dicom_series(liver_dir)
    liver_arr = sitk.GetArrayFromImage(liver_img) > 0

    # --- Tumor masks ---
    tumor_arr = merge_tumors(mask_dir)
    if tumor_arr is None:
        tumor_arr = np.zeros_like(liver_arr, dtype=bool)

    # --- Combine ---
    combined = np.zeros_like(liver_arr, dtype=np.uint8)
    combined[liver_arr] = 1
    combined[tumor_arr] = 2

    mask_img = sitk.GetImageFromArray(combined)
    mask_img.CopyInformation(ct_img)
    mask_out = OUT / f"segmentation-{pid}.nii"
    sitk.WriteImage(mask_img, str(mask_out))

# --- Process all patients with tqdm ---
patients = sorted([p for p in ROOT.iterdir() if p.is_dir() and "3Dircadb1." in p.name])

for i, patient in enumerate(tqdm(patients, desc="Converting 3Dircadb1", unit="case")):
    convert_patient(patient, i + 1)

print(f"\nConversion complete! Saved to: {OUT}")
