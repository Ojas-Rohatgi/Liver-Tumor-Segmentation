import numpy as np
import nibabel as nib
import os
from pathlib import Path
from tqdm import tqdm
import scipy.ndimage as ndi
import sys

# --- Configuration ---
# This is your raw dataset folder
SOURCE_DATA_DIR = Path.cwd() / "LITS_Challenge_Data"

# This is the folder where processed slices will be saved
OUTPUT_DATA_DIR = Path.cwd() / "data_2_5d_tumor_GT"

TARGET_LABEL = 2  # 2 for Tumor. We ONLY want slices with tumors.
N_SLICES = 5  # Number of slices for 2.5D input (5 = middle + 2 above + 2 below)
RESIZE_SHAPE = (512, 512)  # Original shape, no resizing
# ---------------------

# Preprocessing settings (same as in evaluation)
CT_WINDOW_MIN = -100
CT_WINDOW_MAX = 400


def normalize(volume):
    """Clip to CT window and normalize to [0, 1]."""
    volume = np.clip(volume, CT_WINDOW_MIN, CT_WINDOW_MAX)
    volume = (volume - CT_WINDOW_MIN) / (CT_WINDOW_MAX - CT_WINDOW_MIN)
    return volume.astype(np.float32)


def resample_and_normalize(volume, original_spacing, target_shape=RESIZE_SHAPE):
    """
    Resamples the volume to a standard size (e.g., 512x512) and normalizes.
    This version assumes we just want to resize in x-y plane.
    """
    if volume.shape[0] != target_shape[0] or volume.shape[1] != target_shape[1]:
        # Calculate resize factors for x and y, keep z the same
        resize_factor = [
            target_shape[0] / volume.shape[0],
            target_shape[1] / volume.shape[1],
            1.0  # No resize in z-axis
        ]

        # Resample volume
        resized_volume = ndi.zoom(volume, resize_factor, order=1, prefilter=False)
    else:
        resized_volume = volume

    # Normalize the volume after resizing
    normalized_volume = normalize(resized_volume)
    return normalized_volume


def resample_mask(mask, target_shape=RESIZE_SHAPE):
    """Resamples the mask using nearest-neighbor interpolation."""
    if mask.shape[0] != target_shape[0] or mask.shape[1] != target_shape[1]:
        resize_factor = [
            target_shape[0] / mask.shape[0],
            target_shape[1] / mask.shape[1],
            1.0
        ]
        # Resample mask using nearest neighbor (order=0)
        resized_mask = ndi.zoom(mask, resize_factor, order=0, prefilter=False)
    else:
        resized_mask = mask

    return resized_mask.astype(np.uint8)


def process_scan(vol_path, mask_path, img_out_dir, msk_out_dir, context_slices):
    """
    Processes a single NIfTI scan and saves its 2.5D slices as .npy files.
    Only saves slices that contain the target label.
    """
    try:
        vol_nib = nib.load(vol_path)
        vol_data = vol_nib.get_fdata()
        mask_nib = nib.load(mask_path)
        mask_data = mask_nib.get_fdata().astype(np.uint8)

        spacing = vol_nib.header.get_zooms()

        # Resample and normalize
        vol_data_norm = resample_and_normalize(vol_data, spacing, target_shape=RESIZE_SHAPE)
        mask_data_resampled = resample_mask(mask_data, target_shape=RESIZE_SHAPE)

        # --- IMPORTANT ---
        # For TUMOR model, we ONLY want label 2
        mask_data_target = (mask_data_resampled == TARGET_LABEL).astype(np.uint8)

        num_slices = vol_data_norm.shape[2]
        slices_saved = 0

        for i in range(num_slices):
            # Only save slices that contain the TUMOR
            if np.sum(mask_data_target[..., i]) > 0:
                # Create 2.5D slice
                slice_indices = [np.clip(i + j, 0, num_slices - 1) for j in range(-context_slices, context_slices + 1)]

                # Stack slices along the channel dimension (axis 0)
                # Shape becomes (N_SLICES, H, W)
                img_slice_2_5d = vol_data_norm[..., slice_indices].transpose(2, 0, 1)

                # The corresponding mask is just the single center slice
                # Shape (H, W)
                mask_slice = mask_data_target[..., i]

                # Create a unique filename
                scan_name = vol_path.name.split('.')[0]  # e.g., "volume-0"
                filename = f"{scan_name}_slice_{i:03d}.npy"

                # Save the image and mask
                np.save(os.path.join(img_out_dir, filename), img_slice_2_5d)
                np.save(os.path.join(msk_out_dir, filename), mask_slice)
                slices_saved += 1

        return slices_saved

    except Exception as e:
        print(f"Error processing {vol_path.name}: {e}")
        return 0


def main():
    print("Starting 2.5D data preparation for TUMOR model...")

    # Create output directories
    img_out_dir = Path(OUTPUT_DATA_DIR) / "imagesTr"
    msk_out_dir = Path(OUTPUT_DATA_DIR) / "labelsTr"

    img_out_dir.mkdir(parents=True, exist_ok=True)
    msk_out_dir.mkdir(parents=True, exist_ok=True)

    # Find all volume files
    all_volumes = sorted(list(SOURCE_DATA_DIR.glob("**/volume-*.nii*")))
    if not all_volumes:
        print(f"Error: No 'volume-*.nii*' files found in {SOURCE_DATA_DIR}")
        print("Please check your SOURCE_DATA_DIR path.")
        sys.exit(1)

    print(f"Found {len(all_volumes)} scans.")

    total_slices_saved = 0
    context_slices = N_SLICES // 2

    # Loop over all scans
    for vol_path in tqdm(all_volumes, unit="scan"):

        # Find corresponding mask
        file_num = vol_path.name.split('-')[-1].split('.')[0]
        mask_name = f"segmentation-{file_num}"
        mask_path_list = list(SOURCE_DATA_DIR.glob(f"**/{mask_name}.nii*"))

        if not mask_path_list:
            print(f"Warning: No mask found for {vol_path.name}. Skipping.")
            continue

        mask_path = mask_path_list[0]

        slices_saved = process_scan(vol_path, mask_path, img_out_dir, msk_out_dir, context_slices)
        total_slices_saved += slices_saved

    print("\n" + "=" * 50)
    print("Data preparation complete!")
    print(f"Total 2.5D (TUMOR-ONLY) slices saved: {total_slices_saved}")
    print(f"Output directory: {OUTPUT_DATA_DIR}")
    print("=" * 50)


if __name__ == "__main__":
    main()