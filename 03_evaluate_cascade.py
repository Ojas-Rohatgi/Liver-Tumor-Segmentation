#!/usr/bin/env python3
"""
evaluate_all_2_stage.py

Runs a full 2-stage (Liver -> Tumor) cascade evaluation on the entire dataset.
- Stage 1: Predict Liver (all slices)
- Post-process: Get largest liver component, dilate it.
- Stage 2: Predict Tumor (only on slices intersecting the dilated liver mask)
- Post-process: Remove small tumor components.
- Metrics: Calculates global Dice, per-case Dice, VOE, RVD, ASSD, HD95,
           and detection metrics (Precision/Recall) including recall
           broken down by lesion size (Small, Medium, Large).
"""

import torch
import numpy as np
import nibabel as nib
import os
from pathlib import Path
import sys
import pandas as pd
from tqdm import tqdm
import gc
import scipy.ndimage as ndi
from scipy.ndimage import distance_transform_edt, label, binary_dilation, generate_binary_structure

# --- Import our models ---
try:
    from model_2_5D import UNet_2_5D
except ImportError:
    print("Error: Could not import model_2_5D.py. Please ensure it is in the same directory.")
    sys.exit(1)

# --- Import evaluation metrics ---
try:
    # We use medpy for ASSD and HD95, which are standard surface metrics
    from medpy.metric.binary import hd95, assd
except ImportError:
    print("Missing 'medpy'. Please run: pip install medpy")
    exit()

# --- Configuration ---
# --- UPDATED: Paths to final models and data ---
MODEL_SAVE_DIR = Path.cwd() / "trained_models"
LIVER_MODEL_PATH = MODEL_SAVE_DIR / "liver_model_best.pth"
TUMOR_MODEL_PATH = MODEL_SAVE_DIR / "tumor_model_best.pth"
SOURCE_DATA_DIR = Path.cwd() / "3Dircadb1"

# --- UPDATED: Path to save final CSV results ---
RESULTS_DIR = Path.cwd() / "results"
CSV_SAVE_PATH = RESULTS_DIR / "eval_results_2_stage_full_leaderboard.csv"
# -----------------------------------------------

N_INPUT_CHANNELS = 5
N_OUTPUT_CLASSES = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Pre/Post-processing
CT_WINDOW_MIN = -100
CT_WINDOW_MAX = 400
TUMOR_MIN_COMPONENT_SIZE = 50  # Remove components smaller than this (voxels)
LIVER_DILATION_ITERATIONS = 5  # Iterations to dilate liver mask for tumor search

# Lesion Size Categories (voxels)
SIZE_SMALL_MAX = 1000  # < 1k
SIZE_MEDIUM_MAX = 8000  # 1k - 8k


# Large is > 8k

# ---------------------

def check_imports():
    """Checks for and installs missing packages."""
    try:
        import scipy, nibabel, tqdm, pandas, medpy, skimage
    except ImportError:
        print("Missing required packages. Installing...")
        import subprocess
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "scipy", "nibabel", "tqdm", "pandas", "medpy", "scikit-image"])


def normalize(volume):
    """Clip to CT window and normalize to [0, 1]."""
    volume = np.clip(volume, CT_WINDOW_MIN, CT_WINDOW_MAX)
    volume = (volume - CT_WINDOW_MIN) / (CT_WINDOW_MAX - CT_WINDOW_MIN)
    return volume.astype(np.float32)


def get_2_5d_slice_tensor(volume_data_3d, slice_index, context_slices, device):
    """Extract a 2.5D slice tensor centered on `slice_index`."""
    num_slices = volume_data_3d.shape[2]
    indices = [np.clip(slice_index + j, 0, num_slices - 1) for j in range(-context_slices, context_slices + 1)]

    # Input slices are (H, W, C)
    input_slices = volume_data_3d[..., indices]

    # Normalize
    input_tensor_norm = normalize(input_slices)

    # Transpose to (C, H, W)
    input_tensor_norm = np.moveaxis(input_tensor_norm, -1, 0)

    return torch.from_numpy(input_tensor_norm.copy()).unsqueeze(0).to(device)


def get_largest_component(mask_3d):
    """Returns the largest connected component from a 3D binary mask."""
    labeled, num_features = ndi.label(mask_3d)
    if num_features == 0:
        return mask_3d
    try:
        # Find the label of the largest component (excluding background label 0)
        largest_label = 1 + np.argmax([np.sum(labeled == i) for i in range(1, num_features + 1)])
        clean_mask_3d = (labeled == largest_label).astype(np.uint8)
        return clean_mask_3d
    except ValueError:
        # This can happen if no components are found (e.g., all empty)
        return mask_3d


def remove_small_components(mask, min_size=TUMOR_MIN_COMPONENT_SIZE):
    """Remove connected components smaller than min_size voxels."""
    if min_size == 0:
        return mask
    mask_bool = mask.astype(bool)
    labeled, num = ndi.label(mask_bool)
    if num == 0:
        return mask.astype(np.uint8)

    sizes = ndi.sum(mask_bool, labeled, range(1, num + 1))
    keep_labels = [i + 1 for i, s in enumerate(sizes) if s >= min_size]

    if not keep_labels:
        return np.zeros_like(mask, dtype=np.uint8)

    clean = np.isin(labeled, keep_labels).astype(np.uint8)
    return clean


def compute_detection_metrics(pred_mask, gt_mask, overlap_threshold=0.5):
    """
    Calculates LiTS-style lesion detection metrics (Precision/Recall)
    AND lesion recall by size.
    """
    pred_labels, num_pred = ndi.label(pred_mask)
    gt_labels, num_gt = ndi.label(gt_mask)

    # --- Metrics by Size ---
    num_small = 0
    num_medium = 0
    num_large = 0

    if num_gt > 0:
        gt_sizes = ndi.sum(gt_mask, gt_labels, range(1, num_gt + 1))

        # Get indices for each size category
        small_lesions_idx = [i for i, size in enumerate(gt_sizes) if size > 0 and size < SIZE_SMALL_MAX]
        medium_lesions_idx = [i for i, size in enumerate(gt_sizes) if SIZE_SMALL_MAX <= size < SIZE_MEDIUM_MAX]
        large_lesions_idx = [i for i, size in enumerate(gt_sizes) if size >= SIZE_MEDIUM_MAX]

        num_small = len(small_lesions_idx)
        num_medium = len(medium_lesions_idx)
        num_large = len(large_lesions_idx)

    # --- Standard Detection Metrics ---
    if num_gt == 0:  # No GT lesions
        return {'precision': 1.0 if num_pred == 0 else 0.0, 'recall': 1.0,
                'recall_small': 1.0, 'recall_medium': 1.0, 'recall_large': 1.0,
                'num_small': 0, 'num_medium': 0, 'num_large': 0}

    if num_pred == 0:  # We predicted nothing, but there *is* a GT
        return {'precision': 1.0, 'recall': 0.0,
                'recall_small': 0.0 if num_small > 0 else 1.0,
                'recall_medium': 0.0 if num_medium > 0 else 1.0,
                'recall_large': 0.0 if num_large > 0 else 1.0,
                'num_small': num_small, 'num_medium': num_medium, 'num_large': num_large}

    # Create a "hit matrix" of Dice overlaps
    overlap_matrix = np.zeros((num_gt, num_pred))

    for i in range(1, num_gt + 1):
        gt_component = (gt_labels == i)
        gt_vol = np.sum(gt_component)
        for j in range(1, num_pred + 1):
            pred_component = (pred_labels == j)
            pred_vol = np.sum(pred_component)

            intersection = np.sum(np.logical_and(gt_component, pred_component))
            if intersection > 0:
                dice = (2. * intersection) / (gt_vol + pred_vol)
                overlap_matrix[i - 1, j - 1] = dice

    # --- Calculate Recall (Overall) ---
    gt_hits_bool = np.any(overlap_matrix > overlap_threshold, axis=1)
    gt_hits = np.sum(gt_hits_bool)
    recall = gt_hits / num_gt

    # --- Calculate Precision (Overall) ---
    pred_hits = np.sum(np.any(overlap_matrix > overlap_threshold, axis=0))
    precision = pred_hits / num_pred

    # --- Calculate Recall (by Size) ---
    small_hits = np.sum(gt_hits_bool[small_lesions_idx])
    medium_hits = np.sum(gt_hits_bool[medium_lesions_idx])
    large_hits = np.sum(gt_hits_bool[large_lesions_idx])

    recall_small = small_hits / num_small if num_small > 0 else 1.0
    recall_medium = medium_hits / num_medium if num_medium > 0 else 1.0
    recall_large = large_hits / num_large if num_large > 0 else 1.0

    return {'precision': precision, 'recall': recall,
            'recall_small': recall_small, 'recall_medium': recall_medium, 'recall_large': recall_large,
            'num_small': num_small, 'num_medium': num_medium, 'num_large': num_large}


def compute_all_metrics(pred, target, spacing):
    """Compute all LiTS leaderboard metrics."""
    pred = pred.astype(bool)
    target = target.astype(bool)

    # Handle edge cases
    if not np.any(pred) and not np.any(target):  # Perfect True Negative
        metrics = {'Dice': 1.0, 'IoU': 1.0, 'VOE': 0.0, 'RVD': 0.0,
                   'Precision': 1.0, 'Recall': 1.0,
                   'ASSD': 0.0, 'HD95': 0.0}
        det_metrics_50 = compute_detection_metrics(pred, target, 0.5)
        metrics.update(det_metrics_50)
        return metrics

    if not np.any(pred) or not np.any(target):  # Total mismatch
        metrics = {'Dice': 0.0, 'IoU': 0.0, 'VOE': 1.0,
                   'RVD': (float(pred.sum()) - float(target.sum())) / float(target.sum()) if np.any(target) else np.nan,
                   'Precision': 0.0 if np.any(pred) else 1.0,
                   'Recall': 0.0 if np.any(target) else 1.0,
                   'ASSD': np.nan, 'HD95': np.nan}
        det_metrics_50 = compute_detection_metrics(pred, target, 0.5)
        metrics.update(det_metrics_50)
        return metrics

    # --- Voxel Metrics ---
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    tp = intersection
    fp = np.logical_and(pred, np.logical_not(target)).sum()
    fn = np.logical_and(np.logical_not(pred), target).sum()

    dice = (2. * intersection) / (pred.sum() + target.sum())
    iou = intersection / union
    voe = 1.0 - iou
    rvd = (float(pred.sum()) - float(target.sum())) / float(target.sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # --- Surface Metrics (from medpy) ---
    try:
        assd_val = assd(pred, target, voxelspacing=spacing)
    except Exception:
        assd_val = np.nan

    try:
        hd95_val = hd95(pred, target, voxelspacing=spacing)
    except Exception:
        hd95_val = np.nan

    # --- Detection Metrics ---
    det_metrics_50 = compute_detection_metrics(pred, target, 0.5)

    metrics = {
        'Dice': dice, 'IoU': iou, 'VOE': voe, 'RVD': rvd,
        'Precision': precision, 'Recall': recall,
        'ASSD': assd_val, 'HD95': hd95_val,
    }
    metrics.update(det_metrics_50)
    return metrics


def main():
    check_imports()

    # --- Create results directory ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # --------------------------------

    # 1. Load 2-stage models
    print(f"Loading 2-stage cascade models on device: {DEVICE}")
    liver_model = UNet_2_5D(n_channels_in=N_INPUT_CHANNELS, n_channels_out=N_OUTPUT_CLASSES).to(DEVICE)
    tumor_segmenter = UNet_2_5D(n_channels_in=N_INPUT_CHANNELS, n_channels_out=N_OUTPUT_CLASSES).to(DEVICE)

    try:
        liver_model.load_state_dict(torch.load(LIVER_MODEL_PATH, map_location=DEVICE))
        tumor_segmenter.load_state_dict(torch.load(TUMOR_MODEL_PATH, map_location=DEVICE))
    except FileNotFoundError as e:
        print(f"Error: Model file not found. {e}")
        print(f"Please make sure '{LIVER_MODEL_PATH.name}' and '{TUMOR_MODEL_PATH.name}'")
        print(f"are inside the '{MODEL_SAVE_DIR}' directory.")
        sys.exit(1)

    liver_model.eval()
    tumor_segmenter.eval()
    print("Models loaded successfully.")

    # 2. Find ALL volumes
    all_volumes = sorted(list(SOURCE_DATA_DIR.glob("**/volume-*.nii*")))
    print(f"Found {len(all_volumes)} volumes. Starting full leaderboard evaluation...")
    results_list = []

    global_tumor_intersection, global_tumor_pred_sum, global_tumor_gt_sum = 0, 0, 0
    global_liver_intersection, global_liver_pred_sum, global_liver_gt_sum = 0, 0, 0

    # Define a structure for 3D dilation
    dilation_struct = ndi.generate_binary_structure(3, 2)  # 3x3x3 connectivity

    # 3. --- Loop over ALL scans ---
    for vol_path in tqdm(all_volumes, unit="scan"):

        file_num = vol_path.name.split('-')[-1].split('.')[0]
        mask_name = f"segmentation-{file_num}"
        mask_path_list = list(SOURCE_DATA_DIR.glob(f"**/{mask_name}.nii*"))
        if not mask_path_list:
            print(f"Warning: No mask found for {vol_path.name}. Skipping.")
            continue
        mask_path = mask_path_list[0]

        try:
            mask_nib = nib.load(mask_path)
            gt_mask_data_3d = mask_nib.get_fdata().astype(np.uint8)
            vol_nib = nib.load(vol_path)
            vol_data = vol_nib.get_fdata()
        except Exception as e:
            print(f"Error loading {vol_path.name} or {mask_path.name}: {e}. Skipping.")
            continue

        gt_liver_mask = (gt_mask_data_3d > 0).astype(np.uint8)
        gt_tumor_mask = (gt_mask_data_3d == 2).astype(np.uint8)
        voxel_spacing = vol_nib.header.get_zooms()

        pred_liver_3d = np.zeros_like(vol_data, np.uint8)
        pred_tumor_3d = np.zeros_like(vol_data, np.uint8)
        num_slices, context = vol_data.shape[2], N_INPUT_CHANNELS // 2

        # 4. --- RUN THE 2-STAGE CASCADE ---
        with torch.no_grad():

            # --- STAGE 1: Liver Prediction (Full Pass) ---
            for i in range(num_slices):
                input_tensor = get_2_5d_slice_tensor(vol_data, i, context, DEVICE)
                liver_pred_slice = (torch.sigmoid(liver_model(input_tensor)) > 0.5).cpu().numpy().squeeze().astype(
                    np.uint8)
                if np.any(liver_pred_slice):
                    pred_liver_3d[..., i] = liver_pred_slice
                del input_tensor, liver_pred_slice

            # --- Liver Post-Processing & Dilation ---
            clean_liver_mask_3d = get_largest_component(pred_liver_3d)
            # Dilate the liver mask to create a search region for tumors (for "bulging" tumors)
            dilated_liver_mask_3d = binary_dilation(
                clean_liver_mask_3d,
                structure=dilation_struct,
                iterations=LIVER_DILATION_ITERATIONS
            )

            # --- STAGE 2: Tumor Prediction (On Dilated Mask) ---
            for i in range(num_slices):
                # Only run tumor model if dilated liver is present on this slice
                if np.any(dilated_liver_mask_3d[..., i]):
                    input_tensor = get_2_5d_slice_tensor(vol_data, i, context, DEVICE)
                    tumor_pred_slice = (
                            torch.sigmoid(tumor_segmenter(input_tensor)) > 0.5).cpu().numpy().squeeze().astype(
                        np.uint8)

                    # Store prediction WITHOUT multiplying by liver mask
                    pred_tumor_3d[..., i] = tumor_pred_slice
                    del input_tensor, tumor_pred_slice

        torch.cuda.empty_cache()

        # 5. --- Final Tumor Post-Processing ---
        # Remove small components from the final 3D tumor prediction
        clean_tumor_mask_3d = remove_small_components(pred_tumor_3d, TUMOR_MIN_COMPONENT_SIZE)

        # 6. --- Calculate ALL metrics for this scan ---
        liver_metrics = compute_all_metrics(clean_liver_mask_3d, gt_liver_mask, voxel_spacing)
        liver_metrics['scan'] = vol_path.name
        liver_metrics['target'] = 'Liver'
        results_list.append(liver_metrics)

        tumor_metrics = compute_all_metrics(clean_tumor_mask_3d, gt_tumor_mask, voxel_spacing)
        tumor_metrics['scan'] = vol_path.name
        tumor_metrics['target'] = 'Tumor'
        results_list.append(tumor_metrics)

        # --- Accumulate Global Dice values ---
        global_liver_intersection += np.logical_and(clean_liver_mask_3d, gt_liver_mask).sum()
        global_liver_pred_sum += np.sum(clean_liver_mask_3d)
        global_liver_gt_sum += np.sum(gt_liver_mask)

        global_tumor_intersection += np.logical_and(clean_tumor_mask_3d, gt_tumor_mask).sum()
        global_tumor_pred_sum += np.sum(clean_tumor_mask_3d)
        global_tumor_gt_sum += np.sum(gt_tumor_mask)

        # 7. --- Manually clear memory ---
        del vol_data, gt_mask_data_3d, pred_liver_3d, pred_tumor_3d, tumor_metrics, liver_metrics
        del mask_nib, vol_nib, gt_tumor_mask, clean_liver_mask_3d, clean_tumor_mask_3d, dilated_liver_mask_3d
        gc.collect()

    # --- 8. Save Full Results CSV ---
    print("\nFull dataset evaluation complete.")
    results_df = pd.DataFrame(results_list)

    results_df.to_csv(CSV_SAVE_PATH, index=False)
    print(f"Full leaderboard metrics saved to {CSV_SAVE_PATH}")

    # --- 9. Calculate and Print Final Leaderboard ---

    # Calculate Global Dice scores
    global_liver_dice = (2. * global_liver_intersection) / (global_liver_pred_sum + global_liver_gt_sum + 1e-8)
    global_tumor_dice = (2. * global_tumor_intersection) / (global_tumor_pred_sum + global_tumor_gt_sum + 1e-8)

    # --- LIVER METRICS ---
    liver_df = results_df[results_df['target'] == 'Liver'].copy()
    global_liver_rvd = (float(global_liver_pred_sum) - float(global_liver_gt_sum)) / float(global_liver_gt_sum + 1e-8)
    mean_liver_dice_per_case = liver_df['Dice'].mean()
    mean_liver_voe = liver_df['VOE'].mean()
    mean_liver_assd = liver_df['ASSD'].dropna().mean()
    mean_liver_hd95 = liver_df['HD95'].dropna().mean()

    # --- TUMOR METRICS ---
    tumor_df = results_df[results_df['target'] == 'Tumor'].copy()
    global_tumor_rvd = (float(global_tumor_pred_sum) - float(global_tumor_gt_sum)) / float(global_tumor_gt_sum + 1e-8)
    mean_tumor_dice_per_case = tumor_df['Dice'].mean()
    mean_tumor_voe = tumor_df['VOE'].mean()
    mean_tumor_assd = tumor_df['ASSD'].dropna().mean()
    mean_tumor_hd95 = tumor_df['HD95'].dropna().mean()
    mean_p50 = tumor_df['precision_50'].dropna().mean()
    mean_r50 = tumor_df['recall_50'].dropna().mean()
    mean_r_small = tumor_df['recall_small'].dropna().mean()
    mean_r_medium = tumor_df['recall_medium'].dropna().mean()
    mean_r_large = tumor_df['recall_large'].dropna().mean()
    total_small = tumor_df['num_small'].sum()
    total_medium = tumor_df['num_medium'].sum()
    total_large = tumor_df['num_large'].sum()

    # --- PRINT LIVER METRICS ---
    print("\n--- Final 2-Stage Model Metrics (Liver) ---")
    print("--------------------------------------------------------------------------")
    print(f"Dice per case (mean)         |   {mean_liver_dice_per_case:<10.4f}")
    print(f"Dice global                  |   {global_liver_dice:<10.4f}")
    print(f"VOE (mean)                   |   {mean_liver_voe:<10.4f}")
    print(f"RVD (global)                 |   {global_liver_rvd:<+10.4f}")
    print(f"ASSD (mean, mm)              |   {mean_liver_assd:<10.4f}")
    print(f"HD95 (mean, mm)              |   {mean_liver_hd95:<10.4f}")
    print("--------------------------------------------------------------------------")
    # -----------------------------------

    # --- PRINT TUMOR METRICS ---
    print("\n--- Final 2-Stage Model Metrics (Tumor) ---")
    print("--------------------------------------------------------------------------")
    print(f"Dice per case (mean)         |   {mean_tumor_dice_per_case:<10.4f}")
    print(f"Dice global                  |   {global_tumor_dice:<10.4f}")
    print(f"VOE (mean)                   |   {mean_tumor_voe:<10.4f}")
    print(f"RVD (global)                 |   {global_tumor_rvd:<+10.4f}")
    print(f"ASSD (mean, mm)              |   {mean_tumor_assd:<10.4f}")
    print(f"HD95 (mean, mm)              |   {mean_tumor_hd95:<10.4f}  <-- (Replaced MSD)")
    print("--------------------------------------------------------------------------")
    print(f"--- Detection Metrics (at 0.5 overlap) ---")
    print(f"Precision (mean)             |   {mean_p50:<10.4f}")
    print(f"Recall (mean, overall)       |   {mean_r50:<10.4f}")
    print(f"Recall (Small Lesions)       |   {mean_r_small:<10.4f}  (Total: {int(total_small)})")
    print(f"Recall (Medium Lesions)      |   {mean_r_medium:<10.4f}  (Total: {int(total_medium)})")
    print(f"Recall (Large Lesions)       |   {mean_r_large:<10.4f}  (Total: {int(total_large)})")
    print("--------------------------------------------------------------------------")


if __name__ == "__main__":
    check_imports()
    main()