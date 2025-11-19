#!/usr/bin/env python3
"""
predict_single_2_stage.py

Runs the 2-stage cascade (Liver -> dilated -> Tumor) on a *single* CT scan,
PRINTS the evaluation metrics for that scan (if a mask is provided),
and generates 2D slice and 3D interactive visualizations.
"""

import torch
import numpy as np
import nibabel as nib
import random
from pathlib import Path
import sys
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.patches as mpatches
import scipy.ndimage as ndi
from scipy.ndimage import label, binary_dilation, generate_binary_structure
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from skimage import measure

# --- Import evaluation metrics ---
try:
    from medpy.metric.binary import hd95, assd
except ImportError:
    print("Missing 'medpy'. Please run: pip install medpy")
    exit()

# --- Import our models ---
try:
    from model_2_5D import UNet_2_5D
except ImportError:
    print("Error: Could not import model_2_5D.py. Please ensure it is in the same directory.")
    sys.exit(1)

# --- Configuration ---
# --- UPDATED: Paths to final models and data ---
MODEL_SAVE_DIR = Path.cwd() / "trained_models"
LIVER_MODEL_PATH = MODEL_SAVE_DIR / "liver_model_best.pth"
TUMOR_MODEL_PATH = MODEL_SAVE_DIR / "tumor_model_best.pth"
SOURCE_DATA_DIR = Path.cwd() / "3Dircadb1"
RESULTS_DIR = Path.cwd() / "results"
# -----------------------------------------------

# *** SPECIFY THE SCAN TO PREDICT HERE ***
# This script will try to find this volume and its corresponding segmentation
SCAN_ID_TO_PREDICT = "volume-02.nii.gz"  # Example: "volume-02.nii.gz" or "volume-10.nii"
# ----------------------------------------

N_INPUT_CHANNELS = 5
N_OUTPUT_CLASSES = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Pre/Post-processing
CT_WINDOW_MIN = -100
CT_WINDOW_MAX = 400
TUMOR_MIN_COMPONENT_SIZE = 50
LIVER_DILATION_ITERATIONS = 5

# Lesion Size Categories (voxels) - FOR METRICS
SIZE_SMALL_MAX = 1000  # < 1k
SIZE_MEDIUM_MAX = 8000  # 1k - 8k


# Large is > 8k

# ---------------------

def check_imports():
    """Checks for and installs missing packages."""
    try:
        import scipy, nibabel, tqdm, matplotlib, plotly, skimage, medpy
    except ImportError:
        print("Missing required packages. Installing...")
        import subprocess
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install",
             "scipy", "nibabel", "tqdm", "matplotlib", "plotly", "scikit-image", "medpy"])


def normalize(volume):
    """Clip to CT window and normalize to [0, 1]."""
    volume = np.clip(volume, CT_WINDOW_MIN, CT_WINDOW_MAX)
    volume = (volume - CT_WINDOW_MIN) / (CT_WINDOW_MAX - CT_WINDOW_MIN)
    return volume.astype(np.float32)


def get_2_5d_slice_tensor(volume_data_3d, slice_index, context_slices, device):
    """Extract a 2.5D slice tensor centered on `slice_index`."""
    num_slices = volume_data_3d.shape[2]
    indices = [np.clip(slice_index + j, 0, num_slices - 1) for j in range(-context_slices, context_slices + 1)]
    input_slices = volume_data_3d[..., indices]
    input_tensor_norm = normalize(input_slices)
    input_tensor_norm = np.moveaxis(input_tensor_norm, -1, 0)  # (H,W,C) -> (C,H,W)
    return torch.from_numpy(input_tensor_norm.copy()).unsqueeze(0).to(device)


def get_largest_component(mask_3d):
    """Returns the largest connected component from a 3D binary mask."""
    labeled, num_features = ndi.label(mask_3d)
    if num_features == 0:
        return mask_3d
    try:
        largest_label = 1 + np.argmax([np.sum(labeled == i) for i in range(1, num_features + 1)])
        clean_mask_3d = (labeled == largest_label).astype(np.uint8)
        return clean_mask_3d
    except ValueError:
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


# ===================================================================
# --- START: METRIC CALCULATION FUNCTIONS
# ===================================================================

def compute_detection_metrics(pred_mask, gt_mask, overlap_threshold=0.5):
    pred_labels, num_pred = ndi.label(pred_mask)
    gt_labels, num_gt = ndi.label(gt_mask)

    num_small, num_medium, num_large = 0, 0, 0

    if num_gt > 0:
        gt_sizes = ndi.sum(gt_mask, gt_labels, range(1, num_gt + 1))
        small_lesions_idx = [i for i, size in enumerate(gt_sizes) if size > 0 and size < SIZE_SMALL_MAX]
        medium_lesions_idx = [i for i, size in enumerate(gt_sizes) if SIZE_SMALL_MAX <= size < SIZE_MEDIUM_MAX]
        large_lesions_idx = [i for i, size in enumerate(gt_sizes) if size >= SIZE_MEDIUM_MAX]
        num_small = len(small_lesions_idx)
        num_medium = len(medium_lesions_idx)
        num_large = len(large_lesions_idx)

    if num_gt == 0:
        return {'precision': 1.0 if num_pred == 0 else 0.0, 'recall': 1.0,
                'recall_small': 1.0, 'recall_medium': 1.0, 'recall_large': 1.0,
                'num_small': 0, 'num_medium': 0, 'num_large': 0}

    if num_pred == 0:
        return {'precision': 1.0, 'recall': 0.0,
                'recall_small': 0.0 if num_small > 0 else 1.0,
                'recall_medium': 0.0 if num_medium > 0 else 1.0,
                'recall_large': 0.0 if num_large > 0 else 1.0,
                'num_small': num_small, 'num_medium': num_medium, 'num_large': num_large}

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

    gt_hits_bool = np.any(overlap_matrix > overlap_threshold, axis=1)
    gt_hits = np.sum(gt_hits_bool)
    recall = gt_hits / num_gt

    pred_hits = np.sum(np.any(overlap_matrix > overlap_threshold, axis=0))
    precision = pred_hits / num_pred

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

    if not np.any(pred) and not np.any(target):  # Perfect True Negative
        metrics = {'Dice': 1.0, 'IoU': 1.0, 'VOE': 0.0, 'RVD': 0.0,
                   'Precision': 1.0, 'Recall': 1.0, 'ASSD': 0.0, 'HD95': 0.0}
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

    try:
        assd_val = assd(pred, target, voxelspacing=spacing)
    except Exception:
        assd_val = np.nan
    try:
        hd95_val = hd95(pred, target, voxelspacing=spacing)
    except Exception:
        hd95_val = np.nan

    det_metrics_50 = compute_detection_metrics(pred, target, 0.5)
    metrics = {'Dice': dice, 'IoU': iou, 'VOE': voe, 'RVD': rvd,
               'Precision': precision, 'Recall': recall,
               'ASSD': assd_val, 'HD95': hd95_val}
    metrics.update(det_metrics_50)
    return metrics


# ===================================================================
# --- END: METRIC CALCULATION FUNCTIONS
# ===================================================================


def visualize_2d_slice(ct_slice, gt_liver_slice, gt_tumor_slice, pred_liver_slice, pred_tumor_slice, slice_idx,
                       save_path):
    """Generates a 3-panel 2D plot comparing GT and Prediction."""
    print(f"Generating 2D plot for slice {slice_idx}...")
    ct_slice = np.clip(ct_slice, CT_WINDOW_MIN, CT_WINDOW_MAX)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 7))
    fig.suptitle(f"Results: {save_path.stem} (Slice {slice_idx})", fontsize=16)

    # Panel 1: CT Only
    ax1.imshow(ct_slice.T, cmap="gray", origin='lower')
    ax1.set_title("CT Scan")
    ax1.axis("off")

    # Panel 2: Ground Truth
    ax2.imshow(ct_slice.T, cmap="gray", origin='lower')
    ax2.contour(gt_liver_slice.T, colors=['blue'], levels=[0.5], linewidths=1)
    ax2.imshow(np.ma.masked_where(gt_tumor_slice.T == 0, gt_tumor_slice.T), cmap="Reds", alpha=0.6, origin='lower')
    ax2.set_title("Ground Truth (Whole Organ)")
    ax2.axis("off")

    # Panel 3: Prediction
    ax3.imshow(ct_slice.T, cmap="gray", origin='lower')
    ax3.contour(pred_liver_slice.T, colors=['cyan'], levels=[0.5], linewidths=1)
    ax3.imshow(np.ma.masked_where(pred_tumor_slice.T == 0, pred_tumor_slice.T), cmap="autumn", alpha=0.6,
               origin='lower')
    ax3.set_title("Prediction")
    ax3.axis("off")

    handles = [mpatches.Patch(color='blue', label='GT Liver (Whole)'),
               mpatches.Patch(color='red', label='GT Tumor'),
               mpatches.Patch(color='cyan', label='Pred Liver'),
               mpatches.Patch(color='yellow', label='Pred Tumor')]
    fig.legend(handles=handles, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(save_path, dpi=200)
    print(f"2D plot saved to {save_path}")
    plt.close(fig)


def visualize_3d_interactive(ct_volume, liver_mask, tumor_mask,
                             target_max_dim=128,
                             ct_iso_percentile=97,
                             outfile_path="prediction_3d_interactive.html"):
    """
    Fast interactive 3D visualization using marching_cubes -> plotly.Mesh3d.
    Downsamples the volumes for performance.
    """
    print(f"Generating 3D interactive plot (target_max_dim={target_max_dim})...")

    def adaptive_pool(vol, func=np.max):
        max_dim = max(vol.shape)
        if max_dim <= target_max_dim:
            return vol
        scale = int(np.ceil(max_dim / target_max_dim))
        bx = max(1, int(np.ceil(vol.shape[0] / (vol.shape[0] / scale))))
        by = max(1, int(np.ceil(vol.shape[1] / (vol.shape[1] / scale))))
        bz = max(1, int(np.ceil(vol.shape[2] / (vol.shape[2] / scale))))
        block_size = (int(bx), int(by), int(bz))
        return measure.block_reduce(vol, block_size=block_size, func=func, cval=np.min(vol))

    ct_norm = (ct_volume - ct_volume.min()) / (ct_volume.max() - ct_volume.min() + 1e-8)

    ct_small = adaptive_pool(ct_norm, func=np.mean)
    liver_small = adaptive_pool(liver_mask.astype(np.uint8), func=np.max)
    tumor_small = adaptive_pool(tumor_mask.astype(np.uint8), func=np.max)

    traces = []

    try:
        if np.any(ct_small):
            ct_level = np.percentile(ct_small[ct_small > 0], ct_iso_percentile)
            verts_ct, faces_ct, _, _ = measure.marching_cubes(ct_small, level=float(ct_level))
            traces.append(go.Mesh3d(
                x=verts_ct[:, 0], y=verts_ct[:, 1], z=verts_ct[:, 2],
                i=faces_ct[:, 0], j=faces_ct[:, 1], k=faces_ct[:, 2],
                color='lightgray', opacity=0.08, name='CT', flatshading=True
            ))
    except Exception as e:
        print(f"Warning: CT 3D mesh failed: {e}")

    try:
        if np.any(liver_small):
            verts_l, faces_l, _, _ = measure.marching_cubes(liver_small, level=0.5)
            traces.append(go.Mesh3d(
                x=verts_l[:, 0], y=verts_l[:, 1], z=verts_l[:, 2],
                i=faces_l[:, 0], j=faces_l[:, 1], k=faces_l[:, 2],
                color='blue', opacity=0.25, name='Liver', flatshading=True
            ))
    except Exception as e:
        print(f"Warning: Liver 3D mesh failed: {e}")

    try:
        if np.any(tumor_small):
            verts_t, faces_t, _, _ = measure.marching_cubes(tumor_small, level=0.5)
            traces.append(go.Mesh3d(
                x=verts_t[:, 0], y=verts_t[:, 1], z=verts_t[:, 2],
                i=faces_t[:, 0], j=faces_t[:, 1], k=faces_t[:, 2],
                color='red', opacity=0.8, name='Tumor', flatshading=True
            ))
    except Exception as e:
        print(f"Warning: Tumor 3D mesh failed: {e}")

    if not traces:
        print("Nothing to render for 3D plot.")
        return

    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(aspectmode="data",
                   xaxis=dict(visible=False),
                   yaxis=dict(visible=False),
                   zaxis=dict(visible=False)),
        margin=dict(l=0, r=0, b=0, t=30),
        title=f"3D Prediction: {outfile_path.stem}"
    )

    fig.write_html(str(outfile_path), auto_open=True)
    print(f"3D interactive plot saved to {outfile_path}")


def main():
    check_imports()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load models
    print(f"Loading 2-stage cascade models on device: {DEVICE}")
    liver_model = UNet_2_5D(n_channels_in=N_INPUT_CHANNELS, n_channels_out=N_OUTPUT_CLASSES).to(DEVICE)
    tumor_segmenter = UNet_2_5D(n_channels_in=N_INPUT_CHANNELS, n_channels_out=N_OUTPUT_CLASSES).to(DEVICE)

    try:
        liver_model.load_state_dict(torch.load(LIVER_MODEL_PATH, map_location=DEVICE))
        tumor_segmenter.load_state_dict(torch.load(TUMOR_MODEL_PATH, map_location=DEVICE))
    except FileNotFoundError as e:
        print(f"Error: Model file not found. {e}")
        print(f"Please check '{MODEL_SAVE_DIR}'")
        sys.exit(1)

    liver_model.eval()
    tumor_segmenter.eval()
    print("Models loaded successfully.")

    # 2. Find and load single scan data
    try:
        # Find the volume file
        vol_path_list = list(SOURCE_DATA_DIR.glob(f"**/{SCAN_ID_TO_PREDICT}"))
        if not vol_path_list:
            print(f"Error: Could not find volume file {SCAN_ID_TO_PREDICT} in {SOURCE_DATA_DIR}")
            sys.exit(1)
        VOLUME_PATH = vol_path_list[0]

        # Find the corresponding mask
        file_num = VOLUME_PATH.name.split('-')[-1].split('.')[0]
        mask_name = f"segmentation-{file_num}"
        mask_path_list = list(SOURCE_DATA_DIR.glob(f"**/{mask_name}.nii*"))
        MASK_PATH = mask_path_list[0] if mask_path_list else None

        print(f"Loading volume: {VOLUME_PATH.name}")
        vol_nib = nib.load(VOLUME_PATH)
        vol_data = vol_nib.get_fdata()
        voxel_spacing = vol_nib.header.get_zooms()
    except Exception as e:
        print(f"Error: Could not load volume file {VOLUME_PATH}. {e}")
        sys.exit(1)

    gt_liver_mask = np.zeros_like(vol_data, np.uint8)
    gt_tumor_mask = np.zeros_like(vol_data, np.uint8)
    mask_loaded = False

    if MASK_PATH and MASK_PATH.exists():
        print(f"Loading mask: {MASK_PATH.name}")
        mask_nib = nib.load(MASK_PATH)
        gt_mask_data_3d = mask_nib.get_fdata().astype(np.uint8)
        gt_liver_mask = (gt_mask_data_3d > 0).astype(np.uint8)
        gt_tumor_mask = (gt_mask_data_3d == 2).astype(np.uint8)
        mask_loaded = True
    else:
        print("Warning: No mask file provided or found. Will only show prediction.")
        print("         Metric calculation will be skipped.")

    pred_liver_3d = np.zeros_like(vol_data, np.uint8)
    pred_tumor_3d = np.zeros_like(vol_data, np.uint8)
    num_slices, context = vol_data.shape[2], N_INPUT_CHANNELS // 2

    dilation_struct = ndi.generate_binary_structure(3, 2)

    # 3. --- RUN THE 2-STAGE CASCADE ---
    print("Running 2-stage cascade prediction...")
    with torch.no_grad():

        for i in tqdm(range(num_slices), unit="slice", desc="Stage 1: Liver"):
            input_tensor = get_2_5d_slice_tensor(vol_data, i, context, DEVICE)
            liver_pred_slice = (torch.sigmoid(liver_model(input_tensor)) > 0.5).cpu().numpy().squeeze().astype(np.uint8)
            if np.any(liver_pred_slice):
                pred_liver_3d[..., i] = liver_pred_slice
            del input_tensor, liver_pred_slice

        clean_liver_mask_3d = get_largest_component(pred_liver_3d)
        dilated_liver_mask_3d = binary_dilation(
            clean_liver_mask_3d,
            structure=dilation_struct,
            iterations=LIVER_DILATION_ITERATIONS
        )

        for i in tqdm(range(num_slices), unit="slice", desc="Stage 2: Tumor"):
            if np.any(dilated_liver_mask_3d[..., i]):
                input_tensor = get_2_5d_slice_tensor(vol_data, i, context, DEVICE)
                tumor_pred_slice = (
                        torch.sigmoid(tumor_segmenter(input_tensor)) > 0.5).cpu().numpy().squeeze().astype(
                    np.uint8)
                pred_tumor_3d[..., i] = tumor_pred_slice
                del input_tensor, tumor_pred_slice

    torch.cuda.empty_cache()
    print("Inference complete.")

    # 4. --- Final Tumor Post-Processing ---
    clean_tumor_mask_3d = remove_small_components(pred_tumor_3d, TUMOR_MIN_COMPONENT_SIZE)
    print(f"Post-processed tumor mask (removed components < {TUMOR_MIN_COMPONENT_SIZE} voxels).")

    # 5. --- CALCULATE AND PRINT METRICS ---
    if mask_loaded:
        print("\n" + "=" * 50)
        print(f"📊 METRICS FOR SCAN: {VOLUME_PATH.name}")
        print("=" * 50 + "\n")

        print("--- 1. LIVER METRICS ---")
        liver_metrics = compute_all_metrics(clean_liver_mask_3d, gt_liver_mask, voxel_spacing)
        for k, v in liver_metrics.items():
            if 'num_' not in k and 'recall_' not in k and 'precision' not in k:
                print(f"  {k:12s}: {v:0.4f}")

        print("\n--- 2. TUMOR METRICS ---")
        tumor_metrics = compute_all_metrics(clean_tumor_mask_3d, gt_tumor_mask, voxel_spacing)
        for k, v in tumor_metrics.items():
            if 'num_' not in k and 'recall_' not in k and 'precision' not in k:
                print(f"  {k:12s}: {v:0.4f}")

        print("\n--- 3. DETECTION METRICS (Tumor, @0.5 overlap) ---")
        print(f"  Recall:       {tumor_metrics['recall_50']:.4f}")
        print(f"  Precision:    {tumor_metrics['precision_50']:.4f}")
        print(f"  Recall (S):   {tumor_metrics['recall_small']:.4f}  (Total: {tumor_metrics['num_small']})")
        print(f"  Recall (M):   {tumor_metrics['recall_medium']:.4f}  (Total: {tumor_metrics['num_medium']})")
        print(f"  Recall (L):   {tumor_metrics['recall_large']:.4f}  (Total: {tumor_metrics['num_large']})")
        print("\n" + "=" * 50 + "\n")
    else:
        print("\nMetrics skipped (no ground truth mask loaded).\n")

    # 6. --- Find interesting slice for 2D plot ---
    tumor_slice_sums = pred_tumor_3d.sum(axis=(0, 1))
    if np.any(tumor_slice_sums):
        slice_idx = np.argmax(tumor_slice_sums)
    else:
        liver_slice_sums = pred_liver_3d.sum(axis=(0, 1))
        if np.any(liver_slice_sums):
            slice_idx = np.argmax(liver_slice_sums)
        else:
            slice_idx = num_slices // 2

    # 7. --- Generate 2D Plot ---
    png_save_path = RESULTS_DIR / f"prediction_2d_{VOLUME_PATH.name.split('.')[0]}.png"
    visualize_2d_slice(
        vol_data[..., slice_idx],
        gt_liver_mask[..., slice_idx],
        gt_tumor_mask[..., slice_idx],
        clean_liver_mask_3d[..., slice_idx],
        clean_tumor_mask_3d[..., slice_idx],
        slice_idx,
        png_save_path
    )

    # 8. --- Generate 3D Plot ---
    html_save_path = RESULTS_DIR / f"prediction_3d_{VOLUME_PATH.name.split('.')[0]}.html"
    visualize_3d_interactive(
        vol_data,
        clean_liver_mask_3d,
        clean_tumor_mask_3d,
        target_max_dim=128,
        outfile_path=html_save_path
    )


if __name__ == "__main__":
    check_imports()
    main()