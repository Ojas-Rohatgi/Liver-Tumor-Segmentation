# tasks.py
import torch
import numpy as np
import nibabel as nib
import random
from pathlib import Path
import sys
import torch.nn.functional as F
import matplotlib.patches as mpatches
import scipy.ndimage as ndi
from scipy.ndimage import label, binary_dilation, generate_binary_structure
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from skimage import measure
from celery import Celery
import shutil

# --- Import our model ---
# Add current directory to path
sys.path.append(str(Path.cwd()))
try:
    from model_2_5d import UNet_2_5D
except ImportError:
    print("Error: Could not import model_2_5d.py. Make sure it is in this directory.")
    sys.exit(1)

# --- Celery Configuration ---
# This sets up Redis as the "broker" (task list) and "backend" (result store)
CELERY_BROKER_URL = 'redis://localhost:6379/0'
CELERY_RESULT_BACKEND = 'redis://localhost:6379/0'

celery = Celery(
    'tasks',
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND
)
# ------------------------------


# --- Configuration (Copied from inference_worker.py) ---
N_INPUT_CHANNELS = 5
N_OUTPUT_CLASSES = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Pre/Post-processing
CT_WINDOW_MIN = -100
CT_WINDOW_MAX = 400
TUMOR_MIN_COMPONENT_SIZE = 50
LIVER_DILATION_ITERATIONS = 5

# --- Paths to final models ---
MODEL_SAVE_DIR = Path("/app/trained_models")
LIVER_MODEL_PATH = MODEL_SAVE_DIR / "liver_model_best.pth"
TUMOR_MODEL_PATH = MODEL_SAVE_DIR / "tumor_model_best.pth"


# ---------------------
#  HELPER FUNCTIONS
# ---------------------

def normalize(volume):
    # ... (function body remains the same)
    volume = np.clip(volume, CT_WINDOW_MIN, CT_WINDOW_MAX)
    volume = (volume - CT_WINDOW_MIN) / (CT_WINDOW_MAX - CT_WINDOW_MIN)
    return volume.astype(np.float32)


def get_2_5d_slice_tensor(volume_data_3d, slice_index, context_slices, device):
    # ... (function body remains the same)
    num_slices = volume_data_3d.shape[2]
    indices = [np.clip(slice_index + j, 0, num_slices - 1) for j in range(-context_slices, context_slices + 1)]
    input_slices = volume_data_3d[..., indices]
    input_tensor_norm = normalize(input_slices)
    input_tensor_norm = np.moveaxis(input_tensor_norm, -1, 0)
    return torch.from_numpy(input_tensor_norm.copy()).unsqueeze(0).to(device)


def get_largest_component(mask_3d):
    # ... (function body remains the same)
    labeled, num_features = ndi.label(mask_3d)
    if num_features == 0: return mask_3d
    try:
        largest_label = 1 + np.argmax([np.sum(labeled == i) for i in range(1, num_features + 1)])
        return (labeled == largest_label).astype(np.uint8)
    except ValueError:
        return mask_3d


def remove_small_components(mask, min_size=TUMOR_MIN_COMPONENT_SIZE):
    # ... (function body remains the same)
    if min_size == 0: return mask
    mask_bool = mask.astype(bool)
    labeled, num = ndi.label(mask_bool)
    if num == 0: return mask.astype(np.uint8)
    sizes = ndi.sum(mask_bool, labeled, range(1, num + 1))
    keep_labels = [i + 1 for i, s in enumerate(sizes) if s >= min_size]
    if not keep_labels: return np.zeros_like(mask, dtype=np.uint8)
    return np.isin(labeled, keep_labels).astype(np.uint8)


def visualize_2d_slice(ct_slice, pred_liver_slice, pred_tumor_slice, slice_idx, save_path):
    # ... (function body remains the same)
    print(f"Generating 2D plot for slice {slice_idx}...")
    ct_slice = np.clip(ct_slice, CT_WINDOW_MIN, CT_WINDOW_MAX)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 7))
    fig.suptitle(f"Prediction Results (Slice {slice_idx})", fontsize=16)
    ax1.imshow(ct_slice.T, cmap="gray", origin='lower')
    ax1.set_title("CT Scan")
    ax1.axis("off")
    ax2.imshow(ct_slice.T, cmap="gray", origin='lower')
    ax2.contour(pred_liver_slice.T, colors=['cyan'], levels=[0.5], linewidths=1)
    ax2.imshow(np.ma.masked_where(pred_tumor_slice.T == 0, pred_tumor_slice.T), cmap="autumn", alpha=0.6,
               origin='lower')
    ax2.set_title("Prediction")
    ax2.axis("off")
    handles = [mpatches.Patch(color='cyan', label='Pred Liver'), mpatches.Patch(color='yellow', label='Pred Tumor')]
    fig.legend(handles=handles, loc='lower center', ncol=2, bbox_to_anchor=(0.5, 0.01))
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(save_path, dpi=200)
    print(f"2D plot saved to {save_path}")
    plt.close(fig)


def visualize_3d_interactive(ct_volume, liver_mask, tumor_mask, target_max_dim=128, ct_iso_percentile=97,
                             outfile_path="prediction_3d_interactive.html"):
    # ... (function body remains the same)
    print(f"Generating 3D interactive plot (target_max_dim={target_max_dim})...")

    def adaptive_pool(vol, func=np.max):
        max_dim = max(vol.shape)
        if max_dim <= target_max_dim: return vol
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
            traces.append(
                go.Mesh3d(x=verts_ct[:, 0], y=verts_ct[:, 1], z=verts_ct[:, 2], i=faces_ct[:, 0], j=faces_ct[:, 1],
                          k=faces_ct[:, 2], color='lightgray', opacity=0.08, name='CT', flatshading=True))
    except Exception:
        pass
    try:
        if np.any(liver_small):
            verts_l, faces_l, _, _ = measure.marching_cubes(liver_small, level=0.5)
            traces.append(go.Mesh3d(x=verts_l[:, 0], y=verts_l[:, 1], z=verts_l[:, 2], i=faces_l[:, 0], j=faces_l[:, 1],
                                    k=faces_l[:, 2], color='blue', opacity=0.25, name='Liver', flatshading=True))
    except Exception:
        pass
    try:
        if np.any(tumor_small):
            verts_t, faces_t, _, _ = measure.marching_cubes(tumor_small, level=0.5)
            traces.append(go.Mesh3d(x=verts_t[:, 0], y=verts_t[:, 1], z=verts_t[:, 2], i=faces_t[:, 0], j=faces_t[:, 1],
                                    k=faces_t[:, 2], color='red', opacity=0.8, name='Tumor', flatshading=True))
    except Exception:
        pass
    if not traces:
        print("Nothing to render for 3D plot.")
        return
    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(aspectmode="data", xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),
        margin=dict(l=0, r=0, b=0, t=30), title=f"3D Prediction")
    fig.write_html(str(outfile_path), auto_open=False)
    print(f"3D interactive plot saved to {outfile_path}")


# ---------------------
#  NEW CLEANUP TASK (Must be defined first!)
# ---------------------
@celery.task
def cleanup_files(request_id):
    """
    Deletes the uploads and static directories associated with a request_id.
    This runs after a delay set in run_inference_task.
    """
    CLEANUP_DELAY_MINUTES = 60  # For logging
    print(f"Cleanup task starting for request_id: {request_id} after {CLEANUP_DELAY_MINUTES} minutes.")

    # Paths are defined relative to the current working directory (/app)
    upload_dir = Path.cwd() / "uploads" / request_id
    output_dir = Path.cwd() / "static" / request_id

    try:
        if upload_dir.exists():
            shutil.rmtree(upload_dir)
            print(f"Successfully deleted upload directory: {upload_dir}")

        if output_dir.exists():
            shutil.rmtree(output_dir)
            print(f"Successfully deleted static directory: {output_dir}")

    except OSError as e:
        print(f"Error during cleanup of {request_id}: {e}")


# ---------------------
#  THE MAIN CELERY TASK
# ---------------------
# This decorator turns the function into a Celery task
@celery.task(bind=True)
def run_inference_task(self, input_path_str, output_2d_path_str, output_3d_path_str):
    """
    This is the main Celery task that runs the inference.
    """
    # Convert string paths back to Path objects
    input_path = Path(input_path_str)
    output_2d_path = Path(output_2d_path_str)
    output_3d_path = Path(output_3d_path_str)

    try:
        # 1. Load models
        print(f"Loading 2-stage cascade models on device: {DEVICE}")
        liver_model = UNet_2_5D(n_channels_in=N_INPUT_CHANNELS, n_channels_out=N_OUTPUT_CLASSES).to(DEVICE)
        tumor_segmenter = UNet_2_5D(n_channels_in=N_INPUT_CHANNELS, n_channels_out=N_OUTPUT_CLASSES).to(DEVICE)

        liver_model.load_state_dict(torch.load(LIVER_MODEL_PATH, map_location=DEVICE))
        tumor_segmenter.load_state_dict(torch.load(TUMOR_MODEL_PATH, map_location=DEVICE))

        liver_model.eval()
        tumor_segmenter.eval()
        print("Models loaded successfully.")

        # 2. Load single scan data
        print(f"Loading volume: {input_path.name}")
        vol_nib = nib.load(input_path)
        vol_data = vol_nib.get_fdata()

        pred_liver_3d = np.zeros_like(vol_data, np.uint8)
        pred_tumor_3d = np.zeros_like(vol_data, np.uint8)
        num_slices, context = vol_data.shape[2], N_INPUT_CHANNELS // 2
        dilation_struct = ndi.generate_binary_structure(3, 2)

        # 3. --- RUN THE 2-STAGE CASCADE ---
        print("Running 2-stage cascade prediction...")
        with torch.no_grad():
            for i in range(num_slices):
                # Update task state for a progress bar (optional, but good)
                self.update_state(state='PROGRESS', meta={'current': i, 'total': num_slices, 'stage': 'Liver'})

                input_tensor = get_2_5d_slice_tensor(vol_data, i, context, DEVICE)
                liver_pred_slice = (torch.sigmoid(liver_model(input_tensor)) > 0.5).cpu().numpy().squeeze().astype(
                    np.uint8)
                if np.any(liver_pred_slice):
                    pred_liver_3d[..., i] = liver_pred_slice
                del input_tensor, liver_pred_slice

            clean_liver_mask_3d = get_largest_component(pred_liver_3d)
            dilated_liver_mask_3d = binary_dilation(clean_liver_mask_3d, structure=dilation_struct,
                                                    iterations=LIVER_DILATION_ITERATIONS)

            for i in range(num_slices):
                self.update_state(state='PROGRESS', meta={'current': i, 'total': num_slices, 'stage': 'Tumor'})

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

        # 5. --- Find interesting slice for 2D plot ---
        tumor_slice_sums = pred_tumor_3d.sum(axis=(0, 1))
        if np.any(tumor_slice_sums):
            slice_idx = np.argmax(tumor_slice_sums)
        else:
            liver_slice_sums = pred_liver_3d.sum(axis=(0, 1))
            if np.any(liver_slice_sums):
                slice_idx = np.argmax(liver_slice_sums)
            else:
                slice_idx = num_slices // 2

        # 6. --- Generate 2D Plot ---
        visualize_2d_slice(vol_data[..., slice_idx], clean_liver_mask_3d[..., slice_idx],
                           clean_tumor_mask_3d[..., slice_idx], slice_idx, output_2d_path)

        # 7. --- Generate 3D Plot ---
        visualize_3d_interactive(vol_data, clean_liver_mask_3d, clean_tumor_mask_3d, target_max_dim=128, outfile_path=output_3d_path)
        print("All outputs saved.")

        # 8. --- Schedule Cleanup --- <--- 2. ADD CLEANUP SCHEDULING
        # Extract the unique request_id from the output path
        request_id_for_cleanup = Path(output_3d_path_str).parent.name

        # Schedule cleanup for 60 minutes (3600 seconds)
        CLEANUP_DELAY_SECONDS = 3600

        # Schedule the cleanup task to run after the countdown period
        # This will now correctly reference the cleanup_files function
        cleanup_files.apply_async((request_id_for_cleanup,), countdown=CLEANUP_DELAY_SECONDS)
        print(f"Scheduled file cleanup for request {request_id_for_cleanup} in 60 minutes.")

        # Return a success message (will be stored in Redis)
        return {'status': 'Complete', '2d_plot': str(output_2d_path), '3d_plot': str(output_3d_path)}

    except Exception as e:
        # Log the error
        print(f"TASK FAILED: {e}")
        # Propagate the error to Celery
        raise e