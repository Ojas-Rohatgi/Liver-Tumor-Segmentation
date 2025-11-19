import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import glob
import os
import sys
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import random
from pathlib import Path  # Import Path

# --- Import our model ---
try:
    from model_2_5D import UNet_2_5D
except ImportError:
    print("=" * 50)
    print("Error: Could not import UNet_2_5D from model_2_5D.py")
    print("Please make sure 'model_2_5D.py' is in the same directory.")
    print("=" * 50)
    exit()

# --- Configuration ---
DATA_DIR = Path.cwd() / "data_2_5d_liver"
IMAGE_DIR = DATA_DIR / "imagesTr"

N_INPUT_CHANNELS = 5  # Must match the 5 slices we created
N_OUTPUT_CLASSES = 1  # 1 channel for binary segmentation (liver vs. not-liver)

BATCH_SIZE = 8
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_EPOCHS = 100  # Train for a long time
PATIENCE = 10  # Stop after 10 epochs of no validation improvement

# --- UPDATED: Save path inside trained_models/ folder ---
MODEL_SAVE_DIR = Path.cwd() / "trained_models"
MODEL_SAVE_PATH = MODEL_SAVE_DIR / "liver_model_best.pth"


# --------------------------------------------------------


# -----------------------------------------------------------------
# --- Loss Function Classes (Dice + Combined) ---
# -----------------------------------------------------------------
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """Calculates Dice Loss for a batch."""
        probas = torch.sigmoid(logits)
        targets = targets.float()

        # Flatten batch, channel, height, width
        probas = probas.view(probas.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (probas * targets).sum()
        cardinality = probas.sum() + targets.sum()

        dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return dice_loss


class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        bce = self.bce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)

        return (self.bce_weight * bce) + (self.dice_weight * dice)


# -----------------------------------------------------------------
# --- EarlyStopper Class ---
# -----------------------------------------------------------------
class EarlyStopper:
    def __init__(self, patience=5, verbose=False, delta=0, path='model.pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # Make sure the directory exists
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


# -----------------------------------------------------------------

class LiverDataset(Dataset):
    """
    Custom PyTorch Dataset for our 2.5D .npy files.
    INCLUDES CORRECTED DATA AUGMENTATION.
    """

    def __init__(self, image_paths, augment=False):
        self.image_paths = image_paths
        self.augment = augment

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        # All masks are in 'labelsTr'
        mask_path = str(image_path).replace("imagesTr", "labelsTr")

        # Load as (C, H, W)
        image_data = np.load(image_path)
        # Load as (H, W)
        mask_data = np.load(mask_path)

        # Ensure mask is (1, H, W)
        if mask_data.ndim == 2:
            mask_data = np.expand_dims(mask_data, axis=0)

        # --- CORRECTED AUGMENTATION BLOCK ---
        if self.augment:
            # Random horizontal flip (flips axis 2)
            if random.random() > 0.5:
                image_data = np.flip(image_data, axis=2).copy()
                mask_data = np.flip(mask_data, axis=2).copy()

            # Random vertical flip (flips axis 1)
            if random.random() > 0.5:
                image_data = np.flip(image_data, axis=1).copy()
                mask_data = np.flip(mask_data, axis=1).copy()

            # Random 90-degree rotation (rotates axes 1 and 2)
            k = random.randint(0, 3)
            if k > 0:
                image_data = np.rot90(image_data, k=k, axes=(1, 2)).copy()
                mask_data = np.rot90(mask_data, k=k, axes=(1, 2)).copy()
        # ------------------------------------

        # Convert to Tensors
        image_tensor = torch.from_numpy(image_data).float()
        mask_tensor = torch.from_numpy(mask_data).float()

        return image_tensor, mask_tensor


def train():
    print(f"Using device: {DEVICE}")
    print("Searching for data in:", IMAGE_DIR)

    all_image_paths = glob.glob(os.path.join(IMAGE_DIR, "*.npy"))
    if not all_image_paths:
        print("Error: No .npy files found in", IMAGE_DIR)
        print("Please run the 'prepare_2_5d_data.py' script first.")
        return

    print(f"Found {len(all_image_paths)} total 2.5D slices.")

    # Split data
    train_paths, val_paths = train_test_split(all_image_paths, test_size=0.2, random_state=42)

    # --- Enable augmentation only for training set ---
    train_dataset = LiverDataset(train_paths, augment=True)
    val_dataset = LiverDataset(val_paths, augment=False)
    # -------------------------------------------------

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}")

    # Initialize model, loss, optimizer
    model = UNet_2_5D(n_channels_in=N_INPUT_CHANNELS, n_channels_out=N_OUTPUT_CLASSES).to(DEVICE)

    # --- Use Combined Loss ---
    loss_fn = CombinedLoss()
    # -------------------------

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- Initialize EarlyStopper ---
    early_stopper = EarlyStopper(patience=PATIENCE, verbose=True, path=str(MODEL_SAVE_PATH))
    # -------------------------------

    print("Starting training...")

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [Train]", unit="batch")
        for images, masks in pbar:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(images)
            loss = loss_fn(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_loader)

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [Val]", unit="batch")
            for images, masks in pbar_val:
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)

                outputs = model(images)
                loss = loss_fn(outputs, masks)
                val_loss += loss.item()
                pbar_val.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # --- Call early stopper ---
        early_stopper(avg_val_loss, model)

        if early_stopper.early_stop:
            print("Early stopping triggered")
            break  # Exit the training loop
        # --------------------------

    print("\nTraining complete!")
    print(f"Best model saved to {MODEL_SAVE_PATH} (Min Val Loss: {early_stopper.val_loss_min:.4f})")


if __name__ == "__main__":
    train()