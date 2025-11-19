import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import glob
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import random
from pathlib import Path  # Import Path

# --- Import our model ---
try:
    from model_2_5D import UNet_2_5D
except ImportError:
    print("Error: Could not import UNet_2_5D from model_2_5D.py")
    exit()

# --- Configuration ---
DATA_DIR = Path.cwd() / "data_2_5d_tumor_GT"
IMAGE_DIR = DATA_DIR / "imagesTr"

N_INPUT_CHANNELS = 5  # Must match the 5 slices we created
N_OUTPUT_CLASSES = 1  # 1 channel for binary segmentation (tumor vs. not-tumor)

BATCH_SIZE = 8
LEARNING_RATE = 1e-4  # Initial learning rate
NUM_EPOCHS = 100  # Set high, EarlyStopping will handle the rest
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PATIENCE = 10

# --- UPDATED: Save path inside trained_models/ folder ---
MODEL_SAVE_DIR = Path.cwd() / "trained_models"
MODEL_SAVE_PATH = MODEL_SAVE_DIR / "tumor_model_best.pth"


# --------------------------------------------------------


# -----------------------------------------------------------------
# --- Focal Loss Class (Better than BCE for imbalance) ---
# -----------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets):
        bce_loss = self.bce_with_logits(logits, targets)

        probas = torch.sigmoid(logits)
        targets = targets.float()

        # Calculate pt (probability of the correct class)
        pt = probas * targets + (1 - probas) * (1 - targets)

        # Calculate alpha_t (weight for the class)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Calculate focal loss
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# -----------------------------------------------------------------
# --- Dice Loss Class ---
# -----------------------------------------------------------------
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probas = torch.sigmoid(logits)
        targets = targets.float()
        probas = probas.view(probas.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        intersection = (probas * targets).sum()
        cardinality = probas.sum() + targets.sum()
        dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return dice_loss


# -----------------------------------------------------------------
# --- Combined Loss (Focal + Dice) ---
# -----------------------------------------------------------------
class CombinedLoss(nn.Module):
    def __init__(self, focal_weight=0.5, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        self.dice_loss = DiceLoss()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        focal = self.focal_loss(logits, targets)
        dice = self.dice_loss(logits, targets)

        return (self.focal_weight * focal) + (self.dice_weight * dice)


# -----------------------------------------------------------------


# -----------------------------------------------------------------
# --- CORRECTED TumorDataset with Augmentation ---
# -----------------------------------------------------------------
class TumorDataset(Dataset):
    def __init__(self, image_paths, augment=False):
        self.image_paths = image_paths
        self.augment = augment

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = str(image_path).replace("imagesTr", "labelsTr")

        image_data = np.load(image_path)
        mask_data = np.load(mask_path)

        if mask_data.ndim == 2:
            mask_data = np.expand_dims(mask_data, axis=0)

        if self.augment:
            if random.random() > 0.5:
                image_data = np.flip(image_data, axis=2).copy()
                mask_data = np.flip(mask_data, axis=2).copy()
            if random.random() > 0.5:
                image_data = np.flip(image_data, axis=1).copy()
                mask_data = np.flip(mask_data, axis=1).copy()
            k = random.randint(0, 3)
            if k > 0:
                image_data = np.rot90(image_data, k=k, axes=(1, 2)).copy()
                mask_data = np.rot90(mask_data, k=k, axes=(1, 2)).copy()

        image_tensor = torch.from_numpy(image_data).float()
        mask_tensor = torch.from_numpy(mask_data).float()
        return image_tensor, mask_tensor


# -----------------------------------------------------------------

# --- EarlyStopper Class ---
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


def train():
    print(f"Using device: {DEVICE}")
    print("Searching for data in:", IMAGE_DIR)

    all_image_paths = glob.glob(os.path.join(IMAGE_DIR, "*.npy"))
    if not all_image_paths:
        print("Error: No .npy files found in", IMAGE_DIR)
        print("Please run the 'prepare_2_5d_data_tumor.py' script first.")
        return
    print(f"Found {len(all_image_paths)} total 2.5D slices.")

    train_paths, val_paths = train_test_split(all_image_paths, test_size=0.2, random_state=42)
    train_dataset = TumorDataset(train_paths, augment=True)
    val_dataset = TumorDataset(val_paths, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}")

    model = UNet_2_5D(n_channels_in=N_INPUT_CHANNELS, n_channels_out=N_OUTPUT_CLASSES).to(DEVICE)

    # --- Use new CombinedLoss (Focal + Dice) ---
    loss_fn = CombinedLoss(focal_weight=0.5, dice_weight=0.5)
    # ----------------------------------------------------

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    # --- Learning Rate Scheduler (no 'verbose') ---
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.2,
        patience=3
    )
    # -------------------------------------------

    early_stopper = EarlyStopper(patience=PATIENCE, verbose=True, path=str(MODEL_SAVE_PATH))

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

        # --- Get current Learning Rate for printing ---
        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {current_lr:1.0e}")

        # --- Step the scheduler ---
        scheduler.step(avg_val_loss)
        # --------------------------

        # Call early stopper
        early_stopper(avg_val_loss, model)
        if early_stopper.early_stop:
            print("Early stopping triggered")
            break

    print("\nTraining complete!")
    print(f"Best model saved to {MODEL_SAVE_PATH} (Min Val Loss: {early_stopper.val_loss_min:.4f})")


if __name__ == "__main__":
    train()