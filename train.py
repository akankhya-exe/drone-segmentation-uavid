import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import segmentation_models_pytorch as smp
from models.custom_unet import CustomAtrousECAUNet


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    # This makes the training deterministic (but might slow it down slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- THE "SWITCHBOARD" ---
MODEL_TYPE = "CUSTOM"  # Change this to "CUSTOM" after the first run
EPOCHS = 20
BATCH_SIZE = 4 
VAL_IMAGES = ["000800", "000900"] # We'll hide these from the training

class UAVidDataset(Dataset):
    def __init__(self, img_dir, mask_dir, mode="train"):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        # Get all patch filenames
        all_patches = [p for p in os.listdir(img_dir) if p.endswith('.png')]
        
        # LOGICAL SPLIT: Filter patches based on the original image number in the name
        if mode == "train":
            self.images = [p for p in all_patches if not any(v in p for v in VAL_IMAGES)]
        else:
            self.images = [p for p in all_patches if any(v in p for v in VAL_IMAGES)]

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img = cv2.imread(os.path.join(self.img_dir, img_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(self.mask_dir, img_name), cv2.IMREAD_GRAYSCALE)
        
        # Normalize and convert to Tensor
        img = np.transpose(img, (2, 0, 1)).astype(np.float32) / 255.0
        return torch.tensor(img), torch.tensor(mask).long()

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = f"best_model_{MODEL_TYPE.lower()}.pth"

    # 1. Setup the split loaders
    train_ds = UAVidDataset("data/patches/images", "data/patches/labels", mode="train")
    val_ds = UAVidDataset("data/patches/images", "data/patches/labels", mode="val")
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 2. Pick the Model
    if MODEL_TYPE == "BASELINE":
        model = smp.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=3, classes=4).to(device)
    else:
        model = CustomAtrousECAUNet(in_channels=3, classes=4).to(device)

    # 3. Loss & Optimizer (Focal + Dice)
    focal_loss = smp.losses.FocalLoss(mode='multiclass')
    dice_loss = smp.losses.DiceLoss(mode='multiclass')
    criterion = lambda out, m: focal_loss(out, m) + dice_loss(out, m)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    print(f"RUNNING: {MODEL_TYPE} | Training on {len(train_ds)} patches | Validating on {len(val_ds)} patches")

    best_loss = float('inf')
    for epoch in range(EPOCHS):
        model.train()
        t_loss = 0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), masks)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()

        # Validation Check
        model.eval()
        v_loss = 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                v_loss += criterion(model(imgs), masks).item()

        avg_t = t_loss/len(train_loader)
        avg_v = v_loss/len(val_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Train: {avg_t:.4f} | Val: {avg_v:.4f}")

        if avg_v < best_loss:
            best_loss = avg_v
            torch.save(model.state_dict(), save_path)
            print(f" >>> Saved {MODEL_TYPE} weights!")

if __name__ == "__main__":
    main()