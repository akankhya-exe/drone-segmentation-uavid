import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import segmentation_models_pytorch as smp
from models.custom_unet import CustomAtrousECAUNet
import glob
import random
import torchvision.transforms.functional as TF

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

MODEL_TYPE = "CUSTOM"  
EPOCHS = 50           
BATCH_SIZE = 8         
LEARNING_RATE = 1e-4

class UAVidDataset(Dataset):
    def __init__(self, base_dir, mode="train", transform=True):
        self.mode = mode
        self.transform = transform
        
        self.image_paths = sorted(glob.glob(os.path.join(base_dir, mode, "seq*", "Images", "*.png")))
        self.mask_paths = sorted(glob.glob(os.path.join(base_dir, mode, "seq*", "Labels", "*.png")))
        
        if len(self.image_paths) == 0:
            raise RuntimeError(f"Found 0 images in {base_dir}. Check your pathing!")

    def __len__(self):
        return len(self.image_paths)

    def augment(self, image, mask):
        # Horizontal flip
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        # Vertical flip
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        return image, mask

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        
        img = np.transpose(img, (2, 0, 1)).astype(np.float32) / 255.0
        img_tensor = torch.tensor(img)
        mask_tensor = torch.tensor(mask).long()

        if self.mode == "train" and self.transform:
            img_tensor, mask_tensor = self.augment(img_tensor, mask_tensor)

        return img_tensor, mask_tensor

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    save_dir = "/content/drive/MyDrive/drone_model_checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"best_model_{MODEL_TYPE.lower()}.pth")

    data_root = "/content/data" 
    train_ds = UAVidDataset(data_root, mode="train", transform=True)
    val_ds = UAVidDataset(data_root, mode="val", transform=False)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    if MODEL_TYPE == "BASELINE":
        model = smp.Unet(encoder_name="resnet18", encoder_weights="imagenet", in_channels=3, classes=4).to(device)
    else:
        model = CustomAtrousECAUNet(in_channels=3, classes=4).to(device)

    criterion = lambda out, m: smp.losses.FocalLoss(mode='multiclass')(out, m) + \
                               smp.losses.DiceLoss(mode='multiclass')(out, m)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    print(f"STARTING TRAINING: {MODEL_TYPE}")
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    best_v_loss = float('inf')
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

        model.eval()
        v_loss = 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                v_loss += criterion(model(imgs), masks).item()

        avg_t = t_loss/len(train_loader)
        avg_v = v_loss/len(val_loader)
        
        scheduler.step(avg_v)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_t:.4f} | Val Loss: {avg_v:.4f} | LR: {optimizer.param_groups[0]['lr']}")

        if avg_v < best_v_loss:
            best_v_loss = avg_v
            torch.save(model.state_dict(), save_path)
            print(f" >>> New Best Model Saved to Drive!")

if __name__ == "__main__":
    main()