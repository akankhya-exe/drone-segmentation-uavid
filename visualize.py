import os
import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
from models.custom_unet import CustomAtrousECAUNet

# --- CONFIGURATION ---
IMG_SIZE = 512 # Input size for the model
TEST_IMAGE_INDEX = 0 # 0-9 to pick which of your 10 raw images to test
# Define the actual RGB colors from the UAVid dataset for clear presentation
COLOR_MAP = np.array([
    [128, 64, 128],  # Class 0: Background/Misc (Purple)
    [0, 255, 0],     # Class 1: Trees/Vegetation (Green)
    [255, 0, 0],     # Class 2: Cars/Vehicles (Red)
    [0, 0, 255]      # Class 3: Road/Clutter (Blue)
], dtype=np.uint8)

def process_full_image(model, img_path, device):
    # 1. Load the original high-res image
    original_img = cv2.imread(img_path)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    h, w, c = original_img.shape
    
    # Check if stitching is necessary (assumes 1024x1024 inputs)
    if h != IMG_SIZE * 2 or w != IMG_SIZE * 2:
        print("Warning: Image is not 1024x1024, resize might be necessary.")
        original_img = cv2.resize(original_img, (IMG_SIZE*2, IMG_SIZE*2))
        h, w = IMG_SIZE*2, IMG_SIZE*2

    # Initialize full mask [1024x1024]
    full_mask = np.zeros((h, w), dtype=np.uint8)

    # 2. Chop, Predict, and Stitch
    with torch.no_grad():
        for i in range(0, h, IMG_SIZE):
            for j in range(0, w, IMG_SIZE):
                # Chop [512, 512, 3]
                patch = original_img[i:i+IMG_SIZE, j:j+IMG_SIZE, :]
                
                # Normalize and prepare for PyTorch [1, 3, 512, 512]
                img_tensor = np.transpose(patch, (2, 0, 1)).astype(np.float32) / 255.0
                img_tensor = torch.tensor(img_tensor).unsqueeze(0).to(device)
                
                # Predict [1, 4, 512, 512]
                output = model(img_tensor)
                
                # Argmax to get class predictions [512, 512]
                pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
                
                # Stitch back into the full mask
                full_mask[i:i+IMG_SIZE, j:j+IMG_SIZE] = pred

    # 3. Apply the Colormap for presentation
    colored_mask = COLOR_MAP[full_mask]
    return colored_mask, cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Generating Visuals on {device}...")

    # Load All Original Raw Image Paths
    raw_dir = "data/patches/images"
    raw_images = [os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.endswith('.png')]
    if not raw_images:
        print("Error: No raw images found in data/raw_images/images")
        return

    img_path = raw_images[TEST_IMAGE_INDEX]
    print(f"Processing image: {os.path.basename(img_path)}")

    # Load Baseline Model & Weights
    baseline_model = smp.Unet(encoder_name="resnet18", classes=4).to(device)
    baseline_model.load_state_dict(torch.load("best_model_baseline.pth"))
    baseline_model.eval()

    # Load Your Custom Model & Weights
    custom_model = CustomAtrousECAUNet(in_channels=3, classes=4).to(device)
    custom_model.load_state_dict(torch.load("best_model_custom.pth"))
    custom_model.eval()

    # Process!
    baseline_mask, bgr_img = process_full_image(baseline_model, img_path, device)
    custom_mask, _ = process_full_image(custom_model, img_path, device)

    # 4. Create the Final "Hero Image" side-by-side [1024x3072]
    # Resize raw image slightly to match masks (needed for cv2.hconcat)
    bgr_img_resized = cv2.resize(bgr_img, (baseline_mask.shape[1], baseline_mask.shape[0]))
    final_view = cv2.hconcat([bgr_img_resized, baseline_mask, custom_mask])

    # Save the result
    out_path = f"results_comparison_image_{TEST_IMAGE_INDEX}.png"
    cv2.imwrite(out_path, final_view)
    print(f"Visual comparison saved to: {out_path}")

if __name__ == "__main__":
    main()