import os
import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
from models.custom_unet import CustomAtrousECAUNet

TEST_IMAGE_INDEX = 5 # Pick any index from patches folder
# Colors in RGB format
COLOR_MAP = np.array([
    [128, 64, 128],  # Class 0: Background/Misc (Purple)
    [0, 255, 0],     # Class 1: Trees/Vegetation (Green)
    [255, 0, 0],     # Class 2: Cars/Vehicles (Red)
    [0, 0, 255]      # Class 3: Road/Clutter (Blue)
], dtype=np.uint8)

def process_patch(model, img_path, device):
    original_img = cv2.imread(img_path)
    original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    img_tensor = np.transpose(original_img_rgb, (2, 0, 1)).astype(np.float32) / 255.0
    img_tensor = torch.tensor(img_tensor).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_tensor)
        pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    colored_mask_rgb = COLOR_MAP[pred]

    colored_mask_bgr = cv2.cvtColor(colored_mask_rgb, cv2.COLOR_RGB2BGR)
    
    return colored_mask_bgr, original_img

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Generating Visuals on {device}")

    img_dir = "data/patches/images"
    raw_images = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.png')])
    
    if not raw_images:
        print(f"Error: No images found in {img_dir}. Did you run chop_data.py?")
        return

    img_path = raw_images[TEST_IMAGE_INDEX]
    print(f"Processing image: {os.path.basename(img_path)}")

    print("Loading Baseline Model")
    baseline_model = smp.Unet(encoder_name="resnet18", encoder_weights=None, classes=4).to(device)
    baseline_model.load_state_dict(torch.load("weights/best_model_baseline.pth", map_location=device, weights_only=True))
    baseline_model.eval()

    print("Loading Custom Model")
    custom_model = CustomAtrousECAUNet(in_channels=3, classes=4).to(device)
    custom_model.load_state_dict(torch.load("weights/best_model_custom.pth", map_location=device, weights_only=True))
    custom_model.eval()

    baseline_mask, bgr_img = process_patch(baseline_model, img_path, device)
    custom_mask, _ = process_patch(custom_model, img_path, device)

    final_view = cv2.hconcat([bgr_img, baseline_mask, custom_mask])

    out_path = f"results_comparison_image_{TEST_IMAGE_INDEX}.png"
    cv2.imwrite(out_path, final_view)
    print(f"Visual comparison saved to: {out_path}")

if __name__ == "__main__":
    main()