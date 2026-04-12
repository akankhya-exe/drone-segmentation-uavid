import glob
import os
from pathlib import Path
import cv2
import numpy as np
import albumentations as albu
from utils.tools import rgb2label, seed

'''Converts images to patches for processing batch-wise'''

# Directories
INPUT_IMG_DIR = "data/raw_images"
INPUT_MASK_DIR = "data/raw_labels"
OUT_IMG_DIR = "data/patches/images"
OUT_MASK_DIR = "data/patches/labels"
SPLIT_SIZE = 512
STRIDE = 512 # Will be using 512 x 512 patches

def padifneeded(image, mask):
    """Pads the 3840x2160 image to 4096x2160 to make the math work."""
    pad = albu.PadIfNeeded(min_height=2160, min_width=4096, position='bottom_right',
                           border_mode=0, value=[0, 0, 0], mask_value=[255, 255, 255])(image=image, mask=mask)
    return pad['image'], pad['mask']

def main():
    seed(42)

    os.makedirs(OUT_IMG_DIR, exist_ok=True)
    os.makedirs(OUT_MASK_DIR, exist_ok=True)

    img_paths = glob.glob(os.path.join(INPUT_IMG_DIR, "*.png"))

    for img_path in img_paths:
        filename = Path(img_path).name
        mask_path = os.path.join(INPUT_MASK_DIR, filename)

        if not os.path.exists(mask_path):
            print(f"Skipping {filename}, no matching mask found in {INPUT_MASK_DIR}.")
            continue
        
        print(f"Chopping and Relabeling: {filename}")
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        img, mask = padifneeded(img, mask)

        mask = rgb2label(mask)

        img, mask = img[-2048:, -4096:, :], mask[-2048:, -4096:]

        k = 0
        for y in range(0, img.shape[0], STRIDE):
            for x in range(0, img.shape[1], STRIDE):
                img_tile = img[y:y + SPLIT_SIZE, x:x + SPLIT_SIZE]
                mask_tile = mask[y:y + SPLIT_SIZE, x:x + SPLIT_SIZE]

                # Save the patch if it's perfectly 512x512
                if img_tile.shape[0] == SPLIT_SIZE and img_tile.shape[1] == SPLIT_SIZE:
                    base_id = Path(filename).stem
                    out_img_name = os.path.join(OUT_IMG_DIR, f"{base_id}_{k}.png")
                    out_mask_name = os.path.join(OUT_MASK_DIR, f"{base_id}_{k}.png")

                    # Convert image back to BGR for saving
                    img_tile = cv2.cvtColor(img_tile, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(out_img_name, img_tile)
                    cv2.imwrite(out_mask_name, mask_tile.astype(np.uint8))
                    k += 1
        
    print("\nChopping complete! Check your data/patches folders.")
        

if __name__ == "__main__":
    main()