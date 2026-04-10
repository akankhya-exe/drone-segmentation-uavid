import torch
import time
import numpy as np
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from models.custom_unet import CustomAtrousECAUNet
# Assuming the UAVidDataset class is in your train.py, or import it
from train import UAVidDataset 

def evaluate_model(model_path, model_type="CUSTOM"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model
    if model_type == "BASELINE":
        model = smp.Unet(encoder_name="resnet18", classes=4).to(device)
    else:
        model = CustomAtrousECAUNet(in_channels=3, classes=4).to(device)
    
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 2. Setup Test Data (The 2 images we hid)
    test_ds = UAVidDataset("data/patches/images", "data/patches/labels", mode="val")
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    # 3. Initialize Metrics
    tp, fp, fn, tn = [], [], [], []
    latencies = []

    print(f"Evaluating {model_type}...")

    with torch.no_grad():
        for imgs, masks in test_loader:
            imgs, masks = imgs.to(device), masks.to(device)

            # LATENCY MEASUREMENT
            start_time = time.perf_counter()
            output = model(imgs)
            end_time = time.perf_counter()
            latencies.append(end_time - start_time)

            # METRIC CALCULATION
            # Convert output to class predictions
            preds = torch.argmax(output, dim=1)
            
            # Use SMP's built-in metric helpers
            stats = smp.metrics.get_stats(preds, masks, mode='multiclass', num_classes=4)
            tp.append(stats[0])
            fp.append(stats[1])
            fn.append(stats[2])
            tn.append(stats[3])

    # 4. Aggregating Results
    tp = torch.cat(tp)
    fp = torch.cat(fp)
    fn = torch.cat(fn)
    tn = torch.cat(tn)

    miou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
    f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
    avg_latency = np.mean(latencies[5:]) * 1000 # Skip first 5 (warmup), convert to ms

    print(f"\n--- {model_type} RESULTS ---")
    print(f"mIoU Score:     {miou:.4f}")
    print(f"F1 (Dice):      {f1:.4f}")
    print(f"Avg Latency:    {avg_latency:.2f} ms")
    print("-" * 25)

if __name__ == "__main__":
    # Run for Baseline
    evaluate_model("best_model_baseline.pth", "BASELINE")
    # Run for Your Architecture
    evaluate_model("best_model_custom.pth", "CUSTOM")