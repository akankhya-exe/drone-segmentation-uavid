import os
import torch
import numpy as np
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from models.custom_unet import CustomAtrousECAUNet

from train import UAVidDataset 

def evaluate_model(model_path, model_type="CUSTOM", data_root="data"): # def evaluate_model(model_path, model_type="CUSTOM", data_root="/content/data"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_type in ["BASELINE", "BASELINE_SCRATCH"]:
        model = smp.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=3, classes=4).to(device)
    elif model_type == "CUSTOM":
        model = CustomAtrousECAUNet(in_channels=3, classes=4).to(device)
    else:
        raise ValueError("Invalid model_type provided.")
    
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    test_ds = UAVidDataset(data_root, mode="val")
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    tp, fp, fn, tn = [], [], [], []
    latencies = []

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    print(f"\nEvaluating {model_type}")

    print("Warming up GPU")
    dummy_input = torch.randn(1, 3, 512, 512, dtype=torch.float).to(device)
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    with torch.no_grad():
        for i, (imgs, masks) in enumerate(test_loader):
            imgs, masks = imgs.to(device), masks.to(device)

            starter.record()
            output = model(imgs)
            ender.record()
            
            torch.cuda.synchronize() 
            
            if i > 5:
                latencies.append(starter.elapsed_time(ender))

            preds = torch.argmax(output, dim=1)
            stats = smp.metrics.get_stats(preds, masks, mode='multiclass', num_classes=4)
            tp.append(stats[0])
            fp.append(stats[1])
            fn.append(stats[2])
            tn.append(stats[3])

    tp = torch.cat(tp)
    fp = torch.cat(fp)
    fn = torch.cat(fn)
    tn = torch.cat(tn)

    miou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
    f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
    
    avg_latency = np.mean(latencies) 
    fps = 1000 / avg_latency

    print(f"\n{model_type} RESULTS ")
    print(f"mIoU Score:    {miou:.4f}")
    print(f"F1 (Dice):     {f1:.4f}")
    print(f"Avg Latency:   {avg_latency:.2f} ms")
    print(f"Speed:         {fps:.2f} FPS")
    print("====================================")

if __name__ == "__main__":
    weights_dir = "weights"
    
    print("Starting Head-to-Head Evaluation")
    
    custom_path = os.path.join(weights_dir, "best_model_custom.pth")
    if os.path.exists(custom_path):
        evaluate_model(custom_path, "CUSTOM")
    else:
        print(f"Skipping Custom: Could not find {custom_path}")
        
    baseline_path = os.path.join(weights_dir, "best_model_baseline.pth")
    if os.path.exists(baseline_path):
        evaluate_model(baseline_path, "BASELINE")
    else:
        print(f"Skipping Baseline: Could not find {baseline_path}")

    scratch_path = os.path.join(weights_dir, "best_model_baseline_scratch.pth")
    if os.path.exists(scratch_path):
        evaluate_model(scratch_path, "BASELINE_SCRATCH")
    else:
        print(f"Skipping Scratch Baseline: Could not find {scratch_path}")