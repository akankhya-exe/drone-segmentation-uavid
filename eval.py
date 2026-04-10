import os
import torch
import numpy as np
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from models.custom_unet import CustomAtrousECAUNet

from train import UAVidDataset 

def evaluate_model(model_path, model_type="CUSTOM", data_root="/content/data"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_type == "BASELINE":
        model = smp.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=3, classes=4).to(device)
    else:
        model = CustomAtrousECAUNet(in_channels=3, classes=4).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    test_ds = UAVidDataset(data_root, mode="val")
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    tp, fp, fn, tn = [], [], [], []
    latencies = []

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    print(f"\nEvaluating {model_type}...")

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

    print(f"\n======== {model_type} RESULTS ========")
    print(f"mIoU Score:    {miou:.4f}")
    print(f"F1 (Dice):     {f1:.4f}")
    print(f"Avg Latency:   {avg_latency:.2f} ms")
    print(f"Speed:         {fps:.2f} FPS")
    print("====================================")

if __name__ == "__main__":
    drive_path = "/content/drive/MyDrive/drone_model_checkpoints"
    
    print("Starting Head-to-Head Evaluation...")
    
    evaluate_model(f"{drive_path}/best_model_custom.pth", "CUSTOM")
    
    evaluate_model(f"{drive_path}/best_model_baseline.pth", "BASELINE")