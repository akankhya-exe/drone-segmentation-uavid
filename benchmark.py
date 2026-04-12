import torch
import segmentation_models_pytorch as smp
from ptflops import get_model_complexity_info

from models.custom_unet import CustomAtrousECAUNet

def benchmark_model(model, name):
    print(f"Profiling {name}...")
    macs, params = get_model_complexity_info(
        model, 
        (3, 512, 512), 
        as_strings=True,
        print_per_layer_stat=False, 
        verbose=False
    )
    return macs, params

def main():
    print("="*50)
    print("      ARCHITECTURE BENCHMARKS (512x512)      ")
    print("="*50)
 
    custom_model = CustomAtrousECAUNet(in_channels=3, classes=4)
    custom_macs, custom_params = benchmark_model(custom_model, "Custom Atrous-ECA")

    baseline_model = smp.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=3, classes=4)
    base_macs, base_params = benchmark_model(baseline_model, "ResNet-18 Baseline")

    print("\n" + "="*50)
    print(f"{'Model':<25} | {'Parameters':<10} | {'MACs':<10}")
    print("-" * 50)
    print(f"{'ResNet-18 Baseline':<25} | {base_params:<10} | {base_macs:<10}")
    print(f"{'Custom Atrous-ECA':<25} | {custom_params:<10} | {custom_macs:<10}")
    print("="*50)

if __name__ == "__main__":
    main()