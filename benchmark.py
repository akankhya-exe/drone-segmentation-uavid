import torch
from ptflops import get_model_complexity_info

from models.custom_unet import CustomAtrousECAUNet

def main():
    print("Building Custom Atrous-ECA U-Net")
    
    custom_model = CustomAtrousECAUNet(in_channels=3, classes=4)

    macs, params = get_model_complexity_info(
        custom_model, 
        (3, 512, 512), 
        as_strings=True,
        print_per_layer_stat=False, 
        verbose=False
    )

    print("\n" + "="*40)
    print("    CUSTOM ATROUS-ECA BENCHMARKS")
    print("="*40)
    print('{:<35}  {:<8}'.format('Computational complexity (MACs):', macs))
    print('{:<35}  {:<8}'.format('Number of parameters:', params))
    print("="*40)

if __name__ == "__main__":
    main()