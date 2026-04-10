import segmentation_models_pytorch as smp
from torchinfo import summary

def main():
    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights=None, 
        in_channels=3,
        classes=4
    )

    summary(
        model, 
        input_size=(1, 3, 512, 512),
        col_names=["input_size", "output_size", "num_params", "kernel_size"],
        depth=4
    )

if __name__ == "__main__":
    main()