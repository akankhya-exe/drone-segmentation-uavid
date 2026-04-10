import torch
import torch.nn as nn

class ECALayer(nn.Module):
    def __init__(self, k_size=3):
        super(ECALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x).view(b, 1, c) 
        y = self.conv(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class AtrousECAConv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.eca = ECALayer()

    def forward(self, x):
        x = self.conv_block(x)
        x = self.eca(x)
        return x

class CustomAtrousECAUNet(nn.Module):
    def __init__(self, in_channels=3, classes=4):
        super(CustomAtrousECAUNet, self).__init__()
        
        # ==========================================
        # THE STEM (The Front Door Double-Jump)
        # ==========================================
        # Jump 1: Stride=2 Conv (512x512 -> 256x256)
        self.stem_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # Jump 2: MaxPool (256x256 -> 128x128)
        self.stem_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ==========================================
        # THE ENCODER (The 4 Floors)
        # ==========================================
        # Floor 1: Stays at 128x128, 64 channels
        self.enc1 = AtrousECAConv(64, 64, dilation=1)
        
        # Floor 2: Drops to 64x64, 128 channels
        self.pool2 = nn.MaxPool2d(2)
        self.enc2 = AtrousECAConv(64, 128, dilation=1)
        
        # Floor 3: Drops to 32x32, 256 channels
        self.pool3 = nn.MaxPool2d(2)
        self.enc3 = AtrousECAConv(128, 256, dilation=2) # Atrous kicks in for context!
        
        # Floor 4 (Bottleneck): Drops to 16x16, 512 channels
        self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = AtrousECAConv(256, 512, dilation=4) # Massive context via dilation
        
        # ==========================================
        # THE DECODER (Exact torchinfo channel math)
        # ==========================================
        self.up4 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec4 = AtrousECAConv(768, 256, dilation=1) # 512 (Upsampled) + 256 (Floor 3 Skip) = 768
        
        self.up3 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.dec3 = AtrousECAConv(384, 128, dilation=1) # 256 (Upsampled) + 128 (Floor 2 Skip) = 384
        
        self.up2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.dec2 = AtrousECAConv(192, 64, dilation=1)  # 128 (Upsampled) + 64 (Floor 1 Skip) = 192
        
        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1 = AtrousECAConv(128, 32, dilation=1)  # 64 (Upsampled) + 64 (Stem Skip) = 128
        
        self.up0 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.dec0 = AtrousECAConv(32, 16, dilation=1)   # Final resolution push (512x512)
        
        # SEGMENTATION HEAD
        self.out_conv = nn.Conv2d(16, classes, kernel_size=1)

    def forward(self, x):
        # --- THE STEM ---
        e1_stem = self.stem_conv(x)          # Save [B, 64, 256, 256] for late skip connection
        x = self.stem_pool(e1_stem)          # Compresses to [B, 64, 128, 128]
        
        # --- ENCODER ---
        e2 = self.enc1(x)                    # Save [B, 64, 128, 128]
        e3 = self.enc2(self.pool2(e2))       # Save [B, 128, 64, 64]
        e4 = self.enc3(self.pool3(e3))       # Save [B, 256, 32, 32]
        b = self.bottleneck(self.pool4(e4))  # BOTTLENECK [B, 512, 16, 16]
        
        # --- DECODER ---
        d4 = self.up4(b)
        d4 = torch.cat((e4, d4), dim=1)      # Catches 256 channels from e4
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        d3 = torch.cat((e3, d3), dim=1)      # Catches 128 channels from e3
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat((e2, d2), dim=1)      # Catches 64 channels from e2
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat((e1_stem, d1), dim=1) # Catches the 64 channels from the Stem!
        d1 = self.dec1(d1)
        
        d0 = self.up0(d1)
        d0 = self.dec0(d0)
        
        return self.out_conv(d0)
