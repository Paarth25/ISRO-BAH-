import torch
import torch.nn as nn

class UNetBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class AttentionGate(nn.Module):
    """
    Gated Attention Mechanism. Filters low-level encoder features (x)
    using higher-level decoder context (g) before combining them.
    """
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        # Handle spatial alignment if dimensions differ slightly due to padding
        if g1.shape[2:] != x1.shape[2:]:
            g1 = torch.nn.functional.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=False)
        
        avg = self.relu(g1 + x1)
        attention_coefficients = self.psi(avg)
        return x * attention_coefficients

class AttentionUNetStandard(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.enc1 = UNetBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = UNetBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = UNetBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        
        self.bottleneck = UNetBlock(256, 512)
        
        # Attention Gates guarding the skip connections
        self.att3 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = UNetBlock(512, 256)
        
        self.att2 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = UNetBlock(256, 128)
        
        self.att1 = AttentionGate(F_g=64, F_l=64, F_int=32)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = UNetBlock(128, 64)
        
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool1(s1))
        s3 = self.enc3(self.pool2(s2))
        b = self.bottleneck(self.pool3(s3))
        
        # Up-sample & apply gated attention filter over the skip tensor
        up3_out = self.up3(b)
        s3_filtered = self.att3(g=up3_out, x=s3)
        d3 = self.dec3(torch.cat([up3_out, s3_filtered], dim=1))
        
        up2_out = self.up2(d3)
        s2_filtered = self.att2(g=up2_out, x=s2)
        d2 = self.dec2(torch.cat([up2_out, s2_filtered], dim=1))
        
        up1_out = self.up1(d2)
        s1_filtered = self.att1(g=up1_out, x=s1)
        d1 = self.dec1(torch.cat([up1_out, s1_filtered], dim=1))
        
        return self.final(d1)

class AttentionUNetUpscale(nn.Module):
    """
    Applies initial ConvTranspose2d upscaling for Super-Resolution
    before feeding into the Attention-Gated UNet context.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.initial_upscale = nn.ConvTranspose2d(in_channels, 32, kernel_size=4, stride=2, padding=1)
        self.base_unet = AttentionUNetStandard(in_channels=32, out_channels=out_channels)
    def forward(self, x): 
        return self.base_unet(self.initial_upscale(x))