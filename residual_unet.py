import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        return x + self.conv2(self.relu(self.conv1(x)))

class DoubleConvRes(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConvRes, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)

class ResidualUNetSR(nn.Module):
    """
    Input: 1-channel TIR (256x256)
    Encoder maps details -> Bottleneck features Residual Blocks -> Expanded out to 512x512
    """
    def __init__(self, in_channels=1, out_channels=1):
        super(ResidualUNetSR, self).__init__()
        
        # Encoder (Contracting Path)
        self.inc = DoubleConvRes(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConvRes(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConvRes(128, 256))
        
        # Residual Bottleneck
        self.bottleneck = nn.Sequential(
            ResidualBlock(256),
            ResidualBlock(256)
        )
        
        # Decoder (Expanding Path with ConvTranspose2d up-convolution)
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConvRes(256, 128) # Concatenates (128 skip + 128 up) = 256 channels
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConvRes(128, 64)   # Concatenates (64 skip + 64 up) = 128 channels
        
        # Ultimate Decoder Extension to reach 512x512 target boundaries
        self.final_upsample = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # Encoder skip extraction layers
        x1 = self.inc(x)         # [B, 64, 256, 256]
        x2 = self.down1(x1)      # [B, 128, 128, 128]
        x3 = self.down2(x2)      # [B, 256, 64, 64]
        
        b = self.bottleneck(x3)  # [B, 256, 64, 64]
        
        # Decoding stages
        u1 = self.up1(b)         # [B, 128, 128, 128]
        u1 = torch.cat([u1, x2], dim=1)
        u1 = self.conv_up1(u1)
        
        u2 = self.up2(u1)        # [B, 64, 256, 256]
        u2 = torch.cat([u2, x1], dim=1)
        u2 = self.conv_up2(u2)
        
        return self.final_upsample(u2) # Up-scales [B, 64, 256, 256] -> [B, 1, 512, 512]

if __name__ == "__main__":
    model = ResidualUNetSR(in_channels=1, out_channels=1)
    dummy_input = torch.randn(1, 1, 256, 256)
    output = model(dummy_input)
    print(f"=== Residual UNet Shape Verification ===")
    print(f"Input Shape:  {dummy_input.shape}")
    print(f"Output Shape: {output.shape} (Expected: [1, 1, 512, 512])")