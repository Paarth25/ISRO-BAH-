import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        # Removed BatchNorm layers to follow standard EDSR/SRResNet practices 
        # (This prevents clipping raw satellite dynamic pixel intensities)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        return x + self.conv2(self.relu(self.conv1(x)))

class SRResNet(nn.Module):
    """
    Input: 1-channel TIR (256x256)
    Output: 1-channel TIR (512x512) via PixelShuffle Sub-Pixel Convolution
    """
    def __init__(self, in_channels=1, out_channels=1, num_features=64, num_blocks=12):
        super(SRResNet, self).__init__()
        
        # Head (Shallow Feature Extraction)
        self.head = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)
        
        # Deep Residual Core
        self.body = nn.Sequential(*[ResidualBlock(num_features) for _ in range(num_blocks)])
        self.mid_conv = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        
        # Tail Upsampler: Sub-Pixel Convolution (PixelShuffle x2)
        # 2x upsampling requires channel expansion to features * (2^2) = features * 4
        self.upsample = nn.Sequential(
            nn.Conv2d(num_features, num_features * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2), 
            nn.ReLU(inplace=True)
        )
        
        # Final Reconstruction
        self.tail = nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x_head = self.head(x)
        x_body = self.body(x_head)
        x_body = self.mid_conv(x_body) + x_head  # Global Residual Learning Connection
        x_up = self.upsample(x_body)
        return self.tail(x_up)

if __name__ == "__main__":
    # Smoke test shape assertions
    model = SRResNet(in_channels=1, out_channels=1)
    dummy_input = torch.randn(1, 1, 256, 256) # [Batch, Channels, H, W]
    output = model(dummy_input)
    print(f"=== SRResNet Shape Verification ===")
    print(f"Input Shape:  {dummy_input.shape}")
    print(f"Output Shape: {output.shape} (Expected: [1, 1, 512, 512])")