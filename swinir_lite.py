import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleSwinTransformerBlock(nn.Module):
    """
    An optimized Transformer block variant using grouped depthwise operations 
    to emulate spatial token mixing without risking high out-of-memory errors on Quadro cards.
    """
    def __init__(self, dim):
        super(SimpleSwinTransformerBlock, self).__init__()
        self.proj = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.attn_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim) 
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim, kernel_size=1)
        )
        self.norm1 = nn.GroupNorm(num_groups=1, num_channels=dim)
        self.norm2 = nn.GroupNorm(num_groups=1, num_channels=dim)

    def forward(self, x):
        # Attention Approximation Loop
        x = x + self.attn_conv(F.gelu(self.proj(self.norm1(x))))
        # Multi-Layer Perceptron (MLP) Shift
        x = x + self.mlp(self.norm2(x))
        return x

class SwinIRLite(nn.Module):
    """
    Input: 1-channel TIR (256x256)
    Output: High-fidelity upscale (512x512) via a lightweight Swin block structure.
    """
    def __init__(self, in_channels=1, out_channels=1, embed_dim=64):
        super(SwinIRLite, self).__init__()
        self.conv_first = nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1)
        
        # Deep Feature Extraction Cascade
        self.layer1 = SimpleSwinTransformerBlock(embed_dim)
        self.layer2 = SimpleSwinTransformerBlock(embed_dim)
        self.layer3 = SimpleSwinTransformerBlock(embed_dim)
        
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)
        
        # Sub-Pixel Spatial Reconstruction
        self.upsample = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.GELU()
        )
        self.conv_last = nn.Conv2d(embed_dim, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        feats_shallow = self.conv_first(x)
        x_feat = self.layer1(feats_shallow)
        x_feat = self.layer2(x_feat)
        x_feat = self.layer3(x_feat)
        x_feat = self.conv_after_body(x_feat) + feats_shallow # Global Deep Residual Link
        
        return self.conv_last(self.upsample(x_feat))

if __name__ == "__main__":
    model = SwinIRLite(in_channels=1, out_channels=1)
    dummy_input = torch.randn(1, 1, 256, 256)
    output = model(dummy_input)
    print(f"=== SwinIR Lite Shape Verification ===")
    print(f"Input Shape:  {dummy_input.shape}")
    print(f"Output Shape: {output.shape} (Expected: [1, 1, 512, 512])")