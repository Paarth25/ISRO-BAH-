import torch
import torch.nn as nn

class ResidualMlpBlock(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )
        self.norm = nn.LayerNorm(dim)
    def forward(self, x): 
        return x + self.norm(self.net(x))

class SwinIRLight(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, embed_dim=64, pre_upscale=False):
        super().__init__()
        self.pre_upscale = pre_upscale
        self.conv_first = nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1)
        
        self.transformer_block1 = ResidualMlpBlock(embed_dim, embed_dim * 2)
        self.transformer_block2 = ResidualMlpBlock(embed_dim, embed_dim * 2)
        
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)
        
        if not self.pre_upscale:
            self.upsampler = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.Conv2d(embed_dim, out_channels, kernel_size=3, padding=1)
            )
        else:
            self.upsampler = nn.Conv2d(embed_dim, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        fe_init = self.conv_first(x)
        B, C, H, W = fe_init.shape
        tokens = fe_init.flatten(2).transpose(1, 2)
        
        tokens = self.transformer_block1(tokens)
        tokens = self.transformer_block2(tokens)
        
        fe_mid = tokens.transpose(1, 2).view(B, C, H, W)
        fe_deep = self.conv_after_body(fe_mid) + fe_init
        return self.upsampler(fe_deep)