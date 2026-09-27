import torch
import torch.nn as nn
import torchvision.models as models
from .unet import UNetBlock 

class ResNet34UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, pre_upscale=False):
        super().__init__()
        self.pre_upscale = pre_upscale
        
        if self.pre_upscale:
            self.upscale_layer = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1)
            
        self.input_adapter = nn.Conv2d(in_channels, 3, kernel_size=1)
        base_model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        base_layers = list(base_model.children())
        
        # ResNet Encoder Stages
        self.layer0 = nn.Sequential(*base_layers[:3])      # Out: 64 channels
        self.layer0_pool = base_layers[3]                  # MaxPool
        self.layer1 = base_layers[4]                       # Out: 64 channels
        self.layer2 = base_layers[5]                       # Out: 128 channels
        self.layer3 = base_layers[6]                       # Out: 256 channels
        self.layer4 = base_layers[7]                       # Out: 512 channels
        
        # Symmetric Decoder Blocks
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = UNetBlock(512, 256) 
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = UNetBlock(256, 128) 
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = UNetBlock(128, 64)  
        
        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1 = UNetBlock(128, 64) 

        # New Final 5th Upsampler to recover from ResNet's aggressive front end downsampling
        self.up0 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec0 = UNetBlock(32, 32)
        
        self.final = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        if self.pre_upscale: 
            x = self.upscale_layer(x)
            
        x_mapped = self.input_adapter(x)
        
        # Encoder Pipeline
        x0 = self.layer0(x_mapped)
        x1 = self.layer1(self.layer0_pool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        
        # Decoder Pipeline
        d4 = self.dec4(torch.cat([self.up4(x4), x3], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), x2], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), x1], dim=1))
        
        up1_out = self.up1(d2)
        if up1_out.shape[2:] != x0.shape[2:]:
            up1_out = torch.nn.functional.interpolate(up1_out, size=x0.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([up1_out, x0], dim=1))
        
        # 5th Block to scale from 256x256 back up to full 512x512 resolution
        d0 = self.dec0(self.up0(d1))
        
        return self.final(d0)