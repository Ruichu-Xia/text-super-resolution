import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return out + residual
    

class UpsampleBlock(nn.Module):
    def __init__(self, channels: int, upscale_factor: int):
        super(UpsampleBlock, self).__init__()
        self.upsample = nn.Sequential(
            nn.Conv2d(channels, channels * (upscale_factor ** 2), kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(upscale_factor),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.upsample(x)
    

class ResNetSR(nn.Module):
    def __init__(self, upscale_factor=2, num_channels=1, num_features=64, num_res_blocks=16):
        super(ResNetSR, self).__init__()
        
        # Calculate number of upsampling steps needed
        self.num_upsamples = int(math.log2(upscale_factor))
        if 2 ** self.num_upsamples != upscale_factor:
            raise ValueError("Upscale factor must be a power of 2 (2, 4, 8, etc)")
            
        # Initial conv
        self.entry = nn.Conv2d(num_channels, num_features, kernel_size=3, padding=1)
        
        # Residual blocks
        res_blocks = []
        for _ in range(num_res_blocks):
            res_blocks.append(ResidualBlock(num_features))
        self.res_blocks = nn.Sequential(*res_blocks)
        
        # Multiple upsampling layers
        self.upsampling = nn.Sequential()
        current_features = num_features
        for i in range(self.num_upsamples):
            self.upsampling.add_module(f'upsample_{i}', UpsampleBlock(current_features, 2))
            
        # Final output layer
        self.exit = nn.Conv2d(num_features, num_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # Initial conv
        out = self.entry(x)
        
        # Residual blocks
        residual = out
        out = self.res_blocks(out)
        out = out + residual
        
        # Upsampling
        out = self.upsampling(out)
        
        # Final conv
        out = self.exit(out)
        
        return out
    
def test():
    x = torch.randn(3, 3, 256, 256)
    model = ResNetSR(2)
    print(model(x).shape)

if __name__ == "__main__":
    test()