import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, upscale_factor: int, num_res_blocks=8, num_channels=3, num_features=64):
        super(ResNetSR, self).__init__()
    
        self.entry = nn.Conv2d(num_channels, num_features, kernel_size=3, stride=1, padding=1)

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_features) for _ in range(num_res_blocks)]
        )

        self.mid_conv = nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1)

        upsampling = []
        if upscale_factor in [2, 4, 8]:
            for _ in range(int(torch.log2(torch.tensor(upscale_factor)))):
                upsampling.append(UpsampleBlock(num_features, 2))
        else:
            print("Upsacle factor not accepted!")
        self.upsampling = nn.Sequential(*upsampling)

        self.exit = nn.Conv2d(num_features, num_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = F.interpolate(x, scale_factor=self.upsampling[0].upsample[1].upscale_factor, mode='bicubic', align_corners=False)

        out = self.entry(x)
        out = self.res_blocks(out)
        out = self.mid_conv(out)
        out = self.upsampling(out)
        out = self.exit(out)
        out = out + residual

        return out
    
def test():
    x = torch.randn(3, 3, 256, 256)
    model = ResNetSR(2)
    print(model(x).shape)

if __name__ == "__main__":
    test()