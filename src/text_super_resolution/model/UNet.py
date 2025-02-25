import torch
import torch.nn as nn
import torchvision.transforms.functional as TF 


class _ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(_ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return out + residual
    

class _CascadingBlock(nn.Module):
    def __init__(self, channels):
        super(_CascadingBlock, self).__init__()
        self.rb1 = _ResidualBlock(channels)
        self.conv1 = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.rb2 = _ResidualBlock(channels)
        self.conv2 = nn.Conv2d(channels * 3, channels, kernel_size=1)
        self.rb3 = _ResidualBlock(channels)
        self.conv3 = nn.Conv2d(channels * 4, channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        rb1 = self.rb1(x)
        concat1 = torch.cat([rb1, x], dim=1)
        conv1 = self.relu(self.conv1(concat1))

        rb2 = self.rb2(conv1)
        concat2 = torch.cat([concat1, rb2], dim=1)
        conv2 = self.relu(self.conv2(concat2))

        rb3 = self.rb3(conv2)
        concat3 = torch.cat([concat2, rb3], dim=1)
        conv3 = self.relu(self.conv3(concat3))

        return conv3
    

class UpsampleBlock(nn.Module):
    def __init__(self, channels, upscale_factor):
        super(UpsampleBlock, self).__init__()
        self.upsample_block = nn.Sequential(
            nn.Conv2d(channels, channels * upscale_factor ** 2, kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.upsample_block(x)
    

class CARN(nn.Module):
    """ Cascading Residual Network for Super-Resolution """
    def __init__(self, upscale_factor: int):
        pass
    
def test():
    x = torch.randn(3, 3, 256, 256)
    model = CARN()
    print(model(x).shape)

if __name__ == "__main__":
    test()