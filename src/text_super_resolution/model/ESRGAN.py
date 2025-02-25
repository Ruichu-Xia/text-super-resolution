import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class DenseResidualBlock(nn.Module):
    def __init__(self, in_channels, growth_channels=32, bias=True):
        super(DenseResidualBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, growth_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        out = F.leaky_relu(self.conv(x), 0.2, inplace=True)
        out = torch.cat([x, out], 1)
        return out
    

class RRDB(nn.Module):
    """
    Residual-in-Residual Dense Block (RRDB).
    Each RRDB typically has multiple DenseResidualBlocks with a final residual scaling.
    """
    def __init__(self, in_channels, growth_channels=32, num_dense_layers=3, residual_scale=0.2):
        super(RRDB, self).__init__()
        self.residual_scale = residual_scale

        blocks = []
        current_channels = in_channels
        for _ in range(num_dense_layers):
            blocks.append(DenseResidualBlock(current_channels, growth_channels=growth_channels))
            current_channels += growth_channels

        self.body = nn.Sequential(*blocks)
        self.conv_final = nn.Conv2d(current_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        residual = x
        out = self.body(x)
        out = self.conv_final(out)
        return residual + out * self.residual_scale
    

class RRDBNet(nn.Module):
    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 num_features=64,
                 num_rrdb=23,
                 growth_channels=32,
                 scale_factor=2):
        super(RRDBNet, self).__init__()
        self.conv_first = nn.Conv2d(in_channels, num_features, kernel_size=3, stride=1, padding=1)

        rrdb_blocks = []
        for _ in range(num_rrdb):
            rrdb_blocks.append(RRDB(num_features, growth_channels=growth_channels))
        
        self.rrdb_blocks = nn.Sequential(*rrdb_blocks)

        self.conv_after_rrdb = nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1)

        self.upscale = nn.Sequential()

        num_upsamples = 0
        while scale_factor > 1:
            self.upscale.add_module(f'upconv_{num_upsamples}', nn.Conv2d(num_features, num_features * 4, 3, 1, 1))
            self.upscale.add_module(f'pixshuffle_{num_upsamples}', nn.PixelShuffle(2))
            self.upscale.add_module(f'lr_{num_upsamples}', nn.LeakyReLU(0.2, inplace=True))
            sf //= 2
            num_samples += 1

        self.conv_last = nn.Conv2d(num_features, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        fea = self.conv_first(x)
        fea_input = fea

        fea = self.rrdb_blocks(fea)
        fea = self.conv_after_rrdb(fea)
        fea = fea + fea_input

        fea = self.upscale(fea)
        out = self.conv_last(fea)

        return out
    

class Discriminator_VGG(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super(Discriminator_VGG, self).__init__()

        def conv_block(in_f, out_f, normalize=True):
            layers = [nn.Conv2d(in_f, out_f, 3, 1, 1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_f))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_f = in_channels

        for i, out_f in enumerate([base_channels, base_channels*2, base_channels*4, base_channels*8]):
            layers.extend(conv_block(in_f, out_f, normalize=(i>0)))
            layers.extend(conv_block(out_f, out_f, normalize=True))
            layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
            in_f = out_f
        
        layers.extend(conv_block(in_f, base_channels*8, normalize=True))
        layers.extend(conv_block(base_channels*8, base_channels*8, normalize=True))

        self.features = nn.Sequential(*layers)

        self.classifier = nn.Conv2d(base_channels*8, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        feats = self.features(x)
        out = self.classifier(feats)
        return out
    
# Relativistic Average GAN Loss
def ra_discriminator_loss(real_pred, fake_pred):
    """
    real_pred, fake_pred: Discriminator logits (N x 1 x H x W).
    D_Ra(real) = sigmoid(real_pred - mean(fake_pred))
    D_Ra(fake) = sigmoid(fake_pred - mean(real_pred))
    """
    real_mean = torch.mean(fake_pred)
    fake_mean = torch.mean(real_pred)

    D_real = torch.sigmoid(real_pred - real_mean)
    D_fake = torch.sigmoid(fake_pred - fake_mean)

    d_loss_real = F.binary_cross_entropy(D_real, torch.ones_like(D_real))
    d_loss_fake = F.binary_cross_entropy(D_fake, torch.zeros_like(D_fake))

    d_loss = (d_loss_real + d_loss_fake) * 0.5
    return d_loss

def ra_generator_loss(real_pred, fake_pred):
    """
    G wants to maximize log(D_Ra(fake)) + log(1 - D_Ra(real))
    """
    real_mean = torch.mean(fake_pred)
    fake_mean = torch.mean(real_pred)

    D_real = torch.sigmoid(real_pred - real_mean)
    D_fake = torch.sigmoid(fake_pred - fake_mean)

    g_loss_real = F.binary_cross_entropy(D_real, torch.zeros_like(D_real))
    g_loss_fake = F.binary_cross_entropy(D_fake, torch.ones_like(D_fake))

    g_loss = (g_loss_real + g_loss_fake) * 0.5
    
    return g_loss


class VGGFeatureExtractor(nn.Module):
    def __init__(self, layer_index=35):
        super(VGGFeatureExtractor, self).__init__()
        vgg19 = models.vgg19(pretrained=True)
        vgg_features = vgg19.features

        self.feature_extractor = nn.Sequential()
        for i in range(layer_index + 1):
            self.feature_extractor.add_module(str(i), vgg_features[i])
        
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        return self.feature_extractor(x)
    

def perceptual_loss(gen_img, real_img, vgg_extractor):
    gen_features = vgg_extractor(gen_img)
    real_features = vgg_extractor(real_img)
    return F.l1_loss(gen_features, real_features)


def train_esrgan(generator, discriminator, vgg_extractor, dataloader, 
                 g_optimizer, d_optimizer, num_epochs=100, device='cuda'):
    """
    A skeleton for training ESRGAN. In practice, you may:
      - Pretrain the generator on L1 pixel loss
      - Then finetune with adversarial + perceptual losses
    """
    generator.train()
    discriminator.train()
    
    for epoch in range(num_epochs):
        for lr_imgs, hr_imgs in dataloader:
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)

            # === 1. Train Discriminator ===
            d_optimizer.zero_grad()
            
            sr_imgs = generator(lr_imgs).detach()  # generate super-res, detach so G won't be updated here
            real_pred = discriminator(hr_imgs)
            fake_pred = discriminator(sr_imgs)
            
            d_loss = ra_discriminator_loss(real_pred, fake_pred)
            d_loss.backward()
            d_optimizer.step()

            # === 2. Train Generator ===
            g_optimizer.zero_grad()
            
            sr_imgs = generator(lr_imgs)  # forward again, this time for G update
            real_pred = discriminator(hr_imgs)
            fake_pred = discriminator(sr_imgs)
            
            # Adversarial loss
            g_adv_loss = ra_generator_loss(real_pred, fake_pred)
            # Perceptual loss
            g_perc_loss = perceptual_loss(sr_imgs, hr_imgs, vgg_extractor)
            # Pixel loss (optional)
            g_pix_loss = F.l1_loss(sr_imgs, hr_imgs)
            
            # Weighted sum of losses
            # You can tune these weights as needed
            g_loss = 0.001 * g_adv_loss + 1.0 * g_perc_loss + 1.0 * g_pix_loss
            
            g_loss.backward()
            g_optimizer.step()
        
        print(f"Epoch [{epoch+1}/{num_epochs}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")