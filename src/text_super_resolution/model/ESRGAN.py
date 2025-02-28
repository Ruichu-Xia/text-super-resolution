import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math
from tqdm import tqdm


class DenseResidualBlock(nn.Module):
    def __init__(self, in_channels, growth_channels=32, bias=True):
        super(DenseResidualBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, growth_channels, kernel_size=3, stride=1, padding=1, bias=bias
        )

    def forward(self, x):
        out = F.leaky_relu(self.conv(x), 0.2, inplace=True)
        out = torch.cat([x, out], 1)
        return out


class RRDB(nn.Module):
    """
    Residual-in-Residual Dense Block (RRDB).
    Each RRDB typically has multiple DenseResidualBlocks with a final residual scaling.
    """

    def __init__(
        self, in_channels, growth_channels=32, num_dense_layers=3, residual_scale=0.2
    ):
        super(RRDB, self).__init__()
        self.residual_scale = residual_scale

        blocks = []
        current_channels = in_channels
        for _ in range(num_dense_layers):
            blocks.append(
                DenseResidualBlock(current_channels, growth_channels=growth_channels)
            )
            current_channels += growth_channels

        self.body = nn.Sequential(*blocks)
        self.conv_final = nn.Conv2d(
            current_channels, in_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        residual = x
        out = self.body(x)
        out = self.conv_final(out)
        return residual + out * self.residual_scale


class RRDBNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        num_features=64,
        num_rrdb=23,
        growth_channels=32,
        scale_factor=2,
    ):
        super(RRDBNet, self).__init__()
        self.conv_first = nn.Conv2d(
            in_channels, num_features, kernel_size=3, stride=1, padding=1
        )

        rrdb_blocks = []
        for _ in range(num_rrdb):
            rrdb_blocks.append(RRDB(num_features, growth_channels=growth_channels))

        self.rrdb_blocks = nn.Sequential(*rrdb_blocks)

        self.conv_after_rrdb = nn.Conv2d(
            num_features, num_features, kernel_size=3, stride=1, padding=1
        )

        self.upscale = nn.Sequential()

        num_upsamples = 0
        while scale_factor > 1:
            self.upscale.add_module(
                f"upconv_{num_upsamples}",
                nn.Conv2d(num_features, num_features * 4, 3, 1, 1),
            )
            self.upscale.add_module(f"pixshuffle_{num_upsamples}", nn.PixelShuffle(2))
            self.upscale.add_module(
                f"lr_{num_upsamples}", nn.LeakyReLU(0.2, inplace=True)
            )
            scale_factor //= 2
            num_upsamples += 1

        self.conv_last = nn.Conv2d(
            num_features, out_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        print('input started')
        fea = self.conv_first(x)
        print(2)
        fea_input = fea
        fea = self.rrdb_blocks(fea)
        print(3)
        fea = self.conv_after_rrdb(fea)
        print(4)
        fea = fea + fea_input


        fea = self.upscale(fea)
        print(5)
        out = self.conv_last(fea)

        return out


class Discriminator_VGG(nn.Module):
    def __init__(self, in_channels=1, base_channels=64):
        super(Discriminator_VGG, self).__init__()

        def conv_block(in_f, out_f, normalize=True):
            layers = [nn.Conv2d(in_f, out_f, 3, 1, 1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_f))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_f = in_channels

        for i, out_f in enumerate(
            [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        ):
            layers.extend(conv_block(in_f, out_f, normalize=(i > 0)))
            layers.extend(conv_block(out_f, out_f, normalize=True))
            layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
            in_f = out_f

        layers.extend(conv_block(in_f, base_channels * 8, normalize=True))
        layers.extend(conv_block(base_channels * 8, base_channels * 8, normalize=True))

        self.features = nn.Sequential(*layers)

        self.classifier = nn.Conv2d(
            base_channels * 8, 1, kernel_size=1, stride=1, padding=0
        )

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

class MobileNetFeatureExtractor(nn.Module):
    def __init__(self, depth=5):
        super(MobileNetFeatureExtractor, self).__init__()
        # Get a pretrained MobileNetV2
        mobilenet = models.mobilenet_v2(pretrained=True)
        
        # Modify the first layer to accept 1 channel
        original_weight = mobilenet.features[0][0].weight.data
        new_conv = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        with torch.no_grad():
            new_weight = original_weight.sum(dim=1, keepdim=True) / 3.0
            new_conv.weight.data = new_weight
        
        # Replace the first layer
        mobilenet.features[0][0] = new_conv
        
        # Extract only the needed depth
        self.features = nn.Sequential()
        for i in range(min(depth, len(mobilenet.features))):
            self.features.add_module(str(i), mobilenet.features[i])
            
        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        return self.features(x)


class EfficientNetLiteDiscriminator(nn.Module):
    def __init__(self):
        super(EfficientNetLiteDiscriminator, self).__init__()
        
        # Load pretrained EfficientNet-Lite0
        self.efficientnet = models.efficientnet_b0(pretrained=True)
        
        # Modify first conv layer to accept grayscale input
        original_conv = self.efficientnet.features[0][0]
        self.efficientnet.features[0][0] = nn.Conv2d(
            1, 32,
            kernel_size=3, 
            stride=2,
            padding=1,
            bias=False
        )
        
        # Average the RGB weights to create grayscale weights
        with torch.no_grad():
            rgb_weight = original_conv.weight
            self.efficientnet.features[0][0].weight.copy_(
                rgb_weight.sum(dim=1, keepdim=True) / 3.0
            )
            
        # Replace classifier with binary classification head
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(1280, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.efficientnet(x)



def perceptual_loss(gen_img, real_img, vgg_extractor):
    gen_features = vgg_extractor(gen_img)
    real_features = vgg_extractor(real_img)
    return F.l1_loss(gen_features, real_features)


def train_esrgan(
    generator,
    discriminator,
    vgg_extractor,
    training_dataloader,
    validation_dataloader,
    g_optimizer,
    d_optimizer,
    num_epochs=100,
    device="cuda",
):
    """
    A skeleton for training ESRGAN. In practice, you may:
      - Pretrain the generator on L1 pixel loss
      - Then finetune with adversarial + perceptual losses
    """
    generator.train()
    discriminator.train()

    history = {
        "psnr": [],
        "g_pix_loss": [],
        "g_perc_loss": [],
        "g_adv_loss": [],
        "g_loss": [],
        "d_loss": [],
        "val_g_pix_loss": [],
        "val_g_perc_loss": [],
        "val_g_adv_loss": [],
        "val_g_loss": [],
        "val_d_loss": [],
    }

    # Create directory for saved models if it doesn't exist
    os.makedirs("saved_models", exist_ok=True)

    for epoch in range(num_epochs):
        epoch_g_pix_loss = 0.0
        # epoch_g_perc_loss = 0.0
        epoch_g_adv_loss = 0.0
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(training_dataloader, desc="Training", leave=True)

        for lr_imgs, hr_imgs in progress_bar:
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            num_batches += 1

            # === 1. Train Discriminator ===
            d_optimizer.zero_grad()
            # Generate super-resolution images, detach to avoid updating generator
            sr_imgs = generator(lr_imgs).detach()
            # Get discriminator predictions for real and fake images
            real_pred = discriminator(hr_imgs)
            fake_pred = discriminator(sr_imgs)
            # Calculate discriminator loss
            d_loss = ra_discriminator_loss(real_pred, fake_pred)
            d_loss.backward()
            d_optimizer.step()
            epoch_d_loss += d_loss.item()

            # === 2. Train Generator ===
            g_optimizer.zero_grad()
            
            # Generate super-resolution images again, this time for updating generator
            sr_imgs = generator(lr_imgs)
            
            # Get discriminator predictions
            real_pred = discriminator(hr_imgs)
            fake_pred = discriminator(sr_imgs)
            
            # Calculate generator losses
            g_adv_loss = ra_generator_loss(real_pred, fake_pred)
            # g_perc_loss = perceptual_loss(sr_imgs, hr_imgs, vgg_extractor)
            g_pix_loss = F.l1_loss(sr_imgs, hr_imgs)

            
            # Weighted sum of losses
            g_loss = 0.001 * g_adv_loss + 1.0 * g_pix_loss  # + 1.0 * g_perc_loss when using perceptual loss
            
            g_loss.backward()
            g_optimizer.step()
            
            # Accumulate losses for epoch average
            epoch_g_pix_loss += g_pix_loss.item()
            # epoch_g_perc_loss += g_perc_loss.item()
            epoch_g_adv_loss += g_adv_loss.item()
            epoch_g_loss += g_loss.item()

            # Display current losses and learning rates after tqdm bar
            current_g_lr = g_optimizer.param_groups[0]['lr']
            current_d_lr = d_optimizer.param_groups[0]['lr']
            print(f"\rG_loss: {g_loss.item():.4f} | D_loss: {d_loss.item():.4f} | "
                  f"G_lr: {current_g_lr:.6f} | D_lr: {current_d_lr:.6f}", end="")

        # Calculate epoch averages
        epoch_g_pix_loss /= num_batches
        # epoch_g_perc_loss /= num_batches
        epoch_g_adv_loss /= num_batches
        epoch_g_loss /= num_batches
        epoch_d_loss /= num_batches
        
        # Log training losses
        history["g_pix_loss"].append(epoch_g_pix_loss)
        # history["g_perc_loss"].append(epoch_g_perc_loss)
        history["g_adv_loss"].append(epoch_g_adv_loss)
        history["g_loss"].append(epoch_g_loss)
        history["d_loss"].append(epoch_d_loss)

        # Validation step
        generator.eval()
        discriminator.eval()
        
        val_g_pix_loss = 0.0
        # val_g_perc_loss = 0.0
        val_g_adv_loss = 0.0
        val_g_loss = 0.0
        val_d_loss = 0.0
        val_psnr = 0.0
        
        with torch.no_grad():
            for lr_imgs, hr_imgs in tqdm(validation_dataloader):
                lr_imgs = lr_imgs.to(device)
                hr_imgs = hr_imgs.to(device)
                
                sr_imgs = generator(lr_imgs)
                
                # Calculate validation losses
                real_pred = discriminator(hr_imgs)
                fake_pred = discriminator(sr_imgs)
                
                batch_g_adv_loss = ra_generator_loss(real_pred, fake_pred).item()
                # batch_g_perc_loss = perceptual_loss(sr_imgs, hr_imgs, vgg_extractor).item()
                batch_g_pix_loss = F.l1_loss(sr_imgs, hr_imgs).item()
                batch_d_loss = ra_discriminator_loss(real_pred, fake_pred).item()
                
                # Calculate PSNR
                mse = F.mse_loss(sr_imgs, hr_imgs).item()
                batch_psnr = 20 * math.log10(1.0) - 10 * math.log10(mse)
                
                val_g_pix_loss += batch_g_pix_loss
                # val_g_perc_loss += batch_g_perc_loss
                val_g_adv_loss += batch_g_adv_loss
                val_d_loss += batch_d_loss
                val_psnr += batch_psnr
            
            # Calculate validation averages
            val_g_pix_loss /= len(validation_dataloader)
            # val_g_perc_loss /= len(validation_dataloader)
            val_g_adv_loss /= len(validation_dataloader)
            val_g_loss = 0.001 * val_g_adv_loss + 1.0 * val_g_pix_loss # + 1.0 * val_g_perc_loss
            val_d_loss /= len(validation_dataloader)
            val_psnr /= len(validation_dataloader)
            
            # Log validation results
            history["val_g_pix_loss"].append(val_g_pix_loss)
            # history["val_g_perc_loss"].append(val_g_perc_loss)
            history["val_g_adv_loss"].append(val_g_adv_loss)
            history["val_g_loss"].append(val_g_loss)
            history["val_d_loss"].append(val_d_loss)
            history["psnr"].append(val_psnr)
            
        # Save the model checkpoints
        torch.save(generator.state_dict(), f"saved_models/generator_epoch_{epoch+1}.pth")
        torch.save(discriminator.state_dict(), f"saved_models/discriminator_epoch_{epoch+1}.pth")

        # Set models back to training mode
        generator.train()
        discriminator.train()

        print(
            f"Epoch [{epoch+1}/{num_epochs}] | "
            f"Train [D: {epoch_d_loss:.4f}, G: {epoch_g_loss:.4f}] | "
            f"Val [D: {val_d_loss:.4f}, G: {val_g_loss:.4f}, PSNR: {val_psnr:.2f}]"
        )

    return history