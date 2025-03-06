import os
import math
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from PIL import Image


class SobelFilter(torch.nn.Module):
    def __init__(self):
        super(SobelFilter, self).__init__()
        # Sobel kernels as buffers
        self.register_buffer("sobel_x", torch.tensor(
            [[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=torch.float32
        ).unsqueeze(1))  # Shape: [1, 1, 3, 3]

        self.register_buffer("sobel_y", torch.tensor(
            [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype=torch.float32
        ).unsqueeze(1))  # Shape: [1, 1, 3, 3]

    def forward(self, img):
        sobel_x = self.sobel_x.to(dtype=img.dtype, device=img.device)
        sobel_y = self.sobel_y.to(dtype=img.dtype, device=img.device)
        
        grad_x = F.conv2d(img, sobel_x, padding=1)
        grad_y = F.conv2d(img, sobel_y, padding=1)
        return torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)  # epsilon for stability

sobel_filter = SobelFilter()

def edge_loss(pred, target):
    pred_edges = sobel_filter(pred)
    target_edges = sobel_filter(target)
    return F.mse_loss(pred_edges, target_edges)

def psnr(output, target):
    mse = F.mse_loss(output, target)
    psnr_value = 20 * math.log10(1.0) - 10 * math.log10(mse.item())  # assume inputs normalized to [0, 1]
    return psnr_value
class CombinedLoss(nn.Module):
    def __init__(self, reconstruction_loss_fn, edge_weight=0.1, psnr_weight=0.3):
        super(CombinedLoss, self).__init__()
        self.reconstruction_loss_fn = reconstruction_loss_fn
        self.edge_weight = edge_weight
        self.psnr_weight = psnr_weight

    def forward(self, pred, target):
        reconstruction_loss = self.reconstruction_loss_fn(pred, target)
        edge_loss_value = edge_loss(pred, target)
        # psnr_value = psnr(pred, target)
        return reconstruction_loss + (self.edge_weight * edge_loss_value) #- (self.psnr_weight * psnr_value)


def train_model_single_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    scaler,  # Kept for compatibility but no longer used
    grad_clip,
):
    model.train()
    total_loss = 0
    total_psnr = 0
    num_batches = len(train_loader)

    progress_bar = tqdm(train_loader, desc="Training", leave=True)
    for batch in progress_bar:
        input_images, target_images = batch
        input_images = input_images.to(device)
        target_images = target_images.to(device)

        # Forward pass without autocast
        output_images = model(input_images)
        loss = criterion(output_images, target_images)

        # Backward pass and optimization without scaler
        optimizer.zero_grad()
        loss.backward()

        if grad_clip:
            # Gradient clipping without unscaling
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            
        optimizer.step()

        total_loss += loss.item()
        current_psnr = psnr(output_images, target_images)
        total_psnr += current_psnr

        progress_bar.set_postfix({"loss": loss.item(), 
                                  "psnr": current_psnr})

    avg_train_loss = total_loss / num_batches
    avg_train_psnr = total_psnr / num_batches
    return avg_train_loss, avg_train_psnr


def validate_model_single_epoch(
    model,
    val_loader,
    criterion,
    device,
):
    model.eval()
    total_val_loss = 0
    total_psnr = 0

    with torch.no_grad():
        for batch in val_loader:
            input_images, target_images = batch
            input_images = input_images.to(device)
            target_images = target_images.to(device)

            output_images = model(input_images)
            loss = criterion(output_images, target_images)

            total_val_loss += loss.item()
            total_psnr += psnr(output_images, target_images)

    avg_val_loss = total_val_loss / len(val_loader)
    avg_psnr = total_psnr / len(val_loader)
    return avg_val_loss, avg_psnr


def save_checkpoint(epoch, model, optimizer, history, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"ckpt_{epoch}")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
    }, checkpoint_path)
    print(f"Model checkpoint saved at {checkpoint_path}")


def save_samples(epoch, 
                 model, 
                 val_loader, 
                 device, 
                 output_dir, 
                 sample_loader=None, 
                 num_samples=5):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    samples = []

    # use fixed sample loader if provided
    loader = sample_loader if sample_loader is not None else val_loader

    with torch.no_grad():
        for _, (input_images, target_images) in enumerate(loader):
            input_images = input_images.to(device)
            denoised_images = model(input_images)

            # input, denoised, and target images
            for i in range(min(len(input_images), num_samples - len(samples))):
                samples.append((input_images[i].cpu(), denoised_images[i].cpu(), target_images[i].cpu()))

            if len(samples) >= num_samples:
                break

    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    for i, (input_img, denoised_img, target_img) in enumerate(samples):
        axes[i, 0].imshow(input_img.permute(1, 2, 0).numpy(), cmap="gray")
        axes[i, 0].set_title("Input (Downsampled)")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(denoised_img.permute(1, 2, 0).numpy(), cmap="gray")
        axes[i, 1].set_title("Model Restored")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(target_img.permute(1, 2, 0).numpy(), cmap="gray")
        axes[i, 2].set_title("Target (Ground Truth)")
        axes[i, 2].axis("off")

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"epoch_{epoch}_samples.png")
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved {num_samples} samples at epoch {epoch} to {save_path}")


def get_device():
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    return device


def preprocess_image(image, device):
    image = torch.tensor(np.array(image, dtype=np.float32))
    if image.dim() == 2:
        image = image.unsqueeze(0) 
    else:
        image = image.permute(2, 0, 1) 

    image = image / 255.0

    input_tensor = image.unsqueeze(0).to(device)
    return input_tensor


def save_output_image(output_tensor, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img_np = output_tensor.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * 255.0).clip(0, 255).astype(np.uint8)
    if img_np.shape[-1] == 1:
        img_np = np.squeeze(img_np, axis=-1)
    pil_img = Image.fromarray(img_np)
    pil_img.save(save_path, format='PNG')


def restore_full_page(page_num, page_dir, model, device):
    input_image_path = f"{page_dir}/downsampled_4x/Page_{page_num}.png"
    output_image_path = f"{page_dir}/restoration_4x/Page_{page_num}.png"
    image = Image.open(input_image_path).convert('L')
    
    input_tensor = preprocess_image(image, device)
    with torch.no_grad():
        output_tensor = model(input_tensor).squeeze(0)
    save_output_image(output_tensor, output_image_path)
