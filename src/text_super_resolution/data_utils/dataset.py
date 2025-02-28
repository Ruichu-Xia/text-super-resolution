import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class LocalImageDataset(Dataset):
    def __init__(self, input_dir, target_dir, indices, transform=None):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.indices = indices 
        self.transform = transform

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        image_name = f"Image_{self.indices[idx]}.png"
        input_path = os.path.join(self.input_dir, image_name)
        target_path = os.path.join(self.target_dir, image_name)

        input_image = self._load_image(input_path)
        target_image = self._load_image(target_path)

        input_image = torch.tensor(np.array(input_image, dtype=np.float32))
        target_image = torch.tensor(np.array(target_image, dtype=np.float32))

        if input_image.dim() == 2:
            # Convert single channel (H,W) to (1,H,W)
            input_image = input_image.unsqueeze(0)
            target_image = target_image.unsqueeze(0)
            
            # # Repeat the single channel 3 times to get (3,H,W)
            # input_image = input_image.repeat(3, 1, 1)
            # target_image = target_image.repeat(3, 1, 1)
        else:
            # Handle the case where images already have multiple channels
            input_image = input_image.permute(2, 0, 1)
            target_image = target_image.permute(2, 0, 1)
            
            # Ensure 3 channels if needed
            if input_image.size(0) == 1:
                input_image = input_image.repeat(3, 1, 1)
            if target_image.size(0) == 1:
                target_image = target_image.repeat(3, 1, 1)
        
        # Normalize to [0, 1]
        input_image = input_image / 255.0 
        target_image = target_image / 255.0

        return input_image, target_image
    
    def _load_image(self, image_path):
        # Changed to keep the image in grayscale mode for processing
        # We'll convert to 3 channels in __getitem__
        image = Image.open(image_path).convert('L')
        return image

def get_split_indices(num_images, test_indices, val_ratio=0.2, seed=42): 
    all_indices = np.arange(0, num_images)
    train_val_indices = np.setdiff1d(all_indices, test_indices)
    train_indices, val_indices = train_test_split(
        train_val_indices, 
        test_size=val_ratio, 
        random_state=seed
    )
    return train_indices, val_indices