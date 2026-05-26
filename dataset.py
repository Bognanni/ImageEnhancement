import os
import glob
from PIL import Image
import torch
import random
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import urllib.request
import zipfile

class PairedLowLightDataset(Dataset):
    """
    Dataset for paired low-light and normal-light images (e.g., LOL-v1, LOL-v2).
    Expects directory structure:
    root_dir/
        low/
            img1.png
            ...
        high/
            img1.png
            ...
    """
    def __init__(self, root_dir, split='all'):
        self.root_dir = root_dir
        self.split = split
        
        self.low_dir = os.path.join(root_dir, 'low')
        self.high_dir = os.path.join(root_dir, 'high')
        
        low_images = sorted(glob.glob(os.path.join(self.low_dir, '*.*')))
        high_images = sorted(glob.glob(os.path.join(self.high_dir, '*.*')))
        
        if len(low_images) == 0:
            print(f"Warning: No images found in {self.low_dir}")
            
        total_len = len(low_images)
        
        # Deterministic split to allow in-domain validation and testing
        if split == 'train':
            self.low_images = low_images[:int(total_len * 0.8)]
            self.high_images = high_images[:int(total_len * 0.8)]
        elif split == 'val':
            self.low_images = low_images[int(total_len * 0.8):int(total_len * 0.9)]
            self.high_images = high_images[int(total_len * 0.8):int(total_len * 0.9)]
        elif split == 'test':
            self.low_images = low_images[int(total_len * 0.9):]
            self.high_images = high_images[int(total_len * 0.9):]
        else: # 'all' (usato per i dataset di solo test cross-domain come LOL-v1)
            self.low_images = low_images
            self.high_images = high_images

    def __len__(self):
        return len(self.low_images)

    def __getitem__(self, idx):
        low_img_path = self.low_images[idx]
        high_img_path = self.high_images[idx]
        
        low_img = Image.open(low_img_path).convert('RGB')
        high_img = Image.open(high_img_path).convert('RGB')
        
        # Base resize to 256x256
        low_img = low_img.resize((256, 256), Image.BILINEAR)
        high_img = high_img.resize((256, 256), Image.BILINEAR)
        
        # Synchronized Data Augmentation
        if self.split == 'train':
            # Random Horizontal Flip
            if random.random() > 0.5:
                low_img = TF.hflip(low_img)
                high_img = TF.hflip(high_img)
                
            # Random Crop (pad first, then crop)
            if random.random() > 0.5:
                low_img = TF.pad(low_img, 16)
                high_img = TF.pad(high_img, 16)
                i, j, h, w = transforms.RandomCrop.get_params(low_img, output_size=(256, 256))
                low_img = TF.crop(low_img, i, j, h, w)
                high_img = TF.crop(high_img, i, j, h, w)
                
        # To Tensor (scales to [0, 1])
        low_tensor = TF.to_tensor(low_img)
        high_tensor = TF.to_tensor(high_img)
        
        if self.split == 'train':
            # Color jitter applicato SOLO all'input per robustezza
            jitter = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05)
            low_tensor = jitter(low_tensor)
            
        return low_tensor, high_tensor

class UnpairedDataset(Dataset):
    """
    Dataset for unpaired low-light images (e.g., ExDark).
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        
        self.images = []
        for ext in ['png', 'jpg', 'jpeg']:
            self.images.extend(glob.glob(os.path.join(root_dir, '**', f'*.{ext}'), recursive=True))
        
        self.images = sorted(self.images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert('RGB')
        
        img = img.resize((256, 256), Image.BILINEAR)
        tensor = TF.to_tensor(img)
        return tensor

def create_dummy_dataset(base_path, paired=True, num_samples=10):
    """Create dummy data for testing the pipeline"""
    print(f"Creating dummy dataset at {base_path}")
    os.makedirs(base_path, exist_ok=True)
    
    if paired:
        os.makedirs(os.path.join(base_path, 'low'), exist_ok=True)
        os.makedirs(os.path.join(base_path, 'high'), exist_ok=True)
        for i in range(num_samples):
            low = torch.rand(3, 256, 256) * 0.3
            transforms.ToPILImage()(low).save(os.path.join(base_path, 'low', f'dummy_{i}.png'))
            high = torch.rand(3, 256, 256) * 0.8 + 0.2
            transforms.ToPILImage()(high).save(os.path.join(base_path, 'high', f'dummy_{i}.png'))
    else:
        os.makedirs(os.path.join(base_path, 'images'), exist_ok=True)
        for i in range(num_samples):
            img = torch.rand(3, 256, 256) * 0.3
            transforms.ToPILImage()(img).save(os.path.join(base_path, 'images', f'dummy_{i}.png'))

def setup_datasets(data_dir='./data'):
    os.makedirs(data_dir, exist_ok=True)
    
    lol_v2_path = os.path.join(data_dir, 'LOL_v2')
    lol_v1_path = os.path.join(data_dir, 'LOL_v1')
    exdark_path = os.path.join(data_dir, 'ExDark')
    
    if not os.path.exists(os.path.join(lol_v2_path, 'low')):
        print("LOL-v2 dataset not found. Generating dummy dataset for pipeline verification.")
        create_dummy_dataset(lol_v2_path, paired=True, num_samples=32)
        
    if not os.path.exists(os.path.join(lol_v1_path, 'low')):
        print("LOL-v1 dataset not found. Generating dummy dataset for pipeline verification.")
        create_dummy_dataset(lol_v1_path, paired=True, num_samples=16)
        
    exdark_images = glob.glob(os.path.join(exdark_path, '**', '*.*'), recursive=True)
    if len(exdark_images) == 0:
        print("ExDark dataset not found or empty. Generating dummy dataset for pipeline verification.")
        create_dummy_dataset(exdark_path, paired=False, num_samples=16)

def get_dataloaders(data_dir='./data', batch_size=8, num_workers=0):
    setup_datasets(data_dir)
    
    # Train e Validation in-domain su LOL-v2
    train_dataset = PairedLowLightDataset(os.path.join(data_dir, 'LOL_v2'), split='train')
    val_dataset = PairedLowLightDataset(os.path.join(data_dir, 'LOL_v2'), split='val')
    
    # Test set
    test_indomain_dataset = PairedLowLightDataset(os.path.join(data_dir, 'LOL_v2'), split='test')
    test_crossdomain_dataset = PairedLowLightDataset(os.path.join(data_dir, 'LOL_v1'), split='all')
    test_exdark = UnpairedDataset(os.path.join(data_dir, 'ExDark'))
    
    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    test_indomain_loader = DataLoader(test_indomain_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_crossdomain_loader = DataLoader(test_crossdomain_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_exdark_loader = DataLoader(test_exdark, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_indomain_loader, test_crossdomain_loader, test_exdark_loader

if __name__ == '__main__':
    train_loader, val_loader, test_in, test_cross, exdark = get_dataloaders()
    for low, high in train_loader:
        print(f"Train batch: Low shape {low.shape}, High shape {high.shape}")
        break
