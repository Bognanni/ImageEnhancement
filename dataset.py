import os
import glob
from PIL import Image
import torch
import random
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import torchvision.transforms as T


class PairedLowLightDataset(Dataset):
    """
    Dataset class for paired low-light enhancement data. It loads low-light and corresponding high-light images,
    applies data augmentation during training, and returns them as tensors.
    """

    def __init__(self, root_dir, split='all', img_size=256):
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size

        self.low_dir = os.path.join(root_dir, 'low')
        self.high_dir = os.path.join(root_dir, 'high')

        # Images sorted by filename to ensure correct pairing between low-light and high-light images
        low_images = sorted(glob.glob(os.path.join(self.low_dir, '*.*')))
        high_images = sorted(glob.glob(os.path.join(self.high_dir, '*.*')))

        if len(low_images) == 0:
            print(f"Warning: No images found in {self.low_dir}")

        total_len = len(low_images)

        # Split the dataset into train, val, and test sets based on the specified split argument.
        # The splits are 80% for training, 10% for validation, and 10% for testing
        if split == 'train':
            self.low_images = low_images[:int(total_len * 0.8)]
            self.high_images = high_images[:int(total_len * 0.8)]
        elif split == 'val':
            self.low_images = low_images[int(total_len * 0.8):int(total_len * 0.9)]
            self.high_images = high_images[int(total_len * 0.8):int(total_len * 0.9)]
        elif split == 'test':
            self.low_images = low_images[int(total_len * 0.9):]
            self.high_images = high_images[int(total_len * 0.9):]
        else:
            self.low_images = low_images
            self.high_images = high_images

    def __len__(self):
        return len(self.low_images)

    def __getitem__(self, idx):
        """
        Returns the low-light and high-light image pair at the specified index as tensors. During training, 
        data augmentation is applied, including random cropping, horizontal and vertical flipping, and 
        color jittering. During validation and testing, the images are simply resized to the specified 
        image size. The low-light and high-light images are returned as tensors without clamping, allowing 
        for a wider range of values during training and evaluation.
        """
        low_img_path = self.low_images[idx]
        high_img_path = self.high_images[idx]

        low_img = Image.open(low_img_path).convert('RGB')
        high_img = Image.open(high_img_path).convert('RGB')

        if self.split == 'train':
            # Resize to a slightly larger size before random cropping to ensure we have enough pixels for the crop
            resize_dim = int(self.img_size * 1.12)
            low_img = low_img.resize((resize_dim, resize_dim), Image.BILINEAR)
            high_img = high_img.resize((resize_dim, resize_dim), Image.BILINEAR)

            # Crop the same region from both low and high images to maintain alignment
            i, j, h, w = T.RandomCrop.get_params(low_img, output_size=(self.img_size, self.img_size))
            low_img = TF.crop(low_img, i, j, h, w)
            high_img = TF.crop(high_img, i, j, h, w)

            # Flip the images horizontally and vertically with a 50% chance to increase data diversity during training
            if random.random() > 0.5:
                low_img = TF.hflip(low_img)
                high_img = TF.hflip(high_img)

            if random.random() > 0.5:
                low_img = TF.vflip(low_img)
                high_img = TF.vflip(high_img)
        else:
            low_img = low_img.resize((self.img_size, self.img_size), Image.BILINEAR)
            high_img = high_img.resize((self.img_size, self.img_size), Image.BILINEAR)

        low_tensor = TF.to_tensor(low_img)
        high_tensor = TF.to_tensor(high_img)

        if self.split == 'train':
            # Apply color jittering only to the low-light image to simulate different lighting conditions
            # and enhance the model's robustness during training
            jitter = T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0)
            low_tensor = jitter(low_tensor)

        return low_tensor, high_tensor


class UnpairedDataset(Dataset):
    """
    Dataset class for unpaired data. It loads all images from a specified directory and returns them as tensors.
    """

    def __init__(self, root_dir, img_size=256):
        self.root_dir = root_dir
        self.img_size = img_size

        self.images = []
        for ext in ['png', 'jpg', 'jpeg', 'PNG', 'JPG', 'JPEG']:
            self.images.extend(glob.glob(os.path.join(root_dir, '**', f'*.{ext}'), recursive=True))

        self.images = sorted(self.images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        Returns the image at the specified index as a tensor.
        """
        img_path = self.images[idx]
        img = Image.open(img_path).convert('RGB')

        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        tensor = TF.to_tensor(img)
        return tensor



def get_dataloaders(data_dir='./data', batch_size=8, num_workers=4, img_size=256):
    """
    Utility function to create dataloaders for training, validation, and testing.
    It initializes the appropriate datasets based on the specified splits and returns dataloaders for each.
    """
    train_dataset = PairedLowLightDataset(os.path.join(data_dir, 'LOL_v2'), split='train', img_size=img_size)
    val_dataset = PairedLowLightDataset(os.path.join(data_dir, 'LOL_v2'), split='val', img_size=img_size)

    test_indomain_dataset = PairedLowLightDataset(os.path.join(data_dir, 'LOL_v2'), split='test', img_size=img_size)
    test_crossdomain_dataset = PairedLowLightDataset(os.path.join(data_dir, 'LOL_v1'), split='all', img_size=img_size)
    test_exdark = UnpairedDataset(os.path.join(data_dir, 'ExDark'), img_size=img_size)

    # pin_memory=True to speed up data transfer to the GPU, drop_last=True in the training dataloader to ensure that all batches are of the same size
    # The validation and test dataloaders do not use drop_last since we want to evaluate on all available data.
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    test_indomain_loader = DataLoader(
        test_indomain_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_crossdomain_loader = DataLoader(
        test_crossdomain_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_exdark_loader = DataLoader(
        test_exdark, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_indomain_loader, test_crossdomain_loader, test_exdark_loader


if __name__ == '__main__':
    # Sanity Check
    train_loader, val_loader, test_in, test_cross, exdark = get_dataloaders(batch_size=4)
    for low, high in train_loader:
        print(f"Train batch: Low shape {low.shape}, High shape {high.shape}")
        print(
            f"Val: Low [{low.min():.2f}, {low.max():.2f}], High [{high.min():.2f}, {high.max():.2f}]")
        break