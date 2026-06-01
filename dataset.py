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
    Dataset per immagini appaiate Low-Light / Normal-Light (es. LOL-v1, LOL-v2).
    """

    def __init__(self, root_dir, split='all', img_size=256):
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size

        self.low_dir = os.path.join(root_dir, 'low')
        self.high_dir = os.path.join(root_dir, 'high')

        low_images = sorted(glob.glob(os.path.join(self.low_dir, '*.*')))
        high_images = sorted(glob.glob(os.path.join(self.high_dir, '*.*')))

        if len(low_images) == 0:
            print(f"Warning: Nessuna immagine trovata in {self.low_dir}")

        total_len = len(low_images)

        # Split deterministico (come richiesto dalle specifiche per evitare data leak)
        if split == 'train':
            self.low_images = low_images[:int(total_len * 0.8)]
            self.high_images = high_images[:int(total_len * 0.8)]
        elif split == 'val':
            self.low_images = low_images[int(total_len * 0.8):int(total_len * 0.9)]
            self.high_images = high_images[int(total_len * 0.8):int(total_len * 0.9)]
        elif split == 'test':
            self.low_images = low_images[int(total_len * 0.9):]
            self.high_images = high_images[int(total_len * 0.9):]
        else:  # 'all' (usato per i dataset cross-domain come LOL-v1)
            self.low_images = low_images
            self.high_images = high_images

    def __len__(self):
        return len(self.low_images)

    def __getitem__(self, idx):
        low_img_path = self.low_images[idx]
        high_img_path = self.high_images[idx]

        low_img = Image.open(low_img_path).convert('RGB')
        high_img = Image.open(high_img_path).convert('RGB')

        # Sincronizzazione della Data Augmentation
        if self.split == 'train':
            # 1. Resize leggermente più grande per permettere il crop
            resize_dim = int(self.img_size * 1.12)  # ~286 per un target di 256
            low_img = low_img.resize((resize_dim, resize_dim), Image.BILINEAR)
            high_img = high_img.resize((resize_dim, resize_dim), Image.BILINEAR)

            # 2. Random Crop sincronizzato
            i, j, h, w = T.RandomCrop.get_params(low_img, output_size=(self.img_size, self.img_size))
            low_img = TF.crop(low_img, i, j, h, w)
            high_img = TF.crop(high_img, i, j, h, w)

            # 3. Random Horizontal Flip
            if random.random() > 0.5:
                low_img = TF.hflip(low_img)
                high_img = TF.hflip(high_img)

            # 4. Random Vertical Flip (aggiunto per maggiore robustezza spaziale)
            if random.random() > 0.5:
                low_img = TF.vflip(low_img)
                high_img = TF.vflip(high_img)
        else:
            # Per Validation e Test, resize esatto a 256x256 per valutazione consistente
            low_img = low_img.resize((self.img_size, self.img_size), Image.BILINEAR)
            high_img = high_img.resize((self.img_size, self.img_size), Image.BILINEAR)

        # To Tensor (scala i valori in [0, 1])
        low_tensor = TF.to_tensor(low_img)
        high_tensor = TF.to_tensor(high_img)

        if self.split == 'train':
            # Color Jitter lieve, applicato SOLO all'input.
            # NOTA: Rimosso 'hue' per evitare di introdurre color cast artificiali!
            jitter = T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0)
            low_tensor = jitter(low_tensor)

        return low_tensor, high_tensor


class UnpairedDataset(Dataset):
    """
    Dataset per immagini No-Reference / Unpaired (es. ExDark).
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
        img_path = self.images[idx]
        img = Image.open(img_path).convert('RGB')

        # Resize standard per evaluation
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        tensor = TF.to_tensor(img)
        return tensor


def create_dummy_dataset(base_path, paired=True, num_samples=10):
    """Genera un dataset fittizio per verificare che la pipeline non crashi se mancano i dati reali"""
    print(f"Creazione dummy dataset in {base_path}...")
    os.makedirs(base_path, exist_ok=True)

    if paired:
        os.makedirs(os.path.join(base_path, 'low'), exist_ok=True)
        os.makedirs(os.path.join(base_path, 'high'), exist_ok=True)
        for i in range(num_samples):
            low = torch.rand(3, 256, 256) * 0.3
            T.ToPILImage()(low).save(os.path.join(base_path, 'low', f'dummy_{i}.png'))
            high = torch.rand(3, 256, 256) * 0.8 + 0.2
            T.ToPILImage()(high).save(os.path.join(base_path, 'high', f'dummy_{i}.png'))
    else:
        # Struttura per ExDark (spesso divisa in sottocartelle per classe)
        os.makedirs(os.path.join(base_path, 'images'), exist_ok=True)
        for i in range(num_samples):
            img = torch.rand(3, 256, 256) * 0.3
            T.ToPILImage()(img).save(os.path.join(base_path, 'images', f'dummy_{i}.png'))


def setup_datasets(data_dir='./data'):
    """Verifica l'esistenza dei dataset e crea i dummy se non esistono"""
    os.makedirs(data_dir, exist_ok=True)

    lol_v2_path = os.path.join(data_dir, 'LOL_v2')
    lol_v1_path = os.path.join(data_dir, 'LOL_v1')
    exdark_path = os.path.join(data_dir, 'ExDark')

    if not os.path.exists(os.path.join(lol_v2_path, 'low')):
        create_dummy_dataset(lol_v2_path, paired=True, num_samples=32)

    if not os.path.exists(os.path.join(lol_v1_path, 'low')):
        create_dummy_dataset(lol_v1_path, paired=True, num_samples=16)

    exdark_images = glob.glob(os.path.join(exdark_path, '**', '*.*'), recursive=True)
    if len(exdark_images) == 0:
        create_dummy_dataset(exdark_path, paired=False, num_samples=16)


def get_dataloaders(data_dir='./data', batch_size=8, num_workers=4, img_size=256):
    """
    Inizializza i dataloader.
    Ottimizzato con pin_memory=True e drop_last=True per massimizzare
    le performance e la stabilità sulla RTX 2000 Ada.
    """
    setup_datasets(data_dir)

    # Train e Validation in-domain su LOL-v2
    train_dataset = PairedLowLightDataset(os.path.join(data_dir, 'LOL_v2'), split='train', img_size=img_size)
    val_dataset = PairedLowLightDataset(os.path.join(data_dir, 'LOL_v2'), split='val', img_size=img_size)

    # Test set (In-domain, Cross-domain Paired, Cross-domain Unpaired)
    test_indomain_dataset = PairedLowLightDataset(os.path.join(data_dir, 'LOL_v2'), split='test', img_size=img_size)
    test_crossdomain_dataset = PairedLowLightDataset(os.path.join(data_dir, 'LOL_v1'), split='all', img_size=img_size)
    test_exdark = UnpairedDataset(os.path.join(data_dir, 'ExDark'), img_size=img_size)

    # Dataloaders - Pin_memory=True accelera il trasferimento PCIe
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
        print(f"Verifica Train batch: Low shape {low.shape}, High shape {high.shape}")
        print(
            f"Valori minimi/massimi: Low [{low.min():.2f}, {low.max():.2f}], High [{high.min():.2f}, {high.max():.2f}]")
        break