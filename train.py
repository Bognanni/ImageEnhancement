import os
import random
import numpy as np
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast
from tqdm import tqdm
from piq import psnr, ssim, SSIMLoss
import torch.backends.cudnn as cudnn

from dataset import get_dataloaders
from model import AttentionUNet, CompactUNet


def parse_args():
    parser = argparse.ArgumentParser(description="Train Low-Light Enhancement Model")
    parser.add_argument('--model', type=str, default='compact', choices=['compact', 'attention'])
    parser.add_argument('--batch_size', type=int, default=4)

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=3e-4)
    return parser.parse_args()


def set_seed(seed=42):
    """
    Set the random seed for reproducibility across various libraries and ensure deterministic behavior in cuDNN.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train_one_epoch(model, dataloader, optimizer, l1_criterion, ssim_criterion, device, alpha=1.0, beta=0.1):
    """
    Train the model for one epoch. The loss is a combination of L1 loss and SSIM loss, weighted by alpha 
    and beta respectively. The training loop uses mixed precision with autocast for better performance 
    on NVIDIA GPUs. Gradient clipping is applied to prevent exploding gradients, and the progress is 
    displayed using tqdm with the current loss and learning rate in the postfix.
    """
    model.train()
    running_loss = 0.0

    pbar = tqdm(dataloader, desc='Training')

    for _, (low, high) in enumerate(pbar):
        low, high = low.to(device), high.to(device)
        optimizer.zero_grad()

        # Mixed precision training with autocast to speed up training on NVIDIA GPUs while maintaining numerical stability
        with autocast(device_type='cuda', dtype=torch.bfloat16):
            output = model(low)

        # Convert output for loss computation from bfloat16 to float32, which is necessary for accurate 
        # loss calculation, especially for the SSIM loss which can be sensitive to precision
        output = output.float()
        l1_loss = l1_criterion(output, high)
        ssim_loss_val = ssim_criterion(output, high)

        loss = alpha * l1_loss + beta * ssim_loss_val

        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    return running_loss / len(dataloader)


@torch.no_grad()
def validate(model, dataloader, device):
    """
    Validate the model on the validation set. The model is set to evaluation mode, and the PSNR and
    SSIM metrics are computed for each batch. The outputs are converted to float before computing the 
    metrics to ensure they are in the correct format. The average PSNR and SSIM values are returned 
    at the end of the validation loop.
    """
    model.eval()
    val_psnr = 0.0
    val_ssim = 0.0

    pbar = tqdm(dataloader, desc='Validation (In-Domain LOL-v2)')
    for low, high in pbar:
        low, high = low.to(device), high.to(device)

        with autocast(device_type='cuda', dtype=torch.bfloat16):
            output = model(low)

        output = output.float()

        val_psnr += psnr(output, high).item()
        val_ssim += ssim(output, high).item()

    return val_psnr / len(dataloader), val_ssim / len(dataloader)


def main():
    args = parse_args()
    set_seed(42)

    # to maximize performance on NVIDIA GPUs, we enable cuDNN benchmark and set the matmul precision 
    # to high for better performance with bfloat16 tensors. This allows cuDNN to find the best convolution 
    # algorithms for our specific hardware and model architecture, which can significantly speed up training.
    cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | Model: {args.model.upper()} | Batch Size: {args.batch_size}")

    train_loader, val_loader, _, _, _ = get_dataloaders(batch_size=args.batch_size, num_workers=4)

    if args.model == 'attention':
        model = AttentionUNet().to(device)
    else:
        model = CompactUNet().to(device)

    # Optimizer e Scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # The Cosine Annealing learning rate scheduler is used to adjust the learning rate during training. 
    # It starts with the initial learning rate and gradually decreases it following a cosine curve 
    # until it reaches a minimum value (eta_min) at the end of the training epochs.
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    l1_criterion = nn.L1Loss()
    ssim_criterion = SSIMLoss()

    best_psnr = 0.0
    patience = 15
    patience_counter = 0
    save_path = f'checkpoints/best_model_{args.model}.pth'

    os.makedirs('checkpoints', exist_ok=True)

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_loss = train_one_epoch(model, train_loader, optimizer, l1_criterion, ssim_criterion, device)
        print(f"Train Loss: {train_loss:.4f} (LR: {scheduler.get_last_lr()[0]:.6e})")
        
        val_psnr, val_ssim = validate(model, val_loader, device)
        print(f"Val PSNR: {val_psnr:.4f} | Val SSIM: {val_ssim:.4f}")
        scheduler.step()

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"-> New best model saved! (PSNR: {best_psnr:.4f})")
        else:
            patience_counter += 1
            print(f"-> No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("\nEarly Stopping! Ending the training.")
            break

    print(f"\nTraining {args.model.upper()} completed!")


if __name__ == '__main__':
    main()
