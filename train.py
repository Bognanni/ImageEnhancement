import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from piq import ssim, psnr, brisque, SSIMLoss

from dataset import get_dataloaders
from model import AttentionUNet, CompactUNet

def train_one_epoch(model, dataloader, optimizer, scaler, l1_criterion, ssim_criterion, device, alpha=1.0, beta=1.0):
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(dataloader, desc='Training')
    for low, high in pbar:
        low, high = low.to(device), high.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision
        with autocast('cuda'):
            output = model(low)
            l1_loss = l1_criterion(output, high)
            # SSIM loss requires float32
            ssim_loss_val = ssim_criterion(output.float(), high.float())
            
            loss = alpha * l1_loss + beta * ssim_loss_val
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
        
    return running_loss / len(dataloader)

@torch.no_grad()
def validate(model, dataloader, device):
    model.eval()
    val_psnr = 0.0
    val_ssim = 0.0
    
    pbar = tqdm(dataloader, desc='Validation (LOL-v1)')
    for low, high in pbar:
        low, high = low.to(device), high.to(device)
        
        with autocast('cuda'):
            output = model(low)
        
        # Calculate metrics (cast back to float32 for piq)
        output = output.float()
        high = high.float()
        
        val_psnr += psnr(output, high).item()
        val_ssim += ssim(output, high).item()
        
    return val_psnr / len(dataloader), val_ssim / len(dataloader)

def main():
    # Hyperparameters
    batch_size = 8
    epochs = 80
    lr = 1e-4
    patience = 15
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Dataloaders (We only need train and val for training loop)
    train_loader, val_loader, _, _, _ = get_dataloaders(batch_size=batch_size)
    
    # Model
    model = AttentionUNet().to(device)
    # To run baseline, you would use:
    # model = CompactUNet().to(device)
    
    # Optimizer and Loss
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = GradScaler('cuda')
    l1_criterion = nn.L1Loss()
    ssim_criterion = SSIMLoss() # PIQ's SSIM loss computes 1 - SSIM
    
    best_psnr = 0.0
    patience_counter = 0
    
    os.makedirs('checkpoints', exist_ok=True)
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, l1_criterion, ssim_criterion, device)
        print(f"Train Loss: {train_loss:.4f} (LR: {scheduler.get_last_lr()[0]:.6f})")
        
        # Step learning rate scheduler
        scheduler.step()
        
        val_psnr, val_ssim = validate(model, val_loader, device)
        print(f"Validation (In-Domain) PSNR: {val_psnr:.4f}, SSIM: {val_ssim:.4f}")
        
        # Early Stopping check
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            patience_counter = 0
            torch.save(model.state_dict(), 'checkpoints/best_model.pth')
            print(f"New best model saved with PSNR: {best_psnr:.4f}")
        else:
            patience_counter += 1
            print(f"Early stopping patience: {patience_counter}/{patience}")
            
        if patience_counter >= patience:
            print("Early stopping triggered!")
            break
            
    print("\nTraining completato! Per eseguire il test usa: python evaluation.py")

if __name__ == '__main__':
    main()
