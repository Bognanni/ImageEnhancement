import os
import random
import numpy as np
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from piq import psnr, ssim, SSIMLoss
import torch.backends.cudnn as cudnn

from dataset import get_dataloaders
from model import AttentionUNet, CompactUNet


def parse_args():
    parser = argparse.ArgumentParser(description="Train Low-Light Enhancement Model")
    parser.add_argument('--model', type=str, default='compact', choices=['compact', 'attention'],
                        help="Scegli quale architettura addestrare")
    parser.add_argument('--batch_size', type=int, default=4,
                        help="Batch size fisico in VRAM (consigliato 4 per RTX 2000 Ada a 256x256)")
    parser.add_argument('--accum_steps', type=int, default=1,
                        help="Step di Gradient Accumulation (batch virtuale = batch_size * accum_steps)")
    parser.add_argument('--epochs', type=int, default=100, help="Numero massimo di epoche")
    parser.add_argument('--lr', type=float, default=3e-4, help="Learning rate iniziale")
    return parser.parse_args()


def set_seed(seed=42):
    """Imposta il seed per la riproducibilità deterministica"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train_one_epoch(model, dataloader, optimizer, l1_criterion, ssim_criterion, device, accum_steps, alpha=1.0,
                    beta=0.1):
    model.train()
    running_loss = 0.0

    pbar = tqdm(dataloader, desc='Training')
    optimizer.zero_grad()  # Resetta i gradienti all'inizio dell'epoca

    for i, (low, high) in enumerate(pbar):
        low, high = low.to(device), high.to(device)

        # Esecuzione standard in Float32 (TF32)
        output = model(low)

        l1_loss = l1_criterion(output, high)
        ssim_loss_val = ssim_criterion(output.float(), high.float())

        # Combinazione delle loss. Dividiamo per accum_steps per
        # mantenere la magnitudo del gradiente costante.
        loss = (alpha * l1_loss + beta * ssim_loss_val) / accum_steps

        # Backward standard
        loss.backward()

        # Eseguiamo lo step di ottimizzazione solo ogni 'accum_steps' iterazioni
        # o se siamo all'ultimo batch del dataloader
        if ((i + 1) % accum_steps == 0) or ((i + 1) == len(dataloader)):
            # Gradient Clipping per stabilità estrema
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            optimizer.zero_grad()  # Resetta i gradienti per il prossimo ciclo di accumulo

        # Ricalcoliamo la loss in scala reale solo per i log visivi
        real_loss = loss.item() * accum_steps
        running_loss += real_loss
        pbar.set_postfix({'loss': f"{real_loss:.4f}"})

    return running_loss / len(dataloader)


@torch.no_grad()
def validate(model, dataloader, device):
    model.eval()
    val_psnr = 0.0
    val_ssim = 0.0

    pbar = tqdm(dataloader, desc='Validation (In-Domain LOL-v2)')
    for low, high in pbar:
        low, high = low.to(device), high.to(device)

        output = model(low)

        # Metriche full-reference calcolate in fp32
        output = output.float().clamp(0, 1)  # Assicura il range per il calcolo metriche
        high = high.float()

        val_psnr += psnr(output, high).item()
        val_ssim += ssim(output, high).item()

    return val_psnr / len(dataloader), val_ssim / len(dataloader)


def main():
    args = parse_args()
    
    # Imposta il seed per replicare i risultati
    set_seed(42)

    # Ottimizzazioni per GPU architettura Ada Lovelace
    cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(
        f"Device: {device} | Modello: {args.model.upper()} | Batch Reale: {args.batch_size} | Virtual Batch: {args.batch_size * args.accum_steps}")

    # Dataloaders - Usiamo 4 workers per non saturare la CPU di Runpod
    train_loader, val_loader, _, _, _ = get_dataloaders(batch_size=args.batch_size, num_workers=4)

    # Selezione del modello via argparse
    if args.model == 'attention':
        model = AttentionUNet().to(device)
    else:
        model = CompactUNet().to(device)

    # Optimizer e Scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Loss functions come da specifiche dell'assignment (L1 + SSIM)
    l1_criterion = nn.L1Loss()
    ssim_criterion = SSIMLoss()

    # Variabili per l'Early Stopping
    best_psnr = 0.0
    patience = 25
    patience_counter = 0
    save_path = f'checkpoints/best_model_{args.model}.pth'

    os.makedirs('checkpoints', exist_ok=True)

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_loss = train_one_epoch(
            model, train_loader, optimizer,
            l1_criterion, ssim_criterion, device, args.accum_steps
        )
        print(f"Train Loss: {train_loss:.4f} (LR: {scheduler.get_last_lr()[0]:.6e})")
        
        # Validazione in-domain
        val_psnr, val_ssim = validate(model, val_loader, device)
        print(f"Val PSNR: {val_psnr:.4f} | Val SSIM: {val_ssim:.4f}")
        scheduler.step()

        # Logica Early Stopping
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"-> Nuovo best model salvato! (PSNR: {best_psnr:.4f})")
        else:
            patience_counter += 1
            print(f"-> Nessun miglioramento. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("\n*** Early Stopping innescato! Termine dell'addestramento. ***")
            break

    print(f"\nAddestramento {args.model.upper()} completato!")
    print(f"Per testarlo, avvia: python evaluation.py --model {args.model}")


if __name__ == '__main__':
    main()
