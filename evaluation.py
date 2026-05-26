import os
import torch
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from piq import ssim, psnr, brisque

from dataset import get_dataloaders
from model import AttentionUNet

@torch.no_grad()
def evaluate_full_reference(model, dataloader, device):
    model.eval()
    val_psnr = 0.0
    val_ssim = 0.0
    
    pbar = tqdm(dataloader, desc='Full-Reference Eval (LOL-v1)')
    for low, high in pbar:
        low, high = low.to(device), high.to(device)
        output = model(low)
        
        output = output.float()
        high = high.float()
        
        val_psnr += psnr(output, high).item()
        val_ssim += ssim(output, high).item()
        
    return val_psnr / len(dataloader), val_ssim / len(dataloader)

@torch.no_grad()
def evaluate_no_reference(model, dataloader, device):
    model.eval()
    val_brisque = 0.0
    
    pbar = tqdm(dataloader, desc='No-Reference Eval (ExDark)')
    for low in pbar:
        low = low.to(device)
        output = model(low)
        
        output = output.float()
        try:
            val_brisque += brisque(output).item()
        except:
            pass
            
    return val_brisque / len(dataloader)

def visualize_random_samples(model, dataloader, device, num_samples=5, save_path="evaluation_visuals.png", title="Visual Evaluation"):
    """
    Seleziona casualmente num_samples immagini, le processa e le salva in una griglia.
    Usa dataset appaiati (es. LOL) per mostrare: Low-Light -> Enhanced -> Ground Truth
    """
    model.eval()
    
    # Raccogli tutte le immagini in una lista
    all_low, all_high = [], []
    for low, high in dataloader:
        all_low.append(low)
        all_high.append(high)
        
    all_low = torch.cat(all_low, dim=0)
    all_high = torch.cat(all_high, dim=0)
    
    # Seleziona indici casuali
    total_imgs = all_low.shape[0]
    num_samples = min(num_samples, total_imgs)
    indices = random.sample(range(total_imgs), num_samples)
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    fig.suptitle(title, fontsize=16)
    
    for i, idx in enumerate(indices):
        low_img = all_low[idx].unsqueeze(0).to(device)
        high_img = all_high[idx].to(device)
        
        with torch.no_grad():
            enhanced_img = model(low_img).squeeze(0)
            
        # Converti i tensori in formato (H, W, C) per Matplotlib
        # e portali sulla CPU
        low_np = low_img.squeeze(0).cpu().permute(1, 2, 0).numpy()
        enh_np = enhanced_img.cpu().permute(1, 2, 0).numpy()
        high_np = high_img.cpu().permute(1, 2, 0).numpy()
        
        # Plot Low Light
        ax = axes[i, 0] if num_samples > 1 else axes[0]
        ax.imshow(low_np.clip(0, 1))
        ax.set_title('Low Light Input')
        ax.axis('off')
        
        # Plot Enhanced
        ax = axes[i, 1] if num_samples > 1 else axes[1]
        ax.imshow(enh_np.clip(0, 1))
        ax.set_title('Enhanced Output (Model)')
        ax.axis('off')
        
        # Plot Ground Truth
        ax = axes[i, 2] if num_samples > 1 else axes[2]
        ax.imshow(high_np.clip(0, 1))
        ax.set_title('Ground Truth (Normal Light)')
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\nVisualizzazione completata! Immagini salvate in: {save_path}")
    # plt.show() # Decommentare per aprire una finestra interattiva
    plt.close()

def visualize_exdark_samples(model, dataloader, device, num_samples=5, save_path="evaluation_exdark_visuals.png"):
    """
    Seleziona casualmente num_samples immagini da ExDark, le processa e le salva in una griglia.
    Poiché ExDark non ha ground truth, mostrerà solo: Low-Light -> Enhanced
    """
    model.eval()
    
    all_low = []
    for low in dataloader:
        all_low.append(low)
        
    all_low = torch.cat(all_low, dim=0)
    
    total_imgs = all_low.shape[0]
    num_samples = min(num_samples, total_imgs)
    indices = random.sample(range(total_imgs), num_samples)
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(8, 4 * num_samples))
    fig.suptitle('Visual Evaluation (No-Reference: ExDark)', fontsize=16)
    
    for i, idx in enumerate(indices):
        low_img = all_low[idx].unsqueeze(0).to(device)
        
        with torch.no_grad():
            enhanced_img = model(low_img).squeeze(0)
            
        low_np = low_img.squeeze(0).cpu().permute(1, 2, 0).numpy()
        enh_np = enhanced_img.cpu().permute(1, 2, 0).numpy()
        
        ax1 = axes[i, 0] if num_samples > 1 else axes[0]
        ax1.imshow(low_np.clip(0, 1))
        ax1.set_title('Low Light Input')
        ax1.axis('off')
        
        ax2 = axes[i, 1] if num_samples > 1 else axes[1]
        ax2.imshow(enh_np.clip(0, 1))
        ax2.set_title('Enhanced Output (Model)')
        ax2.axis('off')
        
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Visualizzazione ExDark completata! Immagini salvate in: {save_path}")
    plt.close()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    checkpoint_path = 'checkpoints/best_model.pth'
    if not os.path.exists(checkpoint_path):
        print(f"Errore: Checkpoint '{checkpoint_path}' non trovato. Avvia prima il training!")
        return

    # Inizializza il modello e carica i pesi
    model = AttentionUNet().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print("Modello caricato con successo.")
    
    # Carica i dataloader per l'evaluation
    _, _, test_in_loader, test_cross_loader, exdark_loader = get_dataloaders(batch_size=8)
    
    print("\n--- Valutazione Quantitativa ---")
    psnr_in, ssim_in = evaluate_full_reference(model, test_in_loader, device)
    print(f"[In-Domain: LOL-v2] PSNR: {psnr_in:.4f} | SSIM: {ssim_in:.4f}")
    
    psnr_cross, ssim_cross = evaluate_full_reference(model, test_cross_loader, device)
    print(f"[Cross-Domain: LOL-v1] PSNR: {psnr_cross:.4f} | SSIM: {ssim_cross:.4f}")
    
    brisque_score = evaluate_no_reference(model, exdark_loader, device)
    print(f"[Cross-Domain No-Ref: ExDark] BRISQUE: {brisque_score:.4f}")
    
    print("\n--- Valutazione Qualitativa (Visuale) ---")
    # Usa un dataloader con batch=1 per visualizzare facilmente campioni individuali
    _, _, test_in_vis, test_cross_vis, exdark_vis = get_dataloaders(batch_size=1)
    
    visualize_random_samples(model, test_in_vis, device, num_samples=5, save_path="eval_indomain_LOLv2.png", title="In-Domain (LOL-v2)")
    visualize_random_samples(model, test_cross_vis, device, num_samples=5, save_path="eval_crossdomain_LOLv1.png", title="Cross-Domain (LOL-v1)")
    visualize_exdark_samples(model, exdark_vis, device, num_samples=5, save_path="eval_exdark.png")

if __name__ == '__main__':
    main()
