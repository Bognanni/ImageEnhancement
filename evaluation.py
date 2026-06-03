import os
import argparse
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from piq import ssim, psnr, brisque
import torch.backends.cudnn as cudnn

from dataset import PairedLowLightDataset, UnpairedDataset, setup_datasets
from model import AttentionUNet, CompactUNet


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Low-Light Enhancement Model")
    parser.add_argument('--model', type=str, required=True, choices=['compact', 'attention'],
                        help="Scegli quale modello valutare (compact o attention)")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed per riproducibilità (Richiesto dall'Assignment)")
    parser.add_argument('--num_vis', type=int, default=5,
                        help="Numero di immagini da visualizzare per dataset")
    return parser.parse_args()


def set_seed(seed):
    """Garantisce la riproducibilità degli esperimenti come da requisiti PDF."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


@torch.no_grad()
def evaluate_full_reference(model, dataloader, device):
    """Calcola le metriche Full-Reference (PSNR, SSIM) per i dataset appaiati."""
    model.eval()
    val_psnr, val_ssim = 0.0, 0.0

    pbar = tqdm(dataloader, desc='Full-Reference Eval')
    for low, high in pbar:
        low, high = low.to(device), high.to(device)
        output = model(low).float().clamp(0, 1)
        high = high.float()

        val_psnr += psnr(output, high).item()
        val_ssim += ssim(output, high).item()

    return val_psnr / len(dataloader), val_ssim / len(dataloader)


@torch.no_grad()
def evaluate_no_reference(model, dataloader, device):
    """Calcola le metriche No-Reference (BRISQUE) per i dataset spaiati (ExDark)."""
    model.eval()
    val_brisque = 0.0
    valid_batches = 0

    pbar = tqdm(dataloader, desc='No-Reference Eval (ExDark)')
    for low in pbar:
        low = low.to(device)
        output = model(low).float().clamp(0, 1)

        try:
            # BRISQUE può fallire su immagini con varianza quasi nulla
            val_brisque += brisque(output).item()
            valid_batches += 1
        except Exception:
            pass

    return val_brisque / max(1, valid_batches)


def visualize_dataset_samples(model, dataset, device, num_samples=5, save_path="out.png", has_gt=True,
                              title="Visual Evaluation"):
    """
    Ottimizzato: estrae indici casuali direttamente dal dataset senza caricare tutto in RAM.
    """
    model.eval()
    total_imgs = len(dataset)
    num_samples = min(num_samples, total_imgs)
    indices = random.sample(range(total_imgs), num_samples)

    cols = 3 if has_gt else 2
    fig, axes = plt.subplots(num_samples, cols, figsize=(4 * cols, 4 * num_samples))
    fig.suptitle(title, fontsize=16)

    for i, idx in enumerate(indices):
        if has_gt:
            low_img, high_img = dataset[idx]
            high_img = high_img.unsqueeze(0).to(device)
        else:
            low_img = dataset[idx]

        low_img = low_img.unsqueeze(0).to(device)

        with torch.no_grad():
            enhanced_img = model(low_img).clamp(0, 1)

        # Conversioni per Matplotlib
        low_np = low_img.squeeze(0).cpu().permute(1, 2, 0).numpy()
        enh_np = enhanced_img.squeeze(0).cpu().permute(1, 2, 0).numpy()

        # Gestione Assi
        ax_row = axes[i] if num_samples > 1 else axes

        ax_row[0].imshow(low_np)
        ax_row[0].set_title('Low Light Input')
        ax_row[0].axis('off')

        ax_row[1].imshow(enh_np)
        ax_row[1].set_title('Enhanced Output')
        ax_row[1].axis('off')

        if has_gt:
            high_np = high_img.squeeze(0).cpu().permute(1, 2, 0).numpy()
            ax_row[2].imshow(high_np)
            ax_row[2].set_title('Ground Truth')
            ax_row[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"-> Immagini salvate in: {save_path}")
    plt.close()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== Valutazione Modello: {args.model.upper()} | Device: {device} | Seed: {args.seed} ===")

    # Costruzione del percorso pesi coerente con train.py
    checkpoint_path = f'checkpoints/best_model_{args.model}.pth'
    if not os.path.exists(checkpoint_path):
        print(
            f"Errore CRITICO: Checkpoint '{checkpoint_path}' non trovato. Avvia prima il training con --model {args.model}")
        return

    # Inizializzazione dinamica
    if args.model == 'attention':
        model = AttentionUNet().to(device)
    else:
        model = CompactUNet().to(device)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    print("Pesi caricati con successo.")

    # Setup dati
    data_dir = './data'
    setup_datasets(data_dir)

    # Per le metriche usiamo i dataloader (batch_size=4 ottimale per A2000 Ada)
    from torch.utils.data import DataLoader
    test_in_dataset = PairedLowLightDataset(os.path.join(data_dir, 'LOL_v2'), split='test')
    test_cross_dataset = PairedLowLightDataset(os.path.join(data_dir, 'LOL_v1'), split='all')
    exdark_dataset = UnpairedDataset(os.path.join(data_dir, 'ExDark'))

    loader_in = DataLoader(test_in_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)
    loader_cross = DataLoader(test_cross_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)
    loader_exdark = DataLoader(exdark_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

    print("\n--- Valutazione Quantitativa ---")
    psnr_in, ssim_in = evaluate_full_reference(model, loader_in, device)
    print(f"[In-Domain: LOL-v2]      PSNR: {psnr_in:.4f} | SSIM: {ssim_in:.4f}")

    psnr_cross, ssim_cross = evaluate_full_reference(model, loader_cross, device)
    print(f"[Cross-Domain: LOL-v1]   PSNR: {psnr_cross:.4f} | SSIM: {ssim_cross:.4f}")

    brisque_score = evaluate_no_reference(model, loader_exdark, device)
    print(f"[Cross-Domain: ExDark]   BRISQUE (Più basso è meglio): {brisque_score:.4f}")

    print("\n--- Valutazione Qualitativa (Visuale) ---")
    os.makedirs('results_vis', exist_ok=True)

    visualize_dataset_samples(model, test_in_dataset, device, num_samples=args.num_vis,
                              save_path=f"results_vis/{args.model}_indomain_LOLv2.png", has_gt=True,
                              title=f"In-Domain ({args.model.upper()})")

    visualize_dataset_samples(model, test_cross_dataset, device, num_samples=args.num_vis,
                              save_path=f"results_vis/{args.model}_crossdomain_LOLv1.png", has_gt=True,
                              title=f"Cross-Domain LOLv1 ({args.model.upper()})")

    visualize_dataset_samples(model, exdark_dataset, device, num_samples=args.num_vis,
                              save_path=f"results_vis/{args.model}_crossdomain_ExDark.png", has_gt=False,
                              title=f"Cross-Domain ExDark ({args.model.upper()})")

    print(
        "\nValutazione terminata. Controlla la cartella 'results_vis' per identificare difetti come color cast, halo artifacts o over-smoothing come richiesto dal punto 8 del report.")


if __name__ == '__main__':
    main()