import os
import argparse
import torch
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from dataset import PairedLowLightDataset, UnpairedDataset
from model import CompactUNet, AttentionUNet

def parse_args():
    parser = argparse.ArgumentParser(description="Export images for visual inspection")
    parser.add_argument('--model', type=str, default='compact', choices=['compact', 'attention'])
    parser.add_argument('--use_transposed_conv', action='store_true')
    return parser.parse_args()

def export_images(args):
    """
    Exports a set of low-light, predicted, and high-light images for visual inspection.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_name = args.model
    use_transposed = args.use_transposed_conv
    
    bilinear_flag = not use_transposed
    if model_name == 'attention':
        model = AttentionUNet(bilinear=bilinear_flag).to(device)
    else:
        model = CompactUNet(bilinear=bilinear_flag).to(device)
        
    model.load_state_dict(torch.load(f'checkpoints/best_model_{model_name}.pth', map_location=device))
    model.eval()

    data_dir = './data'
    test_dataset = PairedLowLightDataset(os.path.join(data_dir, 'LOL_v1'), split='all', img_size=256)
    # test_dataset = UnpairedDataset(os.path.join(data_dir, 'ExDark'), img_size=256)
    
    loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    out_dir = f'examples/{model_name}_transposed_{use_transposed}'
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Saving images to {out_dir}")
    
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= 30:
                break
                
            if isinstance(batch, list) or isinstance(batch, tuple):
                low, high = batch
                high = high.to(device)
            else:
                low = batch
                high = None
                
            low = low.to(device)
            output = model(low)
            
            save_image(low, os.path.join(out_dir, f'img_{i}_1_low.png'))
            save_image(output, os.path.join(out_dir, f'img_{i}_2_pred.png'))
            if high is not None:
                save_image(high, os.path.join(out_dir, f'img_{i}_3_target.png'))

if __name__ == '__main__':
    args = parse_args()
    export_images(args)
    print("Image export completed.")
