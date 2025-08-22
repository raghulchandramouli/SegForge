# Resolving import issue
import sys
sys.path.append('.')

import argparse, yaml, torch, os
from torch.utils.data import DataLoader, Dataset
from src.model import SAMDecoderOnlyModel
from src.metrics import iou_score, dice_coeff, precision_recall
from src.transforms import get_image_transforms, mask_transform
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

class CASIADataset(Dataset):
    def __init__(self, tp_dir, gt_dir, img_size=1024):
        self.tp_dir = tp_dir
        self.gt_dir = gt_dir
        self.img_size = img_size
        self.image_tf = get_image_transforms(img_size)
        self.samples = self._load_samples()
        
    def _load_samples(self):
        samples = []
        
        for img_file in os.listdir(self.tp_dir):
            if img_file.endswith(('.jpg', '.png', '.tif')):
                img_path = os.path.join(self.tp_dir, img_file)
                
                # CASIA naming: Tp_S_NNN_S_N_pla00070_pla00070_00599 -> Tp_S_NNN_S_N_pla00070_pla00070_00599_gt
                base_name = os.path.splitext(img_file)[0]  # Remove extension
                gt_file = f"{base_name}_gt.png"  # Add _gt suffix
                gt_path = os.path.join(self.gt_dir, gt_file)
                
                if os.path.exists(gt_path):
                    samples.append((img_path, gt_path, img_file))
                else:
                    # Debug: print missing files
                    print(f"Missing GT for: {img_file} -> looking for: {gt_file}")
        
        print(f"Found {len(samples)} CASIA samples with ground truth")
        return samples
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, gt_path, img_name = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(gt_path).convert('L')
        return self.image_tf(img), mask_transform(mask, self.img_size), img_name, img_path, gt_path
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--checkpoint', required=True)
    args = parser.parse_args()
    
    # fixed path
    tp_dir = "/mnt/g/Authenta/data/authenta-inpainting-detection/CASIA/Test/Tp"
    gt_dir = "/mnt/g/Authenta/data/authenta-inpainting-detection/CASIA/Test/GroundTruth"
    output_dir = "/mnt/g/Authenta/data/benchmark/CASIA"
    
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load 
    model = SAMDecoderOnlyModel(
        cfg['model']['sam_model_type'],
        cfg['model']['sam_checkpoint'],
        freeze_sam=cfg['model'].get('freeze_sam', True),
        decoder_channels=cfg['model']['decoder_channels']
    )
    
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model_state'], strict=False)
    model.to(device).eval()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load CASIA dataset
    dataset = CASIADataset(tp_dir, gt_dir, cfg['data']['img_size'])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
    
    ious, dices, precs, recs = [], [], [], []
    
    with torch.no_grad():
        for imgs, masks, img_names, img_paths, gt_paths in tqdm(dataloader, desc="Evaluating CASIA"):
            imgs, masks = imgs.to(device), masks.to(device)
            feats = model.sam.image_encoder(imgs)
            logits = model.forward_from_embeddings(feats[-1] if isinstance(feats, (list, tuple)) else feats)
            
            ious.append(iou_score(logits, masks))
            dices.append(dice_coeff(logits, masks))
            p, r = precision_recall(logits, masks)
            precs.append(p)
            recs.append(r)
            
            # Create triple visualization
            for j in range(imgs.shape[0]):
                # Load original images for visualization
                tampered_img = Image.open(img_paths[j]).resize((512, 512))
                gt_mask = Image.open(gt_paths[j]).resize((512, 512))
                
                # Generate predicted mask
                pred_mask = (torch.sigmoid(logits[j,0]) > 0.5).cpu().numpy()
                pred_mask = Image.fromarray((pred_mask * 255).astype(np.uint8)).resize((512, 512))
                
                # Create combined image with headings
                combined = Image.new('RGB', (1536, 562), 'white')
                
                # Add headings
                draw = ImageDraw.Draw(combined)
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
                except:
                    font = ImageFont.load_default()
                
                draw.text((180, 10), "Tampered Image", fill='black', font=font)
                draw.text((700, 10), "GT Mask", fill='black', font=font)
                draw.text((1250, 10), "Pred Mask", fill='black', font=font)
                
                # Paste images
                combined.paste(tampered_img, (0, 50))
                combined.paste(gt_mask.convert('RGB'), (512, 50))
                combined.paste(pred_mask.convert('RGB'), (1024, 50))
                
                # Save with original filename
                output_path = os.path.join(output_dir, f"triple_{img_names[j]}")
                combined.save(output_path)
    
    print(f"\nCASIA Results:")
    print(f"IoU: {np.mean(ious):.4f}")
    print(f"Dice: {np.mean(dices):.4f}")
    print(f"Precision: {np.mean(precs):.4f}")
    print(f"Recall: {np.mean(recs):.4f}")
    print(f"F1: {2*np.mean(precs)*np.mean(recs)/(np.mean(precs)+np.mean(recs)):.4f}")
    print(f"Triple visualizations saved to: {output_dir}")

if __name__ == "__main__":
    main()