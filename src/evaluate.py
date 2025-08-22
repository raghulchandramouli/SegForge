# Evaluate
import argparse, yaml, torch, os
from torch.utils.data import DataLoader, Subset, Dataset
from src.dataset import SIDDataset
from src.model import SAMDecoderOnlyModel
from src.metrics import iou_score, dice_coeff, precision_recall
import numpy as np
from tqdm import tqdm
from PIL import Image

class FilteredSIDDataset(Dataset):
    """
    Dataset wrapper to filter in Real & Inpainted images only.
    
    Labels:
      0 - Real
      2 - Inpainted 
    """
    
    def __init__(self, base_dataset, target_labels=[0,2], max_samples=500):
        self.base_dataset = base_dataset
        self.target_labels = target_labels
        self.filtered_indices = []
        
        # find indices with target labels
        for i in range(len(base_dataset)):
            _, _, meta = base_dataset[i]
            if hasattr(meta, '__getitem__') and 'label' in meta:
                label = meta['label']
                
            else:
                sample = base_dataset.dataset[i]
                label = sample['label']
                
            if label in target_labels:
                self.filtered_indices.append(i)
                if len(self.filtered_indices) >= max_samples:
                    break
                
    def __len__(self):
        return len(self.filtered_indices)
    
    def __getitem__(self, idx):
        real_idx = self.filtered_indices[idx]
        return self.base_dataset[real_idx]

def validate_sid(cfg, ckpt_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = SAMDecoderOnlyModel(
        cfg['model']['sam_model_type'], 
        cfg['model']['sam_checkpoint'],
        freeze_sam=cfg['model'].get('freeze_sam', True),
        decoder_channels=cfg['model']['decoder_channels']
    )
    
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state'], strict=False)
    print("Model loaded from")
    
    model.to(device).eval()
    
    # Load SID validation dataset - filter for real (0) and tampered (2) only
    base_dataset = SIDDataset("saberzl/SID_Set", split="validation", 
                             img_size=cfg['data']['img_size'], cache_dir="./data")
    filtered_dataset = FilteredSIDDataset(base_dataset, target_labels=[0, 2], max_samples=500)
    val_loader = DataLoader(filtered_dataset, batch_size=4, shuffle=False, num_workers=4)
    
    print(f"Found {len(filtered_dataset)} real/tampered images (excluding synthetic)")
    
    # Create output directory
    os.makedirs("./validation_results", exist_ok=True)
    
    ious, dices, precs, recs = [], [], [], []
    sample_count = 0
    
    with torch.no_grad():
        for imgs, masks, _ in tqdm(val_loader, desc="Validating real/tampered"):
            imgs, masks = imgs.to(device), masks.to(device)
            
            feats = model.sam.image_encoder(imgs)
            logits = model.forward_from_embeddings(feats[-1] if isinstance(feats, (list, tuple)) else feats)
            
            ious.append(iou_score(logits, masks))
            dices.append(dice_coeff(logits, masks))
            p, r = precision_recall(logits, masks)
            precs.append(p)
            recs.append(r)
            
            # Save visualizations
            for j in range(imgs.shape[0]):
                # Convert tensors to images
                img_np = imgs[j].cpu().permute(1,2,0).numpy()
                img_np = (img_np * np.array([0.229,0.224,0.225]) + np.array([0.485,0.456,0.406]))
                img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
                
                gt_mask = (masks[j,0].cpu().numpy() * 255).astype(np.uint8)
                pred_mask = (torch.sigmoid(logits[j,0]) > 0.5).cpu().numpy().astype(np.uint8) * 255
                
                # Create triple visualization
                combined = Image.new('RGB', (1536, 512))
                combined.paste(Image.fromarray(img_np).resize((512,512)), (0,0))
                combined.paste(Image.fromarray(gt_mask).convert('RGB').resize((512,512)), (512,0))
                combined.paste(Image.fromarray(pred_mask).convert('RGB').resize((512,512)), (1024,0))
                
                combined.save(f"./validation_results/sample_{sample_count:03d}.png")
                sample_count += 1
    
    print(f"SID Validation Results (real + tampered only):")
    print(f"IoU: {np.mean(ious):.4f}")
    print(f"Dice: {np.mean(dices):.4f}")
    print(f"Precision: {np.mean(precs):.4f}")
    print(f"Recall: {np.mean(recs):.4f}")
    print(f"Saved {sample_count} validation images to ./validation_results/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--checkpoint', required=True)
    args = parser.parse_args()
    
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    validate_sid(cfg, args.checkpoint)