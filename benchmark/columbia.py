# benchmark/columbia.py
import sys
sys.path.append('.')

import argparse, yaml, torch, os
from torch.utils.data import DataLoader, Dataset
from src.model import SAMDecoderOnlyModel
from src.transforms import get_image_transforms
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

class ColumbiaDataset(Dataset):
    def __init__(self, img_dir, img_size=1024):
        self.img_dir = img_dir
        self.img_size = img_size
        self.image_tf = get_image_transforms(img_size)
        self.samples = self._load_samples()
        
    def _load_samples(self):
        samples = []
        
        for img_file in os.listdir(self.img_dir):
            if img_file.endswith(('.jpg', '.png', '.tif')):
                img_path = os.path.join(self.img_dir, img_file)
                samples.append((img_path, img_file))
        
        print(f"Found {len(samples)} Columbia images")
        return samples
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, img_name = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        return self.image_tf(img), img_name, img_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--checkpoint', required=True)
    args = parser.parse_args()
    
    # Fixed paths
    img_dir = "/mnt/g/Authenta/data/authenta-inpainting-detection/Columbia/4cam_splc"
    output_dir = "/mnt/g/Authenta/data/benchmark/Columbia"
    
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
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
    
    # Load Columbia dataset
    dataset = ColumbiaDataset(img_dir, cfg['data']['img_size'])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
    
    with torch.no_grad():
        for imgs, img_names, img_paths in tqdm(dataloader, desc="Inferencing Columbia"):
            imgs = imgs.to(device)
            feats = model.sam.image_encoder(imgs)
            logits = model.forward_from_embeddings(feats[-1] if isinstance(feats, (list, tuple)) else feats)
            
            # Create visualization for each image
            for j in range(imgs.shape[0]):
                # Load original image
                original_img = Image.open(img_paths[j]).resize((512, 512))
                
                # Generate predicted mask
                pred_mask = (torch.sigmoid(logits[j,0]) > 0.5).cpu().numpy()
                pred_mask = Image.fromarray((pred_mask * 255).astype(np.uint8)).resize((512, 512))
                
                # Create dual visualization (Original | Predicted Mask)
                combined = Image.new('RGB', (1024, 562), 'white')
                
                # Add headings
                draw = ImageDraw.Draw(combined)
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
                except:
                    font = ImageFont.load_default()
                
                draw.text((200, 10), "Original Image", fill='black', font=font)
                draw.text((700, 10), "Pred Mask", fill='black', font=font)
                
                # Paste images
                combined.paste(original_img, (0, 50))
                combined.paste(pred_mask.convert('RGB'), (512, 50))
                
                # Save with original filename
                output_path = os.path.join(output_dir, f"pred_{img_names[j]}")
                combined.save(output_path)
                
                # Also save just the mask
                mask_path = os.path.join(output_dir, f"mask_{img_names[j]}")
                pred_mask.save(mask_path)
    
    print(f"Columbia inference completed!")
    print(f"Visualizations saved to: {output_dir}")
    print(f"Generated {len(dataset)} prediction masks")

if __name__ == "__main__":
    main()
