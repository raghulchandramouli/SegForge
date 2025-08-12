import argparse, yaml, torch, os
from torch.utils.data import DataLoader
from .dataset import HFStreamingDataset
from .model import SAMDecoderOnlyModel
from .metrics import iou_score, dice_coeff, precision_recall
from PIL import Image
import numpy as np
from tqdm import tqdm

def run_inference(cfg, ckpt_path, out_dir):
    device = torch.device(cfg['device'] if torch.cuda.is_available() else 'cpu')
    ds = HFStreamingDataset(cfg['data']['hf_repo'], split=cfg['data']['hf_split_test'], img_size=cfg['data']['img_size'], cache_dir=cfg['data'].get('hf_cache_dir'))
    loader = DataLoader(ds, batch_size=1, num_workers=0)

    model = SAMDecoderOnlyModel(cfg['model']['sam_model_type'], cfg['model']['sam_checkpoint'], freeze_sam=cfg['model'].get('freeze_sam', True), decoder_channels=cfg['model'].get('decoder_channels',[256,128,64]))
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu')['model_state'] if os.path.isdir(ckpt_path) or ckpt_path.endswith('.pth') else torch.load(ckpt_path))
    model.to(device)
    model.eval()

    ious=[]; dices=[]; precs=[]; recs=[]
    os.makedirs(out_dir, exist_ok=True)

    for imgs, masks, img_id in tqdm(loader, desc="Inference"):
        imgs = imgs.to(device); masks = masks.to(device)
        with torch.no_grad():
            feats = model.sam.image_encoder(imgs)
            if isinstance(feats, (list, tuple)):
                feats = feats[-1]
            logits = model.forward_from_embeddings(feats)
        ious.append(iou_score(logits, masks))
        dices.append(dice_coeff(logits, masks))
        p,r = precision_recall(logits, masks)
        precs.append(p); recs.append(r)

        # save mask overlay
        pred_mask = (torch.sigmoid(logits) > 0.5).cpu().numpy()[0,0]
        # convert input image to uint8 and overlay red mask
        img_np = (imgs.cpu().numpy()[0].transpose(1,2,0) * np.array([0.229,0.224,0.225]) + np.array([0.485,0.456,0.406]))
        img_np = np.clip(img_np*255,0,255).astype(np.uint8)
        overlay = img_np.copy()
        overlay[pred_mask==1] = [255,0,0]
        out_path = os.path.join(out_dir, f"{img_id if img_id is not None else 'sample'}.png")
        Image.fromarray(overlay).save(out_path)

    # summary
    import statistics
    print("Results:")
    print("IoU:", statistics.mean(ious))
    print("Dice:", statistics.mean(dices))
    print("Precision:", statistics.mean(precs))
    print("Recall:", statistics.mean(recs))
