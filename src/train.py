import argparse, yaml, torch
from torch.utils.data import DataLoader
from .utils import load_config, set_seed, makedirs
from .dataset import SIDDataset, LocalInpaintingDataset
from .model import SAMDecoderOnlyModel
from .trainer import Trainer
import os

def build_loaders(cfg):
    if cfg['data']['use_hf']:
        # Use HuggingFace SID dataset
        train_ds = SIDDataset(cfg['data']['hf_repo'], split=cfg['data']['hf_split_train'],
                             img_size=cfg['data']['img_size'], cache_dir=cfg['data'].get('hf_cache_dir', './data'))
        val_ds = SIDDataset(cfg['data']['hf_repo'], split=cfg['data']['hf_split_val'],
                           img_size=cfg['data']['img_size'], cache_dir=cfg['data'].get('hf_cache_dir', './data'))
    else:
        # Use local inpainting dataset
        train_ds = LocalInpaintingDataset(
            cfg['data']['img_dir'], cfg['data']['mask_dir'],
            img_size=cfg['data']['img_size'], train=True, 
            train_split=cfg['data']['train_split']
        )
        val_ds = LocalInpaintingDataset(
            cfg['data']['img_dir'], cfg['data']['mask_dir'],
            img_size=cfg['data']['img_size'], train=False,
            train_split=cfg['data']['train_split']
        )
    
    train_loader = DataLoader(train_ds, batch_size=cfg['train']['batch_size'], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=cfg['train']['batch_size'], shuffle=False, num_workers=2)
    return train_loader, val_loader
    
def main(config):
    cfg = load_config(config)
    set_seed(cfg.get('seed', 42))
    device = torch.device(cfg['device'] if torch.cuda.is_available() else 'cpu')
    makedirs(cfg['logging']['checkpoint_dir']); makedirs(cfg['logging']['tb_logdir'])
    train_loader, val_loader = build_loaders(cfg)

    model = SAMDecoderOnlyModel(
        sam_type=cfg['model']['sam_model_type'],
        sam_ckpt=cfg['model']['sam_checkpoint'],
        freeze_sam=cfg['model'].get('freeze_sam', True),
        decoder_channels=cfg['model'].get('decoder_channels', [256,128,64])
    ).to(device)

    
     # ADD THIS CHECK HERE 
    
    #adapter_params = sum(p.numel() for p in model.adapter.parameters() if p.requires_grad)
    sam_trainable = sum(p.numel() for p in model.sam.parameters() if p.requires_grad)
    decoder_trainable = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"SAM trainable params: {sam_trainable:,}")
    print(f"Decoder trainable params: {decoder_trainable:,}")
    #print(f"Adapter trainable params: {adapter_params:,}")
    print(f"Total trainable params: {total_trainable:,}")
    
    
    if sam_trainable > 0:
        print(" SAM is UNFROZEN - will be trained")
    else:
        print("SAM is FROZEN - only decoder will be trained")
        
    # if adapter_params > 0:
    #     print("Adapter is UNFROZEN - will be trained")
    # else:
    #     print("Adapter is FROZEN - only decoder will be trained")
    
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'])
    
    # Resume from checkpoint if specified
# Resume from checkpoint if specified
    start_epoch = 1
    if 'resume_from' in cfg['train']:
        checkpoint = torch.load(cfg['train']['resume_from'], map_location='cpu')
        model.load_state_dict(checkpoint['model_state'], strict=False)
        # Skip optimizer loading when parameter count changes
        # if 'opt_state' in checkpoint:
        #     optimizer.load_state_dict(checkpoint['opt_state'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"Resumed from epoch {start_epoch-1} (optimizer reset)")


    trainer = Trainer(model, optimizer, train_loader, val_loader, cfg, device)

    for epoch in range(start_epoch, cfg['train']['epochs']+1):
        tloss = trainer.train_epoch(epoch)
        vloss, viou, vdice, vp, vr = trainer.validate(epoch)
        print(f"Epoch {epoch}: train_loss={tloss:.4f} val_loss={vloss:.4f} iou={viou:.4f} dice={vdice:.4f}")
        if epoch % cfg['logging']['save_every'] == 0:
            trainer.save(epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    args = parser.parse_args()
    main(args.config)
