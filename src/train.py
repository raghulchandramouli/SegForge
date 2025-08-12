import argparse, yaml, torch
from torch.utils.data import DataLoader
from .utils import load_config, set_seed, makedirs
from .dataset import SIDDataset
from .model import SAMDecoderOnlyModel
from .trainer import Trainer
import os

def build_loaders(cfg):
    
    train_ds = SIDDataset(cfg['data']['hf_repo'], split=cfg['data']['hf_split_train'],
                         img_size=cfg['data']['img_size'], cache_dir=cfg['data'].get('hf_cache_dir', './data'))
    val_ds = SIDDataset(cfg['data']['hf_repo'], split=cfg['data']['hf_split_val'],
                       img_size=cfg['data']['img_size'], cache_dir=cfg['data'].get('hf_cache_dir', './data'))
    
    train_loader = DataLoader(train_ds, batch_size=cfg['train']['batch_size'], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=cfg['train']['batch_size'], shuffle=False, num_workers=2)
    return train_loader, val_loader


def main(config):
    cfg = load_config(config)
    set_seed(cfg.get('seed', 42))
    device = torch.device(cfg['device'] if torch.cuda.is_available() else 'cpu')
    makedirs(cfg['logging']['checkpoint_dir']); makedirs(cfg['logging']['tb_logdir'])
    train_loader, val_loader = build_loaders(cfg)

    # instantiate model
    model = SAMDecoderOnlyModel(
        sam_type=cfg['model']['sam_model_type'],
        sam_ckpt=cfg['model']['sam_checkpoint'],
        freeze_sam=cfg['model'].get('freeze_sam', True),
        decoder_channels=cfg['model'].get('decoder_channels', [256,128,64])
    ).to(device)

    # Only decoder params should be trainable (sanity check)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable params: {sum(p.numel() for p in trainable)}")

    optimizer = torch.optim.AdamW(trainable, lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'])
    trainer = Trainer(model, optimizer, train_loader, val_loader, cfg, device)

    for epoch in range(1, cfg['train']['epochs']+1):
        tloss = trainer.train_epoch(epoch)
        vloss, viou, vdice, vp, vr = trainer.validate(epoch)
        print(f"Epoch {epoch}: train_loss={tloss:.4f} val_loss={vloss:.4f} iou={viou:.4f} dice={vdice:.4f} prec={vp:.4f} rec={vr:.4f}")
        if epoch % cfg['logging']['save_every'] == 0:
            trainer.save(epoch)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    args = parser.parse_args()
    main(args.config)
