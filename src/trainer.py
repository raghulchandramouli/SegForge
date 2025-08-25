import os, time
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from .utils import save_checkpoint, makedirs
from .metrics import dice_coeff, iou_score, precision_recall

class Trainer:
    def __init__(self, model, optimizer, train_loader, val_loader, cfg, device):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device
        self.writer = SummaryWriter(cfg['logging']['tb_logdir'])
        makedirs(cfg['logging']['checkpoint_dir'])
        self.amp = cfg['train'].get('amp', True)
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.amp and device.type=='cuda')


    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        n = 0
        log_every = self.cfg['logging'].get('log_every_n_batches', 50)
        
        pbar = tqdm(self.train_loader, desc=f'Train Epoch {epoch}')
        for batch_idx, batch in enumerate(pbar):
            imgs, masks, _ = batch
            imgs = imgs.to(self.device)
            masks = masks.to(self.device)
            
            with torch.amp.autocast('cuda', enabled=self.amp and self.device.type=='cuda'):
                logits = self.model.forward_from_image(imgs)

            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, masks)
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item() * imgs.size(0)
            n += imgs.size(0)
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Log for every batch for real-time tracking
            if batch_idx % log_every == 0:
                global_step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('train/batch_loss', loss.item(), global_step)
                self.writer.add_scalar('train/running_avg_loss', running_loss/n, global_step)
                
        avg_loss = running_loss / max(n,1)
        self.writer.add_scalar('train/loss', avg_loss, epoch)
        return avg_loss

    def validate(self, epoch):
        self.model.eval()
        running_loss = 0.0
        ious = []
        dices = []
        precisions = []
        recalls = []
        n = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Val Epoch {epoch}')
            for batch in pbar:
                imgs, masks, _ = batch
                imgs = imgs.to(self.device)
                masks = masks.to(self.device)
                logits = self.model.forward_from_image(imgs)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, masks)
                running_loss += loss.item() * imgs.size(0)
                ious.append(iou_score(logits, masks))
                dices.append(dice_coeff(logits, masks))
                p, r = precision_recall(logits, masks)
                precisions.append(p); recalls.append(r)
                n += imgs.size(0)
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        avg_loss = running_loss / max(n,1)
        avg_iou = sum(ious)/len(ious) if ious else 0.0
        avg_dice = sum(dices)/len(dices) if dices else 0.0
        avg_p = sum(precisions)/len(precisions) if precisions else 0.0
        avg_r = sum(recalls)/len(recalls) if recalls else 0.0

        self.writer.add_scalar('val/loss', avg_loss, epoch)
        self.writer.add_scalar('val/iou', avg_iou, epoch)
        self.writer.add_scalar('val/dice', avg_dice, epoch)
        self.writer.add_scalar('val/precision', avg_p, epoch)
        self.writer.add_scalar('val/recall', avg_r, epoch)

        return avg_loss, avg_iou, avg_dice, avg_p, avg_r

    def save(self, epoch):
        path = os.path.join(self.cfg['logging']['checkpoint_dir'], f'epoch_{epoch}.pth')
        save_checkpoint({'epoch': epoch, 'model_state': self.model.state_dict(), 'opt_state': self.optimizer.state_dict()}, path)
        return path
