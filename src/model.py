import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything import sam_model_registry
from .utils import download_sam_checkpoint




class ConvUpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return self.block(x)

class DecoderHead(nn.Module):
    def __init__(self, in_ch, channels=[256,128,64,32]):
        super().__init__()
        self.up1 = ConvUpBlock(in_ch, channels[0])
        self.up2 = ConvUpBlock(channels[0], channels[1])
        self.up3 = ConvUpBlock(channels[1], channels[2])
        self.up4 = ConvUpBlock(channels[2], channels[3])
        self.out_conv = nn.Conv2d(channels[3], 1, kernel_size=1)

    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        logits = self.out_conv(x)
        return logits


class SAMDecoderOnlyModel(nn.Module):
    """
    Wraps SAM image encoder as feature extractor (frozen) and trains a conv decoder on top of embeddings.
    - sam_type: 'vit_b' etc
    - sam_ckpt: path to SAM checkpoint
    """
    def __init__(self, sam_type, sam_ckpt, emb_channels=None, freeze_sam=True, decoder_channels=[256,128,64]):
        super().__init__()
        #Download SAM chechpoints if not already present:
        download_sam_checkpoint(sam_type, sam_ckpt)
        # load SAM
        self.sam = sam_model_registry[sam_type](checkpoint=sam_ckpt)
        # We'll use SAM's image encoder to extract embeddings.
        # Freeze if required:
        if freeze_sam:
            for p in self.sam.parameters():
                p.requires_grad = False

        # Determine embedding channels: try to query image_encoder output if available
        if emb_channels is None:
            # best-effort: many SAM variants end with a projection dim, try to access example attribute
            try:
                # Some SAMs have image_encoder output channels attribute; adapt per implementation
                emb_channels = self.sam.image_encoder.neck[-1].out_channels
            except Exception:
                # fallback
                emb_channels = 256

        self.decoder = DecoderHead(emb_channels, channels=decoder_channels)

    def forward_from_image(self, img):
        """
        Run SAM image encoder -> get embeddings -> decoder
        img: Bx3xHxW normalized
        """
        # use SAM's image encoder. exact API depends on version; here we assume sam.image_encoder returns list or tensor
        feats = self.sam.image_encoder(img)
        if isinstance(feats, (list, tuple)):
            feats = feats[-1]
        # feats: BxCxhxw
        logits = self.decoder(feats)
        return logits

    def forward_from_embeddings(self, emb):
        """Directly pass embeddings (B x C x h x w) to decoder."""
        return self.decoder(emb)
