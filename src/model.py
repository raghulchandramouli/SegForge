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
    
# class LowLevelAdapter(nn.Module):
#     """
#     Pre-processes input image to extract low-level manipulation artifacts
#     before feeding to SAM Encoder
    
#     Architecture:
#         - Multi-head CNN to detect texture inconsistencies
#         - Learns compression artifacts, noise patterns, blending boundaries
#         - Outputs enchanced  3-channels image for SAM Processing
        
#     Why??
#         - hopefully domain gap between natural images and manipulation detection
#         - Feature Extraction while adapting to task
#         - Trainable end-to-end with unfrozen  
#     """
    
#     def __init__(self, in_channels=3):
#         super().__init__()
#         self.low_level = nn.Sequential(
#             nn.Conv2d(in_channels, 32, 3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.Conv2d(32, 3, 1)  # Back to 3 channels
#         )
     
#     def forward(self, x):
#         return self.low_level(x)
            
        
class SAMDecoderOnlyModel(nn.Module):
    """
    SAM encoder (frozen) + custom decoder - NO ADAPTER
    """
    def __init__(self, sam_type, sam_ckpt, emb_channels=None, freeze_sam=True, decoder_channels=[256,128,64]):
        super().__init__()
        download_sam_checkpoint(sam_type, sam_ckpt)
        
        # Load SAM (frozen by default)
        self.sam = sam_model_registry[sam_type](checkpoint=sam_ckpt)
        
        # Freeze SAM parameters
        if freeze_sam:
            for p in self.sam.parameters():
                p.requires_grad = False
        
        if emb_channels is None:
            try:
                emb_channels = self.sam.image_encoder.neck[-1].out_channels
            except Exception:
                emb_channels = 256

        self.decoder = DecoderHead(emb_channels, channels=decoder_channels)

    def forward_from_image(self, img):
        """Direct SAM encoder -> decoder (no adapter)"""
        feats = self.sam.image_encoder(img)  # No adapter preprocessing
        if isinstance(feats, (list, tuple)):
            feats = feats[-1]
        logits = self.decoder(feats)
        return logits

    def forward_from_embeddings(self, emb):
        return self.decoder(emb)

