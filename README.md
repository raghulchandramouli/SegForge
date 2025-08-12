# SegForge: SAM-based Image Manipulation Detection

A PyTorch implementation for detecting and segmenting manipulated regions in images using SAM (Segment Anything Model) as a feature extractor with a custom decoder head.

## Overview

SegForge trains a segmentation model to detect tampered regions in images from the SID_Set dataset. The model uses a frozen SAM encoder for feature extraction and trains only a lightweight decoder head for segmentation.

## Dataset

**SID_Set (Social media Image Detection dataSet)**
- **300K images** with comprehensive annotations
- **3 categories:**
  - `0`: Real images (no manipulation)
  - `1`: Fully synthetic images 
  - `2`: Tampered images (with binary masks for manipulated regions)
- **Splits:** 210K train, 30K validation, 60K test

## Architecture

- **Encoder:** Frozen SAM ViT-B image encoder
- **Decoder:** Custom 4-layer upsampling decoder (256→128→64→32 channels)
- **Output:** 1024×1024 binary segmentation mask
- **Loss:** Binary Cross Entropy with Logits

## Installation

```bash
# Clone repository
git clone <https://github.com/raghulchandramouli/SegForge.git>
cd SegForge

# Create conda environment
conda create -n SegForge python=3.11
conda activate SegForge

# Install dependencies
pip install -r requirements.txt
```

## Train with default config
```bash
python -m src.train --config config.yaml
```

## Monitor training with TensorBoard
```bash
tensorboard --logdir=./logs
```

## *yaml config files*
```bash
model:
  sam_model_type: "vit_b"
  decoder_channels: [256, 128, 64, 32]

data:
  hf_repo: "saberzl/SID_Set"
  img_size: 1024

train:
  epochs: 6
  batch_size: 4
  lr: 0.0005
  weight_decay: 0.006
```

# Project Structure
``` bash
SegForge/
├── src/
│   ├── train.py          # Training script
│   ├── model.py          # SAM + decoder model
│   ├── dataset.py        # SID_Set dataset 
|
│   ├── trainer.py        # Training loop with metrics
│   ├── transforms.py     # Image/mask preprocessing
│   ├── metrics.py        # IoU, Dice, Precision/Recall
│   └── utils.py          # Utilities & SAM checkpoint download
├── config.yaml           # Training configuration
├── checkpoints/          # Model checkpoints
├── logs/                 # TensorBoard logs
└── data/                 # Downloaded dataset 
```
