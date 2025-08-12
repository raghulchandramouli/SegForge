import os, random, yaml
import numpy as np
import torch
from pathlib import Path
import urllib.request

def load_config(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
        return config
        
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def makedirs(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def download_sam_checkpoint(sam_type, checkpoint_path):
    """Download SAM checkpoint if it doesn't exist."""
    if os.path.exists(checkpoint_path):
        return
    
    urls = {
        'vit_b': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth',
        'vit_l': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth', 
        'vit_h': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'
    }
    
    makedirs(os.path.dirname(checkpoint_path))
    print(f"Downloading {sam_type} checkpoint...")
    urllib.request.urlretrieve(urls[sam_type], checkpoint_path)


def save_checkpoint(state, path):
    makedirs(os.path.dirname(path))
    torch.save(state, path)
    
