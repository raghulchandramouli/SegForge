# src/dataset.py  (HFStreamingDataset portion)
from torch.utils.data import IterableDataset, Dataset
import torch
from datasets import load_dataset
from PIL import Image
import numpy as np
from .transforms import get_image_transforms, mask_transform

class SIDDataset(Dataset):
    """Regular Dataset for downloaded SID_Set with proper progress bars"""
    def __init__(self, repo, split="train", img_size=1024, cache_dir="./data"):
        self.img_size = img_size
        self.image_tf = get_image_transforms(img_size)
        print(f"Loading {split} dataset...")
        self.dataset = load_dataset(repo, split=split, cache_dir=cache_dir)
        print(f"Loaded {len(self.dataset)} samples")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        img = sample['image']
        mask = sample.get('mask')
        label = sample['label']
        img_id = sample.get('img_id', idx)
        
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        
        # Only tampered images (label=2) have masks
        if label == 2 and mask is not None:
            if not isinstance(mask, Image.Image):
                mask = Image.fromarray(mask)
            mask_tensor = mask_transform(mask, self.img_size)
        else:
            mask_tensor = torch.zeros(1, self.img_size, self.img_size)
        
        return self.image_tf(img), mask_tensor, img_id




class HFStreamingDataset(IterableDataset):
    """
    Streams samples from a HF dataset like saberzl/SID_Set.
    Yields tuples: (img_tensor, mask_tensor, meta_dict)
    meta_dict contains: {'img_id': str, 'label': int}
    Notes:
      - If label != 2 (not tampered), the mask is set to all-zero (negative example).
      - Use DataLoader(..., num_workers=0) with this IterableDataset.
      - Optionally pass `max_samples` to limit samples (useful for debugging).
    """
    def __init__(self, repo, split="train", img_size=1024, cache_dir=None, use_auth_token=False, max_samples=None):
        self.repo = repo
        self.split = split
        self.cache_dir = cache_dir
        self.use_auth_token = use_auth_token
        self.img_size = img_size
        self.image_tf = get_image_transforms(img_size)
        self.max_samples = int(max_samples) if max_samples is not None else None

    def _build_dataset(self):
        load_kwargs = {}
        if self.cache_dir:
            load_kwargs['cache_dir'] = self.cache_dir
        if self.use_auth_token:
            load_kwargs['use_auth_token'] = True
        # streaming True -> iterator-like dataset
        return load_dataset(self.repo, split=self.split, streaming=True, **load_kwargs)

    def __iter__(self):
        ds = self._build_dataset()
        count = 0
        for sample in ds:
            # sample expected keys: image, mask, label, img_id
            img = sample.get('image', None)
            mask = sample.get('mask', None)
            label = sample.get('label', None)
            img_id = sample.get('img_id', sample.get('id', None))

            if img is None or label is None:
                # skip bad samples
                continue

            # coerce to PIL.Image if needed
            if not isinstance(img, Image.Image):
                img = Image.fromarray(np.array(img))
            # For tampered images (label==2) mask should exist, but be defensive:
            if label == 2 and mask is not None:
                if not isinstance(mask, Image.Image):
                    mask = Image.fromarray(np.array(mask))
            else:
                # create an all-zero single-channel PIL image matching img size
                mask = Image.new("L", img.size, 0)

            img_t = self.image_tf(img)                 # tensor 3xHxW normalized
            mask_t = mask_transform(mask, self.img_size)  # tensor 1xHxW binary {0,1}

            meta = {'img_id': img_id, 'label': int(label)}
            yield img_t, mask_t, meta

            count += 1
            if self.max_samples is not None and count >= self.max_samples:
                break
