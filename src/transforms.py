from torchvision import transforms
from PIL import Image
import torch

def get_image_transforms(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Lambda(lambda x: x.convert('RGB') if x.mode != 'RGB' else x),  # Convert to RGB
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])


def mask_transform(mask, img_size):
    # Ensure PIL
    if not isinstance(mask, Image.Image):
        mask = Image.fromarray(mask)

    # Convert to grayscale
    mask = mask.convert('L')

    # Resize with NEAREST to preserve labels
    mask = mask.resize((img_size, img_size), resample=Image.NEAREST)

    # Convert to tensor and binarize
    mask_tensor = transforms.ToTensor()(mask)  # Shape: (1, H, W)
    mask_tensor = (mask_tensor > 0.5).float()

    # Ensure float32 for loss functions like BCE/Dice
    return mask_tensor.type(torch.float32)
