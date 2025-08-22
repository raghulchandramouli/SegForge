import torch
ckpt = torch.load('/mnt/g/Authenta/research/authenta-inpainting-detection/SegForge/checkpoints/finetuned/epoch_6.pth', map_location='cpu')

# Look for custom decoder weights (not SAM's mask_decoder)
custom_decoder_keys = [k for k in ckpt['model_state'].keys() if k.startswith('decoder.')]
print('Custom decoder keys:', len(custom_decoder_keys))
print('Custom decoder weights:', custom_decoder_keys[:10])

if custom_decoder_keys:
    sample_weight = ckpt['model_state'][custom_decoder_keys[0]]
    print(f'Custom decoder weight stats: mean={sample_weight.mean():.6f}, std={sample_weight.std():.6f}')
else:
    print('No custom decoder weights found!')
    print('All keys starting with "decoder":', [k for k in ckpt['model_state'].keys() if 'decoder' in k.lower()][:5])
