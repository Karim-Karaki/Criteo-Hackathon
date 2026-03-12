import torch

clip = torch.load(r'C:\Users\karim\Desktop\Data Science Projects\Vast AI\outputs\clip_hierarchical_best.pth', map_location='cpu')
print("=== CLIP ===")
print(type(clip))
if isinstance(clip, dict):
    print(list(clip.keys())[:10])



dino = torch.load(
    r'C:\Users\karim\Desktop\Data Science Projects\Vast AI\outputs\dino_best.pth',
    map_location='cpu',
    weights_only=True
)
print(type(dino))
print(list(dino.keys())[:10])


print("\n=== DINO ===")

import torch
dino = torch.load(r'C:\Users\karim\Desktop\Data Science Projects\Vast AI\outputs\dino_best.pth', map_location='cpu', weights_only=True)

# Count blocks to identify model size
blocks = [k for k in dino.keys() if 'backbone.blocks.' in k]
max_block = max([int(k.split('backbone.blocks.')[1].split('.')[0]) for k in blocks])
print(f"Number of blocks: {max_block + 1}")
# 12 blocks = ViT-S or ViT-B
# 24 blocks = ViT-L

print(dino['backbone.cls_token'].shape)
# Small  → torch.Size([1, 1, 384])
# Base   → torch.Size([1, 1, 768])