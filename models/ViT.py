"""
Author: Georgios Voulgaris
Date: 17/02/2026
Description: Light implementation of ViT_Tiny transformer implementation for image classification.
"""

import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size=128, num_heads=4, mlp_ratio=4, dropout=0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True, dropout=dropout
        )

        self.norm2 = nn.LayerNorm(hidden_size)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * mlp_ratio, hidden_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Attention block
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]

        # MLP block
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
    def __init__(self, img_size, in_channels, patch_size, hidden_size,
                 num_layers, num_heads=8, num_classes=2, dropout=0.1):
        super().__init__()

        self.patch_size = patch_size

        # Patch embedding (linear version)
        self.fc_in = nn.Linear(in_channels * patch_size * patch_size, hidden_size)

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size).normal_(std=0.02))

        # Positional embedding (CLS + patches)
        seq_length = (img_size // patch_size) ** 2
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, seq_length + 1, hidden_size).normal_(std=0.02)
        )

        # Transformer encoder blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, mlp_ratio=4, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Final norm (ViT uses this)
        self.norm = nn.LayerNorm(hidden_size)

        # Classifier head
        self.fc_out = nn.Linear(hidden_size, num_classes)

    def forward(self, image):
        bs = image.shape[0]

        # Extract patches
        patch_seq = self.extract_patches(image)

        # Patch embedding
        patch_emb = self.fc_in(patch_seq)

        # Add CLS token
        cls = self.cls_token.expand(bs, 1, -1)
        x = torch.cat((cls, patch_emb), dim=1)

        # Add positional embeddings
        x = x + self.pos_embedding

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final norm + CLS token output
        x = self.norm(x)
        cls_out = x[:, 0]

        return self.fc_out(cls_out)

    def extract_patches(self, img):
        bs, c, h, w = img.size()
        p = self.patch_size

        patches = img.unfold(2, p, p).unfold(3, p, p)
        patches = patches.permute(0, 2, 3, 1, 4, 5)
        patches = patches.reshape(bs, -1, c * p * p)
        return patches

"""
# Test Validation
dummy_image = torch.randn(4, 3, 256, 256)  # Batch size=4, 3-channels, 256x256 image
vit = ViT(img_size=256, in_channels=3, patch_size=16, hidden_size=128, num_layers=4, num_heads=4, num_class=2)
patches = vit.extract_patches(dummy_image)
print(patches.shape)  # Expected: (4, (256//16)**2, 3*16*16) = (4, 256, 768)
"""
