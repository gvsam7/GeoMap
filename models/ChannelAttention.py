"""
Author: Georgios Voulgaris
Date: 27/12/2024
Description: This Channel Attention module balances efficiency, flexibility, and robustness while aligning with the
             global feature extraction needs of the Multi-Temporal and Multi-Spectral dataset.
             It combines:
             1. Global Context Extraction:
                Both average pooling and max pooling are used to capture complementary information:
                 - Avg pooling emphasizes the global distribution of features.
                 - Max pooling highlights the most dominant features.
                This dual-path approach ensures the attention mechanism aligns well with multi-spectral data where key
                features (e.g., spectral signatures of cement plants) might be subtle or dominant.
             2. Efficient Shared Layers:
                 - Using a shared MLP for both pooling pathways minimizes redundancy, which is particularly important
                   for Landsat-8 data with large feature maps and intermediate fusion (where multiple feature maps are
                   concatenated).
             3. Skip Connection for Robustness:
                 - The skip connection ensures the raw features aren't completely suppressed, which is helpful for
                   datasets where spectral or temporal nuances in the original data are critical.
                 - Particularly useful for intermediate fusion since concatenated features may include noise or
                   irrelevant details.
             4. Parameter-Free Pooling Combination:
                 - Summing the outputs of the pooling pathways (rather than using learnable weights) ensures simplicity
                   and avoids overfitting, especially given the relatively balanced contribution of global vs. local
                   features in remote sensing datasets.
             5. Reduction Ratio of 2:
                - Keeps the module lightweight, which is essential for handling the multiple channels (multi-spectral +
                  temporal) in Landsat-8 bands.
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=2):
        super(ChannelAttention, self).__init__()
        reduced_channels = in_channels // reduction_ratio

        # Shared MLP for efficiency
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, 1, bias=False)
        )

        # Parameter-free combination of pooling outputs
        self.sigmoid = nn.Sigmoid()

        # Skip connection for robustness
        self.use_skip_connection = True  # Optional flag

    def forward(self, x):
        # Global context modeling
        avg_out = self.shared_mlp(nn.AdaptiveAvgPool2d(1)(x))
        max_out = self.shared_mlp(nn.AdaptiveMaxPool2d(1)(x))
        attention = self.sigmoid(avg_out + max_out)

        # Apply attention and skip connection
        if self.use_skip_connection:
            return x + x * attention
        else:
            return x * attention
