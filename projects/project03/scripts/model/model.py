#!/usr/bin/env python3

"""
CNextUNet-Baseline model for Project03
Daniel Villarruel-Yanez (2025.11.25)
"""

import torch.nn as nn
from the_well.benchmark.models.unet_convnext import UNetConvNext

class CNextUNetbaseline(nn.Module):
    """
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 initial_dimension: int,
                 up_down_blocks: int,
                 blocks_per_stage: int,
                 bottleneck_blocks: int,
                 spatial_resolution: tuple,
                 spatial_dims: int = 2):
         
         super().__init__()
         
         self.model = UNetConvNext(
             dim_in=in_channels,
             dim_out=out_channels,
             n_spatial_dims=spatial_dims,
             spatial_resolution=spatial_resolution,
             stages=up_down_blocks,
             blocks_per_stage=blocks_per_stage,
             blocks_at_neck=bottleneck_blocks,
             init_features=initial_dimension
             )
         
    def forward(self, x):
        return self.model(x)