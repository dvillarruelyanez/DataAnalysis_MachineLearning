#!/usr/bin/env python3

"""
Inferences from ML models for Project03
Daniel Villarruel-Yanez (2025.11.25)
"""

import os
import torch
from einops import rearrange

from .model import CNextUNetbaseline
from .train import ModelTrainer

class ModelPredictor:
    """
    """
    def __init__(self, dataset, weights_path, stats_path, device='cuda', n_input=4, n_output=1):
        self.device = device
        self.n_input = n_input
        self.n_output = n_output

        self.trainer = ModelTrainer(dataset, CNextUNetbaseline, device, n_workers=0)

        if not os.path.exists(stats_path):
            raise FileNotFoundError(f'Stats file not found at: {stats_path}')
        
        print(f'Loading stats from {stats_path}...')
        stats = torch.load(stats_path, map_location=device)
        self.trainer.mu = stats['mu']
        self.trainer.sigma = stats['sigma']

        in_channels = n_input * self.trainer.F
        out_channels = n_output * self.trainer.F
        grid = (256, 256)

        self.trainer.model_instance = self.trainer.model(
            in_channels=in_channels,
            out_channels=out_channels,
            spatial_resolution=grid,
            initial_dimension=42,
            up_down_blocks=4,
            blocks_per_stage=2,
            bottleneck_blocks=1
        ).to(device)

        if not os.path.exists(weights_path):
            raise FileNotFoundError(f'Weights file not found at: {weights_path}')

        print('Loading weights from {weights_path}...')
        state_dict = torch.load(weights_path, map_location=device)
        self.trainer.model_instance.load_state_dict(state_dict)

        self.trainer.model_instance.eval()
        print('Model loaded and ready for inference')

    def predict(self, sample_idx, split='test'):
        """
        """
        dataset = self.trainer._loader(split, self.n_input, self.n_output)

        if sample_idx >= len(dataset):
            raise IndexError(f'Index {sample_idx} out of bounds for split {split}')
        
        sample = dataset[sample_idx]

        x      = sample['input_fields'].unsqueeze(0)
        x_norm = self.trainer._preprocess(x)
        x_in   = rearrange(x_norm, "B Ti Lx Ly F -> B (Ti F) Lx Ly")
        
        with torch.no_grad():
            y_pred_norm = self.trainer.model_instance(x_in)

        y_pred = self.trainer._postprocess(y_pred_norm)

        y_true = sample['output_fields'].unsqueeze(0)
        y_true = rearrange(y_true, "B To Lx Ly F -> B (To F) Lx Ly")

        return {
            'input': x.cpu().numpy(),
            'true': y_true.cpu().numpy(),
            'prediction': y_pred.cpu().numpy()
        }