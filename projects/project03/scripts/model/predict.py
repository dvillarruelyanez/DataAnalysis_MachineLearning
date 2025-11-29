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
    def __init__(self, dataset, weights, device='cuda', n_input=4, n_output=1):
        self.device = device
        self.n_input = n_input
        self.n_output = n_output

        self.trainer = ModelTrainer(dataset, CNextUNetbaseline, device, n_workers=0)
        
        print('Setting up model and calculating normalisation stats...')
        self.trainer._setup(n_input, n_output)

        if not os.path.exists(weights):
            raise FileNotFoundError(f'Weights file not found at: {weights}')

        print('Loading weights from {weights}...')
        state_dict = torch.load(weights, map_location=device)
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
        y_true = rearrange(y_true, "B Ti Lx Ly F -> B (Ti F) Lx Ly")

        return {
            'input': x.cpu().numpy(),
            'true': y_true.cpu().numpy(),
            'prediction': y_pred.cpu().numpy()
        }