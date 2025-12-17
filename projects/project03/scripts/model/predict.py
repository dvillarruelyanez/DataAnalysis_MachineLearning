#!/usr/bin/env python3

"""
Inference script for Project03
Daniel Villarruel-Yanez (2025.11.30) - Refactored
"""

import os
import argparse
import numpy as np
import torch
from einops import rearrange
from torch.utils.data import DataLoader

from new_train import ModelTrainer
from model import CNextUNetbaseline

class ModelPredictor:
    def __init__(self, 
                 checkpoint_path: str, 
                 data_path: str, 
                 device: str = 'cuda', 
                 n_input: int = 4, 
                 n_output: int = 1):
        
        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        self.n_input = n_input
        self.n_output = n_output

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        stats = checkpoint.get('stats', None)
        if stats is None:
            raise KeyError("Checkpoint does not contain normalization stats ('mu', 'sigma').")
        
        self.trainer = ModelTrainer(
            path=data_path, 
            model=CNextUNetbaseline, 
            device=self.device, 
            mu=stats['mu'], 
            sigma=stats['sigma']
        )

        in_channels = n_input * self.trainer.F
        out_channels = n_output * self.trainer.F
        grid = (256, 256)

        self.model = CNextUNetbaseline(
            in_channels=in_channels,
            out_channels=out_channels,
            spatial_resolution=grid,
            initial_dimension=42,
            up_down_blocks=4,
            blocks_per_stage=2,
            bottleneck_blocks=1
        ).to(self.device)
        
        self.model.to(memory_format=torch.channels_last)

        if 'model' not in checkpoint:
            raise KeyError("Checkpoint does not contain model state_dict.")
            
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()
        print("Model loaded and ready for inference.")

    def get_dataset(self, split='test'):
        """Helper to get the dataset using the trainer's loader logic."""
        return self.trainer._loader(split, self.n_input, self.n_output)

    def predict(self, sample_idx: int, split: str = 'test'):
        """
        Performs inference on a single sample from the specified split.
        """
        dataset = self.get_dataset(split)

        if sample_idx >= len(dataset):
            raise IndexError(f'Index {sample_idx} out of bounds for split {split} (size: {len(dataset)})')
        
        sample = dataset[sample_idx]

        x_raw = sample['input_fields']
        if isinstance(x_raw, np.ndarray):
            x_raw = torch.from_numpy(x_raw)
        x_raw = x_raw.unsqueeze(0)

        x_norm = self.trainer._preprocess(x_raw)

        x_in = rearrange(x_norm, "B T X Y F -> B (T F) X Y")
        x_in = x_in.contiguous(memory_format=torch.channels_last)

        with torch.no_grad():
            with torch.amp.autocast(self.device):
                y_pred_norm = self.model(x_in)
        
        y_pred = self.trainer._postprocess(y_pred_norm)

        y_pred = rearrange(y_pred, "B (T F) X Y -> B T X Y F", F=self.trainer.F)
        
        y_true = sample['output_fields']
        if isinstance(y_true, np.ndarray):
            y_true = torch.from_numpy(y_true)
        y_true = y_true.unsqueeze(0) # Add Batch

        return {
            'input': x_raw.cpu().numpy(),
            'true': y_true.cpu().numpy(),
            'prediction': y_pred.cpu().numpy()
        }

def main():
    parser = argparse.ArgumentParser(description='Active Matter Model Predictor')
    parser.add_argument('checkpoint', type=str, help='Path to best_model.pth')
    parser.add_argument('data_path', type=str, help='Path to active matter dataset')
    parser.add_argument('-i', '--index', type=int, default=0, help='Sample index to predict')
    parser.add_argument('-s', '--split', type=str, default='test', help='Dataset split (train/valid/test)')
    parser.add_argument('-d', '--device', type=str, default='cuda')
    
    args = parser.parse_args()

    predictor = ModelPredictor(
        checkpoint_path=args.checkpoint,
        data_path=args.data_path,
        device=args.device
    )

    try:
        results = predictor.predict(args.index, args.split)
        
        print(f"\nPrediction successful for sample {args.index} in {args.split} split.")
        print(f"Input shape: {results['input'].shape}")
        print(f"Prediction shape: {results['prediction'].shape}")
        print(f"Ground Truth shape: {results['true'].shape}")
        
        mse = np.mean((results['prediction'] - results['true'])**2)
        print(f"MSE: {mse:.6f}")

        save_path = f'prediction_{args.split}_{args.index}.npz'
        np.savez(save_path, **results)
        print(f"Results saved to {save_path}")

    except Exception as e:
        print(f"Error during prediction: {e}")

if __name__ == '__main__':
    main()