#!/usr/bin/env python3

"""
Training script for Project03 (Optimised)
Daniel Villarruel-Yanez (2025.11.30) - Refactored
"""

import os
import logging
import argparse
import numpy as np
from tqdm import tqdm
from einops import rearrange

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from the_well.data import WellDataset
from model import CNextUNetbaseline
from physics import PhysicsLoss

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

class ModelTrainer:
    def __init__(self, path, model, device: str = 'cuda', n_workers: int = None, mu=None, sigma=None):
        self.path = path
        self.model = model
        
        if device == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        if n_workers is None:
            self.n_workers = 15
        else:
            self.n_workers = int(n_workers)

        self.F = 11
        self.mu = mu
        self.sigma = sigma
        self.model_instance = None

    def _loader(self, split, n_steps_input, n_steps_output):
        return WellDataset(
            well_base_path=self.path,
            well_dataset_name='active_matter',
            well_split_name=split,
            n_steps_input=n_steps_input,
            n_steps_output=n_steps_output,
            use_normalization=False
        )
    
    def _setup(self, train_dataset, n_input, n_output):
        logging.info('Setting up model and normalisation')

        if self.mu is None or self.sigma == None:
            max_samples = min(30, len(train_dataset))
            sample_idx  = np.linspace(0, len(train_dataset) - 1, num=max_samples).astype(int)

            mu_sum = torch.zeros(self.F, dtype=torch.float32)
            sigma_sum = torch.zeros(self.F, dtype=torch.float32)
            count = 0

            for idx in sample_idx:
                item = train_dataset[idx]
                x = item['input_fields']

                if isinstance(x, np.ndarray):
                    x = torch.from_numpy(x).float()
                else:
                    x = x.float()

                x = x.reshape(-1, self.F)

                mu_sum += x.mean(dim=0)
                sigma_sum += x.std(dim=0)
                count += 1

            self.mu = (mu_sum / count).to(self.device)
            self.sigma = (sigma_sum / count).to(self.device)
            self.sigma[self.sigma == 0.0] = 1.0

            logging.info('Normalisation stats calculated on device %s', str(self.device))

        else:

            if isinstance(self.mu, np.ndarray):
                self.mu = torch.from_numpy(self.mu).float().to(self.device)
            else:
                self.mu = self.mu.to(self.device)
            if isinstance(self.sigma, np.ndarray):
                self.sigma = torch.from_numpy(self.sigma).float().to(self.device)
            else:
                self.sigma = self.sigma.to(self.device)

            logging.info('Using provided mu, sigma')

        in_channels = n_input * self.F
        out_channels = n_output * self.F
        grid = (256, 256)

        self.model_instance = self.model(
            in_channels=in_channels,
            out_channels=out_channels,
            spatial_resolution=grid,
            initial_dimension=42,
            up_down_blocks=4,
            blocks_per_stage=2,
            bottleneck_blocks=1
            ).to(self.device)
        
        self.model_instance.to(memory_format=torch.channels_last)
    
    def _preprocess(self, x):
        x = x.to(self.device, non_blocking=True).float()
        return (x - self.mu) / self.sigma
    
    def _postprocess(self, y):
        n_channels = y.shape[1]
        repeat = n_channels // self.F
        mu_b = self.mu.repeat(repeat).view(1, n_channels, 1, 1)
        sigma_b = self.sigma.repeat(repeat).view(1, n_channels, 1, 1)
        return (y * sigma_b) + mu_b
    
    def save_stats(self, save_dir): 
        torch.save({'mu': self.mu.cpu(), 'sigma': self.sigma.cpu()}, os.path.join(save_dir, 'stats.pt'))

    def train(self, batch_size, accum_steps, epochs, lr, patience, n_input, n_output, mode='baseline', warmup_epochs=0):
        
        effective_batch_size = batch_size * accum_steps
        logging.info(f"Config: Batch={batch_size}, Accum={accum_steps} (Effective={effective_batch_size}), Workers={self.n_workers}")

        train_dataset = self._loader('train', n_input, n_output)
        valid_dataset  = self._loader('valid', n_input, n_output)
        
        self._setup(train_dataset, n_input, n_output)

        criterion_physics = None
        if mode == 'hybrid':
            criterion_physics = PhysicsLoss(
                spatial_dims=(256, 256),
                dx=0.0390625,
                dt=0.25
            ).to(self.device)
            logging.info("Physics Loss enabled.")

        loader_kwargs = {
            'batch_size': batch_size,
            'num_workers': max(0, min(self.n_workers, 4)),
            'pin_memory': True,
            'persistent_workers': self.n_workers > 0,
            'prefetch_factor': 2 
        }
        train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
        val_loader   = DataLoader(valid_dataset, shuffle=False, **loader_kwargs)

        optimizer = torch.optim.Adam(self.model_instance.parameters(), lr=lr)
        scaler = torch.amp.GradScaler('cuda')
        criterion_mse = nn.MSELoss()

        history = {'train_loss': [], 'val_loss': [], 'val_phys': []}
        best_val_loss = float('inf')
        patience_count = 0
        ramp_up_epochs = 10

        for epoch in range(epochs):
            
            alpha = 0.0
            if mode == 'hybrid':
                if epoch < warmup_epochs:
                    alpha = 0.0
                else:
                    progress = (epoch - warmup_epochs) / max(1, ramp_up_epochs)
                    alpha = min(1.0, progress)
                
                if epoch == warmup_epochs and criterion_physics is not None:
                    logging.info("Warmup done. Adding Physics parameters to optimiser.")
                    optimizer.add_param_group({'params': criterion_physics.parameters(), 'lr': lr * 0.1})
                    best_val_loss = float('inf')
                    patience_count = 0

            self.model_instance.train()
            train_loss_avg = 0.0
            optimizer.zero_grad(set_to_none=True)
            
            pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{epochs}")
            for i, batch in enumerate(pbar):
                
                x_raw = batch['input_fields']
                y_raw = batch['output_fields']

                xnorm = self._preprocess(x_raw)
                ynorm = self._preprocess(y_raw)
                
                xnorm = rearrange(xnorm, "B T X Y F -> B (T F) X Y").contiguous(memory_format=torch.channels_last)
                ynorm = rearrange(ynorm, "B T X Y F -> B (T F) X Y").contiguous(memory_format=torch.channels_last)

                with torch.amp.autocast('cuda'):
                    fx = self.model_instance(xnorm)
                    loss_mse = criterion_mse(fx, ynorm)
                    loss = loss_mse

                    phys_term = 0.0
                    if mode == 'hybrid' and alpha > 0 and criterion_physics is not None:
                        fx_phys = self._postprocess(fx.float())
                        xprev = x_raw[:, -1, ...].to(self.device).float()
                        xprev = rearrange(xprev, "B X Y F -> B F X Y")
                        yphys = y_raw[:, 0, ...].to(self.device).float()
                        yphys = rearrange(yphys, "B X Y F -> B F X Y")

                        l_phys, _, _ = criterion_physics(fx_phys, yphys, xprev)
                        loss = loss_mse + (alpha * l_phys)
                        phys_term = l_phys.item()
                
                loss = loss / accum_steps
                scaler.scale(loss).backward()

                if (i + 1) % accum_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model_instance.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                train_loss_avg += loss.item() * accum_steps
                pbar.set_postfix(mse=f"{loss_mse.item():.4f}", phys=f"{phys_term:.2f}")

            train_loss_avg /= len(train_loader)
            history['train_loss'].append(train_loss_avg)

            self.model_instance.eval()
            val_loss_avg = 0.0
            val_phys_avg = 0.0
            
            with torch.no_grad():
                for batch in val_loader:
                    x_raw = batch['input_fields']
                    y_raw = batch['output_fields']
                    xnorm = self._preprocess(x_raw)
                    ynorm = self._preprocess(y_raw)
                    xnorm = rearrange(xnorm, "B T X Y F -> B (T F) X Y").contiguous(memory_format=torch.channels_last)
                    ynorm = rearrange(ynorm, "B T X Y F -> B (T F) X Y").contiguous(memory_format=torch.channels_last)

                    with torch.amp.autocast('cuda'):
                        fx = self.model_instance(xnorm)
                        loss_mse = criterion_mse(fx, ynorm)
                        loss = loss_mse

                        if mode == 'hybrid' and alpha > 0 and criterion_physics is not None:
                            fx_phys = self._postprocess(fx.float())
                            xprev = x_raw[:, -1, ...].to(self.device).float()
                            xprev = rearrange(xprev, "B X Y F -> B F X Y")
                            yphys = y_raw[:, 0, ...].to(self.device).float()
                            yphys = rearrange(yphys, "B X Y F -> B F X Y")
                            l_phys, _, err_only = criterion_physics(fx_phys, yphys, xprev)
                            
                            loss = loss_mse + (alpha * l_phys)
                            val_phys_avg += err_only.item()

                    val_loss_avg += loss.item()

            val_loss_avg /= len(val_loader)
            val_phys_avg /= len(val_loader)
            history['val_loss'].append(val_loss_avg)
            history['val_phys'].append(val_phys_avg)

            logging.info(f"Epoch {epoch+1}: Train={train_loss_avg:.5f} | Val={val_loss_avg:.5f}")

            if val_loss_avg < best_val_loss:
                best_val_loss = val_loss_avg
                patience_count = 0
                
                save_dir = f'./outputs/{mode}'
                os.makedirs(save_dir, exist_ok=True)
                
                torch.save({
                    'epoch': epoch,
                    'model': self.model_instance.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'stats': {'mu': self.mu.cpu(), 'sigma': self.sigma.cpu()}
                }, os.path.join(save_dir, 'best_model.pth'))
                
            else:
                patience_count += 1
                if patience_count >= patience:
                    logging.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        return history

def main():
    parser = argparse.ArgumentParser(description='Active Matter Model Trainer')
    parser.add_argument('path', type=str, help='Path to active matter dataset')
    parser.add_argument('-m', '--mode', type=str, choices=['baseline', 'hybrid'], required=True)
    parser.add_argument('-d', '--device', type=str, default='cuda')
    parser.add_argument('-b', '--batch', type=int, default=4, help='Physical batch size')
    parser.add_argument('-a', '--accum', type=int, default=4, help='Gradient accumulation steps')
    
    args = parser.parse_args()

    trainer = ModelTrainer(args.path, CNextUNetbaseline, args.device)
    
    warmup = 15 if args.mode == 'hybrid' else 0
    lr = 5e-3 if args.mode == 'baseline' else 1e-4 
    
    hist = trainer.train(
        batch_size=args.batch,
        accum_steps=args.accum,
        epochs=156, 
        lr=lr, 
        patience=5, 
        n_input=4, 
        n_output=1, 
        mode=args.mode,
        warmup_epochs=warmup
    )

    os.makedirs('./outputs', exist_ok=True)
    with open(f'./outputs/loss_{args.mode}.dat', 'w') as f:
        f.write('epoch train_loss val_loss val_phys_error\n')
        for i in range(len(hist['train_loss'])):
            f.write(f"{i+1} {hist['train_loss'][i]:.6f} {hist['val_loss'][i]:.6f} {hist['val_phys'][i]:.6f}\n")

if __name__ == '__main__':
    main()