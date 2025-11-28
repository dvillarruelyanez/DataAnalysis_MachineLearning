#!/usr/bin/env python3

"""
Training script for Project03
Daniel Villarruel-Yanez (2025.11.25)
"""

import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from einops import rearrange
from torch.utils.data import DataLoader

from the_well.data import WellDataset

import logging
import argparse

from .model import CNextUNetbaseline
from .physics import PhysicsLoss

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class ModelTrainer:
    def __init__(self, path, model, device, n_workers, mu=None, sigma=None):
        self.path = path
        self.model = model

        if isinstance(device, str):
            self.device = torch.device('cuda' if (device == 'cuda' and torch.cuda.is_available()) else 'cpu')
        elif isinstance(device, torch.device):
            self.device = device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
        self.n_workers = int(n_workers) if n_workers is not None else 0
        self.F = 11
        self.mu = mu
        self.sigma = sigma
        self.model_instance = None

    def _loader(self, split, n_steps_input, n_steps_output):
        dataset = WellDataset(
            well_base_path = self.path,
            well_dataset_name = 'active_matter',
            well_split_name = split,
            n_steps_input = n_steps_input,
            n_steps_output = n_steps_output,
            use_normalization = False
        )

        return dataset
    
    def _setup(self, train_dataset, n_input, n_output):
        logging.info('Setting up model and normalisation')

        if self.mu == None and self.sigma == None:
            max_samples = min(30, len(train_dataset))
            sample_idx  = np.linspace(0, len(train_dataset) - 1, num=max_samples).astype(int)

            mu_sum = torch.zeros(self.F)
            sigma_sum = torch.zeros(self.F)
            count = 0

            for idx in sample_idx:
                x = train_dataset[idx]['input_fields']
                x = x.reshape(-1, self.F)

                mu_sum += x.mean(dim=0)
                sigma_sum += x.std(dim=0)
                count += 1

            self.mu = (mu_sum / count).to(self.device)
            self.sigma = (sigma_sum / count).to(self.device)
            self.sigma[self.sigma == 0] = 1.0

            logging.info('Normalisation stats calculated!')

        else:

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
    
    def _preprocess(self, x):
        if self.mu is None:
            raise RuntimeError('Call _setup first!')
        
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = x.to(self.device)
        return (x - self.mu) / self.sigma
    
    def _postprocess(self, y):
        if self.mu is None:
            raise RuntimeError('Call _setup first')
        
        n_channels = y.shape[1]
        if n_channels == self.F:        
            mu_b = self.mu.view(1, self.F, 1, 1)
            sigma_b = self.sigma.view(1, self.F, 1, 1)
        else:
            mu_b = self.mu.repeat(n_channels // self.F).view(1, n_channels, 1, 1)
            sigma_b = self.sigma.repeat(n_channels // self.F).view(1, n_channels, 1, 1)

        return (y * sigma_b) + mu_b
    
    def save_stats(self, save_dir):
        stats_path = os.path.join(save_dir, 'stats.pt')
        torch.save({
            'mu': self.mu,
            'sigma': self.sigma
        }, stats_path)
        logging.info(f'Normalisation stats saved to {stats_path}')
    
    def train_benchmark(self, batch, epochs, lr, patience, n_input, n_output):
        self._setup(n_input, n_output)
        optimizer = torch.optim.Adam(self.model_instance.parameters(), lr=lr)

        train_dataset = self._loader('train', n_input, n_output)
        valid_dataset  = self._loader('valid', n_input, n_output)

        train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True, num_workers=self.n_workers)
        val_loader = DataLoader(valid_dataset, batch_size=batch, shuffle=False, num_workers=self.n_workers)

        criterion = nn.MSELoss()

        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_count = 0

        for epoch in range(epochs):

            train_loss = 0.0
            self.model_instance.train()
            print(f'EPOCH {epoch + 1} / {epochs}')

            for batch in (bar := tqdm(train_loader)):
                x = batch['input_fields']
                xnorm = self._preprocess(x)
                xnorm = rearrange(xnorm, "B Ti Lx Ly F -> B (Ti F) Lx Ly")

                y = batch['output_fields']
                ynorm = self._preprocess(y)
                ynorm = rearrange(ynorm, "B To Lx Ly F -> B (To F) Lx Ly")

                fx = self.model_instance(xnorm)

                loss = criterion(fx, ynorm)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                bar.set_postfix(loss=loss.item())
                train_loss += loss.item()

            train_loss /= max(1, len(train_loader))
            train_losses.append(train_loss)

            self.model_instance.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in (bar := tqdm(val_loader)):
                    x = batch['input_fields']
                    x = self._preprocess(x)
                    x = rearrange(x, "B Ti Lx Ly F -> B (Ti F) Lx Ly")
                    
                    y = batch['output_fields']
                    y = self._preprocess(y)
                    y = rearrange(y, "B To Lx Ly F -> B (To F) Lx Ly")
                    
                    fx = self.model_instance(x)

                    loss = criterion(fx, y)
                    bar.set_postfix(loss=loss.item())
                    val_loss += loss.item()

            val_loss /= max(1, len(val_loader))
            val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_count = 0

                all_dir = './outputs/'
                os.makedirs(all_dir, exist_ok=True)
                save_dir = './outputs/baseline'
                os.makedirs(save_dir, exist_ok=True)

                torch.save(self.model_instance.state_dict(), os.path.join(save_dir, 'best_model.pth'))
                self.save_stats(save_dir)

                logging.info(f"Saved best model to {save_dir + 'best_baseline.pth'}")
            else:
                patience_count += 1

            if patience_count >= patience:
                print('Early stop triggered')
                break

        return train_losses, val_losses
    
    def train(self, batch, epochs, warmup_epochs, lr, patience, n_input, n_output):

        train_dataset = self._loader('train', n_input, n_output)
        valid_dataset  = self._loader('valid', n_input, n_output)

        self._setup(train_dataset, n_input, n_output)

        criterion_physics = PhysicsLoss(
            spatial_dims=(256, 256),
            dx=1.0,
            dt=0.5
        ).to(self.device)

        params = list(self.model_instance.parameters()) + list(criterion_physics.parameters())
        optimizer = torch.optim.Adam(params, lr=lr)

        train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True, num_workers=self.n_workers)
        val_loader = DataLoader(valid_dataset, batch_size=batch, shuffle=False, num_workers=self.n_workers)

        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_count = 0

        for epoch in range(epochs):

            is_warmup = epoch < warmup_epochs

            if epoch % 5 == 0:
                with torch.no_grad():
                    ws = torch.exp(-criterion_physics.log_vars)
                    print(f"Epoch {epoch} Weights [MSE, Cont, Div, Sym, KE]:\n{ws.cpu().numpy()}")

            for batch in (bar := tqdm(train_loader)):
                x = batch['input_fields']

                xprev = x[:, -1, ...].to(self.device)
                xprev = rearrange(xprev, "B Lx Ly F -> B F Lx Ly")

                x = self._preprocess(x)
                xnorm = rearrange(x, "B Ti Lx Ly F -> B (Ti F) Lx Ly")

                y = batch['output_fields']
                
                yphys = y[:, 0, ...].to(self.device)
                yphys = rearrange(yphys, "B Lx Ly F -> B F Lx Ly")

                ynorm = self._preprocess(y)
                ynorm = rearrange(ynorm, "B To Lx Ly F -> B (To F) Lx Ly")

                fx = self.model_instance(xnorm)

                if is_warmup:
                    loss = F.mse_loss(fx, ynorm)
                else:
                    loss_mse = F.mse_loss(fx, ynorm)

                    fx_phys = self._postprocess(fx)

                    loss_physics, components = criterion_physics(fx_phys, yphys, xprev)

                    loss = loss_mse + loss_physics
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model_instance.parameters(), 1.0)
                optimizer.step()

                bar.set_postfix(loss=loss.item(), phase="Warmup" if is_warmup else "Hybrid")
                train_loss += loss.item()

            train_loss /= max(1, len(train_loader))
            train_losses.append(train_loss)

            self.model_instance.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in (bar := tqdm(val_loader)):
                    x = batch['input_fields']

                    xprev = x[:, -1, ...].to(self.device)
                    xprev = rearrange(xprev, "B Lx Ly F -> B F Lx Ly")

                    x = self._preprocess(x)
                    xnorm = rearrange(x, "B Ti Lx Ly F -> B (Ti F) Lx Ly")

                    y = batch['output_fields']
                
                    yphys = y[:, 0, ...].to(self.device)
                    yphys = rearrange(yphys, "B Lx Ly F -> B F Lx Ly")

                    ynorm = self._preprocess(y)
                    ynorm = rearrange(ynorm, "B To Lx Ly F -> B (To F) Lx Ly")

                    fx = self.model_instance(xnorm)

                    if is_warmup:
                        loss = F.mse_loss(fx, ynorm)
                    else:
                        loss_mse = F.mse_loss(fx, ynorm)

                        fx_phys = self._postprocess(fx)
                        loss_physics, components = criterion_physics(fx_phys, yphys, xprev)

                        loss = loss_mse + loss_physics

                    bar.set_postfix(loss=loss.item())
                    val_loss += loss.item()

            val_loss /= max(1, len(val_loader))
            val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_count = 0

                save_dir = './outputs/'
                os.makedirs(save_dir, exist_ok=True)

                torch.save(self.model_instance.state_dict(), save_dir + 'best_model.pth')
                self.save_stats(save_dir)

                logging.info(f"Saved best model to {save_dir + 'best_model.pth'}")
            else:
                patience_count += 1

            if patience_count >= patience:
                print('Early stop triggered')
                break

        print(f'mu: {self.mu}')
        print(f'sigma: {self.sigma}')

        return train_losses, val_losses

def main():
    parser = argparse.ArgumentParser(
        prog = 'UNetConvNext baseline model trainer',
        description = 'Model trainer for the active_matter dataset of The Well'
    )

    parser.add_argument('path', type=str, help='Path to active matter dataset')
    parser.add_argument('-t', '--training', type=int, help='Type of training (1) Baseline, (2) Hybrid')
    parser.add_argument('-n', '--num', type=int, help='Number of processors')
    parser.add_argument('-d', '--device', type=str, default='cuda', help='Device to use (cuda or cpu)')

    args = parser.parse_args()
    if not os.path.isdir(args.path):
        logging.error(f'Path to active_matter dataset not found: {args.path}')
        return
    
    if args.training not in [1, 2]:
        logging.error(f'Unsupported mode: {args.training}')
    
    path = args.path
    n_workers = args.num
    device = args.device
    mode = args.training
    
    trainer = ModelTrainer(path, CNextUNetbaseline, device, n_workers)

    if mode == 1:
        train, valid = trainer.train_benchmark(batch=4, epochs=156, lr=5e-3, patience=5, n_input=4, n_output=1)
    elif mode == 2:
        train, valid = trainer.train(batch=4, epochs=156, warmup_epochs=5, lr=5e-3, patience=5, n_input=4, n_output=1)

    print('TRAINING successful')

    outfile = 'losses.dat'
    with open(outfile, 'w') as f:
        f.write('epoch train_loss valid_loss\n')
        maxlen = max(len(train), len(valid))
        for i in range(maxlen):
            t = train[i] if i < len(train) else 0
            v = valid[i] if i < len(valid) else 0
            f.write(f'{i+1} {t} {v}\n')

    print('losses.dat file located in ./outputs')

if __name__ == '__main__':
    main()