#!/usr/bin/env python3

"""
Training script for Project03 (corrected + optimised for GPU)
Daniel Villarruel-Yanez (2025.11.30) - updated
"""

import os
import logging
import argparse
import numpy as np

from tqdm import tqdm
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from the_well.data import WellDataset

from model import CNextUNetbaseline
from physics import PhysicsLoss

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

torch.backends.cudnn.benchmark = True

class ModelTrainer:
    def __init__(self, path, model, device: str = 'cuda', n_workers: int = None, mu=None, sigma=None):
        self.path = path
        self.model = model

        if isinstance(device, str):
            if device == 'cuda' and torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')

        elif isinstance(device, torch.device):
            self.device = device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
        if n_workers is None:
            try:
                cpu_count = os.cpu_count() or 4
            except Exception:
                cpu_count = 4
            self.n_workers = max(1, min(15, cpu_count - 1))

        else:
            self.n_workers = int(n_workers)

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
    
    def _preprocess(self, x):
        if self.mu is None or self.sigma is None:
            raise RuntimeError('Call _setup first!')
        
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        else:
            x = x.float()

        x = x.to(self.device, non_blocking=(self.device.type == 'cuda'))
        return (x - self.mu) / self.sigma
    
    def _postprocess(self, y):
        if self.mu is None or self.sigma is None:
            raise RuntimeError('Call _setup first')
        
        n_channels = y.shape[1]
        if n_channels == self.F:
            mu_b = self.mu.view(1, self.F, 1, 1)
            sigma_b = self.sigma.view(1, self.F, 1, 1)
        else:
            repeat = n_channels // self.F
            mu_b = self.mu.repeat(repeat).view(1, n_channels, 1, 1)
            sigma_b = self.sigma.repeat(repeat).view(1, n_channels, 1, 1)

        mu_b = mu_b.to(y.device, dtype=y.dtype)
        sigma_b = sigma_b.to(y.device, dtype=y.dtype)

        return (y * sigma_b) + mu_b
    
    def save_stats(self, save_dir): 
        stats_path = os.path.join(save_dir, 'stats.pt')
        torch.save({
            'mu': self.mu.detach().cpu(),
            'sigma': self.sigma.detach().cpu()
        }, stats_path)
        logging.info(f'Normalisation stats saved to {stats_path}')
    
    def train_benchmark(self, batch, epochs, lr, patience, n_input, n_output):

        train_dataset = self._loader('train', n_input, n_output)
        valid_dataset  = self._loader('valid', n_input, n_output)

        self._setup(train_dataset, n_input, n_output)
        optimizer = torch.optim.Adam(self.model_instance.parameters(), lr=lr)

        train_loader = DataLoader(train_dataset,
                                  batch_size=batch,
                                  shuffle=True,
                                  num_workers=self.n_workers,
                                  pin_memory=(self.device.type == 'cuda'),
                                  persistent_workers=(self.n_workers > 0),
                                  prefetch_factor=2)
        
        val_loader = DataLoader(valid_dataset,
                                  batch_size=batch,
                                  shuffle=True,
                                  num_workers=self.n_workers,
                                  pin_memory=(self.device.type == 'cuda'),
                                  persistent_workers=(self.n_workers > 0),
                                  prefetch_factor=2)
        
        criterion = nn.MSELoss()

        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_count = 0

        for epoch in range(epochs):

            self.model_instance.train()

            train_loss = 0.0

            logging.info(f'Starting epoch {epoch+1}/{epochs} (benchmark)')

            for batch in (bar := tqdm(train_loader)):

                x_raw = batch['input_fields']
                y_raw = batch['output_fields']

                xnorm = self._preprocess(x_raw)
                xnorm = rearrange(xnorm, "B Ti Lx Ly F -> B (Ti F) Lx Ly")

                ynorm = self._preprocess(y_raw)
                ynorm = rearrange(ynorm, "B To Lx Ly F -> B (To F) Lx Ly")

                fx = self.model_instance(xnorm)

                loss = criterion(fx, ynorm)
                
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_instance.parameters(), 1.0)
                optimizer.step()

                bar.set_postfix(loss=loss.item())

                train_loss += float(loss.detach().cpu().item())

            train_loss /= max(1, len(train_loader))
            train_losses.append(train_loss)

            self.model_instance.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in (bar := tqdm(val_loader)):
                    x_raw = batch['input_fields']
                    y_raw = batch['output_fields']

                    xnorm = self._preprocess(x_raw)
                    xnorm = rearrange(xnorm, "B Ti Lx Ly F -> B (Ti F) Lx Ly")

                    ynorm = self._preprocess(y_raw)
                    ynorm = rearrange(ynorm, "B To Lx Ly F -> B (To F) Lx Ly")
                    
                    fx = self.model_instance(xnorm)

                    loss = criterion(fx, ynorm)
                    bar.set_postfix(loss=loss.item())
                    val_loss += loss.detach().cpu().item()

            val_loss /= max(1, len(val_loader))
            val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_count = 0

                all_dir = './outputs/'
                os.makedirs(all_dir, exist_ok=True)
                save_dir = './outputs/baseline'
                os.makedirs(save_dir, exist_ok=True)

                torch.save(self.model_instance.state_dict(), os.path.join(save_dir, 'best_baseline.pth'))
                self.save_stats(save_dir)

                logging.info(f"Saved best model to {save_dir + 'best_baseline.pth'}")
            else:
                patience_count += 1

            if patience_count >= patience:
                print('Early stop triggered')
                break

        return train_losses, val_losses
    
    def train(self, batch, epochs, warmup_epochs, lr, patience, n_input, n_output, use_amp=True, channels_last=False, compile_model=False):

        train_dataset = self._loader('train', n_input, n_output)
        valid_dataset  = self._loader('valid', n_input, n_output)

        self._setup(train_dataset, n_input, n_output)

        criterion_physics = PhysicsLoss(
            spatial_dims=(256, 256),
            dx=1.0,
            dt=0.5
        ).to(self.device)

        self.model_instance.to(self.device)

        if channels_last:
            self.model_instance.to(memory_format=torch.channels_last)

        if compile_model:
            try:
                self.model_instance = torch.compile(self.model_instance)
                logging.info('Model compiles with torch.compile')
            
            except Exception as e:
                logging.warning(f'torch.compile failed: {e}')

        model_params = list(self.model_instance.parameters())
        physics_params = list(criterion_physics.parameters())

        optimizer = torch.optim.Adam([{'params': model_params, 'lr': lr}], lr=lr)

        def opt_physics(optimizer, physics_params, physics_lr=None):
            pg = {'params': physics_params, 'lr': physics_lr if physics_lr is not None else lr, 'weight_decay': 0.0}
            optimizer.add_param_group(pg)

        train_loader = DataLoader(train_dataset,
                                  batch_size=batch, 
                                  shuffle=True,
                                  num_workers=self.n_workers,
                                  pin_memory=(self.device.type == 'cuda'),
                                  persistent_workers=(self.n_workers > 0),
                                  prefetch_factor=2)
        
        val_loader = DataLoader(valid_dataset,
                                  batch_size=batch, 
                                  shuffle=False,
                                  num_workers=self.n_workers,
                                  pin_memory=(self.device.type == 'cuda'),
                                  persistent_workers=(self.n_workers > 0),
                                  prefetch_factor=2)
        
        scaler = torch.amp.GradScaler(enabled=use_amp and (self.device.type == 'cuda'))

        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_count = 0

        for epoch in range(epochs):

            is_warmup = epoch < warmup_epochs

            if epoch == warmup_epochs:
                opt_physics(optimizer, physics_params, physics_lr=lr * 0.1)
                logging.info("Physics parameters added to optimiser")

            
            with torch.no_grad():
                try:
                    ws = torch.exp(-criterion_physics.log_vars)
                    logging.info("Epoch %d physics weights: %s", epoch + 1, np.array2string(ws.detach().cpu().numpy(), precision=4))
                except Exception:
                    pass

            self.model_instance.train()
            train_loss = 0.0

            for batch in (bar := tqdm(train_loader)):
                
                x_raw = batch['input_fields']
                y_raw = batch['output_fields']

                if isinstance(x_raw, np.ndarray):
                    xprev = torch.from_numpy(x_raw[:, -1, ...]).float().to(self.device, non_blocking=(self.device.type == 'cuda'))
                else:
                    xprev = x_raw[:, -1, ...].float().to(self.device, non_blocking=(self.device.type == 'cuda'))

                xprev = rearrange(xprev, "B Lx Ly F -> B F Lx Ly")

                if isinstance(y_raw, np.ndarray):
                    yphys = torch.from_numpy(y_raw[:, 0, ...]).float().to(self.device, non_blocking=(self.device.type == 'cuda'))
                else:
                    yphys = y_raw[:, 0, ...].float().to(self.device, non_blocking=(self.device.type == 'cuda'))

                yphys = rearrange(yphys, "B Lx Ly F -> B F Lx Ly")

                xnorm = self._preprocess(x_raw)      # already moved to device
                xnorm = rearrange(xnorm, "B Ti Lx Ly F -> B (Ti F) Lx Ly")
                ynorm = self._preprocess(y_raw)
                ynorm = rearrange(ynorm, "B To Lx Ly F -> B (To F) Lx Ly")

                if channels_last:
                    xnorm = xnorm.contiguous(memory_format=torch.channels_last)
                    ynorm = ynorm.contiguous(memory_format=torch.channels_last)

                with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                    
                    fx = self.model_instance(xnorm)
                    
                    if is_warmup:
                        loss = F.mse_loss(fx, ynorm)
                    else:
                        loss_mse = F.mse_loss(fx, ynorm)
                        
                        fx_phys = self._postprocess(fx)

                        loss_physics, components = criterion_physics(fx_phys, yphys, xprev)

                        loss = loss_mse + loss_physics
                
                optimizer.zero_grad(set_to_none=True)

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model_instance.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model_instance.parameters(), 1.0)
                    optimizer.step()

                bar.set_postfix(loss=loss.item(), phase="Warmup" if is_warmup else "Hybrid")
                
                train_loss += float(loss.detach().cpu().item())

            train_loss /= max(1, len(train_loader))
            train_losses.append(train_loss)
            logging.info(f"Epoch {epoch+1} train loss: {train_loss:.6f}")

            self.model_instance.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in (bar := tqdm(val_loader)):
                    x_raw = batch['input_fields']
                    y_raw = batch['output_fields']

                    if isinstance(x_raw, np.ndarray):
                        xprev = torch.from_numpy(x_raw[:, -1, ...]).float().to(self.device, non_blocking=(self.device.type == 'cuda'))
                    else:
                        xprev = x_raw[:, -1, ...].float().to(self.device, non_blocking=(self.device.type == 'cuda'))

                    xprev = rearrange(xprev, "B Lx Ly F -> B F Lx Ly")

                    if isinstance(y_raw, np.ndarray):
                        yphys = torch.from_numpy(y_raw[:, 0, ...]).float().to(self.device, non_blocking=(self.device.type == 'cuda'))
                    else:
                        yphys = y_raw[:, 0, ...].float().to(self.device, non_blocking=(self.device.type == 'cuda'))

                    yphys = rearrange(yphys, "B Lx Ly F -> B F Lx Ly")

                    xnorm = self._preprocess(x_raw)      # already moved to device
                    xnorm = rearrange(xnorm, "B Ti Lx Ly F -> B (Ti F) Lx Ly")
                    ynorm = self._preprocess(y_raw)
                    ynorm = rearrange(ynorm, "B To Lx Ly F -> B (To F) Lx Ly")

                    if channels_last:
                        xnorm = xnorm.contiguous(memory_format=torch.channels_last)
                        ynorm = ynorm.contiguous(memory_format=torch.channels_last)
                        
                    if is_warmup:
                        loss = F.mse_loss(self.model_instance(xnorm), ynorm)
                    else:
                        fx = self.model_instance(xnorm)
                        
                        loss_mse = F.mse_loss(fx, ynorm)    
                        fx_phys = self._postprocess(fx)
                        loss_physics, components = criterion_physics(fx_phys, yphys, xprev)

                        loss = loss_mse + loss_physics

                    bar.set_postfix(loss=loss.item())
                    val_loss += float(loss.detach().cpu().item())

            val_loss /= max(1, len(val_loader))
            val_losses.append(val_loss)
            logging.info(f"Epoch {epoch+1} val loss: {val_loss:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_count = 0
                save_dir = './outputs/'
                os.makedirs(save_dir, exist_ok=True)
                ckpt_path = os.path.join(save_dir, f'best_model_epoch{epoch+1}.pth')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state': self.model_instance.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'mu': self.mu.detach().cpu(),
                    'sigma': self.sigma.detach().cpu()
                }, ckpt_path)
                self.save_stats(save_dir)
                logging.info(f"Saved best model to {ckpt_path}")
            else:
                patience_count += 1

            if patience_count >= patience:
                print('Early stop triggered')
                break

        logging.info(f'Final mu: {self.mu}')
        logging.info(f'Final sigma: {self.sigma}')

        return train_losses, val_losses

def main():
    parser = argparse.ArgumentParser(
        prog = 'Project03 Model Trainer (benchmark + hybrid)',
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
        train, valid = trainer.train(batch=4, epochs=156, warmup_epochs=15, lr=1.5e-4, patience=5, n_input=4, n_output=1)

    print('TRAINING successful')

    os.makedirs('./outputs', exist_ok=True)
    outfile = './outputs/losses.dat'
    with open(outfile, 'w') as f:
        f.write('epoch train_loss valid_loss\n')
        maxlen = max(len(train), len(valid))
        for i in range(maxlen):
            t = train[i] if i < len(train) else 0.0
            v = valid[i] if i < len(valid) else 0.0
            f.write(f'{i+1} {t} {v}\n')

    logging.info('Training finished. Losses written to %s', outfile)

if __name__ == '__main__':
    main()