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

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
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

            self.n_workers = min(4, cpu_count)

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
        
        self.model_instance.to(memory_format=torch.channels_last)
    
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
    
    def train(self, batch, epochs, warmup_epochs, lr, patience, n_input, n_output, use_amp=True, channels_last=True, compile_model=False):

        train_dataset = self._loader('train', n_input, n_output)
        valid_dataset  = self._loader('valid', n_input, n_output)

        self._setup(train_dataset, n_input, n_output)

        criterion_physics = PhysicsLoss(
            spatial_dims=(256, 256),
            dx=0.0390625,
            dt=0.25
        ).to(self.device)

        criterion_physics.log_vars.requires_grad_(False)

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

        # optimizer = torch.optim.Adam([{'params': model_params, 'lr': lr}], lr=lr)

        # def opt_physics(optimizer, physics_params, physics_lr=None):
        #     pg = {'params': physics_params, 'lr': physics_lr if physics_lr is not None else lr, 'weight_decay': 0.0}
        #     optimizer.add_param_group(pg)

        optimizer = torch.optim.Adam([
            {'params': model_params, 'lr': lr},
            {'params': physics_params, 'lr': lr * 0.1}
        ], lr=lr)

        train_loader = DataLoader(train_dataset,
                                  batch_size=batch, 
                                  shuffle=True,
                                  num_workers=max(0, min(self.n_workers, 4)),
                                  pin_memory=False,
                                  persistent_workers=(self.n_workers > 0),
                                  prefetch_factor=1)
        
        val_loader = DataLoader(valid_dataset,
                                  batch_size=batch, 
                                  shuffle=False,
                                  num_workers=max(0, min(self.n_workers, 4)),
                                  pin_memory=False,
                                  persistent_workers=(self.n_workers > 0),
                                  prefetch_factor=1)
        
        scaler = torch.amp.GradScaler(enabled=use_amp)

        train_history = {'mse': [], 'phys': [], 'total': []}
        val_history   = {'mse': [], 'phys': [], 'total': []}

        best_val_loss = float('inf')
        patience_count = 0
        ramp_up_epochs = 10

        for epoch in range(epochs):

            is_warmup = epoch < warmup_epochs

            if is_warmup:
                alpha = 0.0
            else:
                progress = (epoch - warmup_epochs) / max(1, ramp_up_epochs)
                alpha = min(1.0, progress)

            if epoch == warmup_epochs:
                logging.info("Warmup finished. Unfreezing physics parameters and resetting early stopping baseline.")
                criterion_physics.log_vars.requires_grad_(True)
                best_val_loss = float('inf')
                patience_count = 0
                # opt_physics(optimizer, physics_params, physics_lr=lr * 0.1)
                # logging.info("Physics parameters added to optimiser")

            
            with torch.no_grad():
                try:
                    ws = torch.exp(-torch.clamp(criterion_physics.log_vars, min=-10.0, max=10.0))
                    logging.info("Epoch %d physics weights: %s", epoch + 1, np.array2string(ws.detach().cpu().numpy(), precision=4))
                except Exception:
                    pass

            self.model_instance.train()
            epoch_mse   = 0.0
            epoch_phys  = 0.0
            epoch_total = 0.0
            n_batches_done = 0

            for batch in (bar := tqdm(train_loader)):
                
                x_raw = batch['input_fields'].to(self.device, non_blocking=(self.device.type=='cuda'))
                y_raw = batch['output_fields'].to(self.device, non_blocking=(self.device.type=='cuda'))

                xnorm = self._preprocess(x_raw)
                xnorm = rearrange(xnorm, "B Ti Lx Ly F -> B (Ti F) Lx Ly")

                ynorm = self._preprocess(y_raw)
                ynorm = rearrange(ynorm, "B To Lx Ly F -> B (To F) Lx Ly")

                if channels_last:
                    xnorm = xnorm.contiguous(memory_format=torch.channels_last)
                    ynorm = ynorm.contiguous(memory_format=torch.channels_last)

                with torch.amp.autocast('cuda', enabled=use_amp):
                    fx = self.model_instance(xnorm)
                
                loss_mse = F.mse_loss(fx, ynorm)
                    
                if is_warmup:
                    loss = loss_mse
                    phys_error_display = 0.0

                else:
                    with torch.amp.autocast(enabled=False):
                        fx32 = fx.float()
                        fx_phys = self._postprocess(fx32)

                        xprev = x_raw[:, -1, ...].float().to(self.device) 
                        xprev = rearrange(xprev, "B Lx Ly F -> B F Lx Ly")

                        yphys = y_raw[:, 0, ...].float()
                        yphys = rearrange(yphys, "B Lx Ly F -> B F Lx Ly")

                        loss_physics_total, _, loss_physics_error_only = criterion_physics(fx_phys, yphys, xprev)

                        loss = loss_mse + (alpha * loss_physics_total)

                        phys_error_display = loss_physics_error_only.item()

                if not torch.isfinite(loss) or torch.isnan(loss) or not torch.isfinite(loss_mse):
                    logging.warning("Non-finite loss detected (train) - skipping batch")
                    optimizer.zero_grad(set_to_none=True)
                    continue                
                
                optimizer.zero_grad(set_to_none=True)

                backward_succeeded = False
                if use_amp:
                    try:
                        scaler.scale(loss).backward()
                        backward_succeeded = True
                    except RuntimeError as e:
                        logging.warning("GradScaler backward failed with RuntimeError: %s. Falling back to fp32 backward for this batch.", e)
                        torch.cuda.empty_cache()
                        backward_succeeded = False

                if not use_amp or not backward_succeeded:
                    with torch.amp.autocast(enabled=False):
                        fx = self.model_instance(xnorm)
                        loss_mse = F.mse_loss(fx, ynorm)

                        if is_warmup:
                            loss = loss_mse
                            phys_error_display = 0.0
                        else:
                            fx32 = fx.float()
                            fx_phys = self._postprocess(fx32)
                            xprev = x_raw[:, -1, ...].float().to(self.device)
                            xprev = rearrange(xprev, "B Lx Ly F -> B F Lx Ly")
                            yphys = y_raw[:, 0, ...].float().to(self.device)
                            yphys = rearrange(yphys, "B Lx Ly F -> B F Lx Ly")
                            loss_physics_total, _, loss_physics_error = criterion_physics(fx, yphys, xprev)
                            loss2 = loss_mse + (alpha * loss_physics_total)
                            phys_error_display = float(loss_physics_error.detach().cpu().item())

                    if not torch.isfinite(loss2) or not torch.isfinite(loss_mse):
                        logging.warning("Non-finite loss detected during fp32 fallback - skipping batch")
                        optimizer.zero_grad(set_to_none=True)
                        continue

                    loss2.backward()
                    torch.nn.utils.clip_grad_norm_(self.model_instance.parameters(), 1.0)
                    optimizer.step()
                    epoch_mse += float(loss_mse.detach().cpu().item())
                    epoch_phys += phys_error_display
                    epoch_total += float(loss2.detach().cpu().item())
                    n_batches_done += 1
                    bar.set_postfix(mse=f"{loss_mse.item():.6f}", phys=f"{phys_error_display:.6f}", alpha=f"{alpha:.2f}")
                    with torch.no_grad():
                        criterion_physics.log_vars.clamp_(-10.0, 10.0)
                    continue

                #scaler.scale(loss).backward()

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model_instance.parameters(), 1.0)

                grads_finite = True
                for p in list(self.model_instance.parameters()) + list(criterion_physics.parameters()):
                    if p.grad is not None:
                        if not torch.isfinite(p.grad).all():
                            grads_finite = False
                            break

                if not grads_finite:
                    logging.warning("Non-finite gradients detected - skipping optimizer step")
                    optimizer.zero_grad(set_to_none=True)
                    scaler.update()
                    continue

                scaler.step(optimizer)
                scaler.update()

                with torch.no_grad():
                    criterion_physics.log_vars.clamp_(-10.0, 10.0)

                epoch_mse += float(loss_mse.detach().cpu().item())
                epoch_phys += phys_error_display
                epoch_total += float(loss.detach().cpu().item())
                n_batches_done += 1

                bar.set_postfix(mse=f"{loss_mse.item():.4f}", phys=f"{phys_error_display:.4f}", alpha=f"{alpha:.2f}")

            if n_batches_done == 0:
                logging.warning("All training batches skipped in epoch %d", epoch + 1)
                n_batches_done = 1
                
            train_history['mse'].append(epoch_mse / n_batches_done)
            train_history['phys'].append(epoch_phys / n_batches_done)
            train_history['total'].append(epoch_total / n_batches_done)

            self.model_instance.eval()
            val_mse = 0.0
            val_phys = 0.0
            val_total = 0.0
            n_val_done = 0

            with torch.no_grad():
                for batch in (bar := tqdm(val_loader)):
                    x_raw = batch['input_fields'].to(self.device, non_blocking=(self.device.type=='cuda'))
                    y_raw = batch['output_fields'].to(self.device, non_blocking=(self.device.type=='cuda'))

                    xnorm = self._preprocess(x_raw)
                    xnorm = rearrange(xnorm, "B Ti Lx Ly F -> B (Ti F) Lx Ly")
                    ynorm = self._preprocess(y_raw)
                    ynorm = rearrange(ynorm, "B To Lx Ly F -> B (To F) Lx Ly")

                    if channels_last:
                        xnorm = xnorm.contiguous(memory_format=torch.channels_last)
                        ynorm = ynorm.contiguous(memory_format=torch.channels_last)

                    with torch.amp.autocast('cuda', enabled=use_amp):
                        fx = self.model_instance(xnorm)
                        loss_mse = F.mse_loss(fx, ynorm)
                        
                        if is_warmup:
                            loss = loss_mse
                            phys_error_display = 0.0

                        else:
                            with torch.amp.autocast(enabled=False):
                                fx32 = fx.float()
                                fx_phys = self._postprocess(fx32)

                                xprev = x_raw[:, -1, ...].float()
                                xprev = rearrange(xprev, "B Lx Ly F -> B F Lx Ly")

                                yphys = y_raw[:, 0, ...].float()
                                yphys = rearrange(yphys, "B Lx Ly F -> B F Lx Ly")

                                loss_physics_total, _, loss_physics_error_only = criterion_physics(fx_phys, yphys, xprev)

                                loss = loss_mse + (alpha * loss_physics_total)

                                phys_error_display = loss_physics_error_only.item()

                    if not torch.isfinite(loss) or not torch.isfinite(loss_mse):
                        logging.warning("Non-finite loss detected in validation - skipping batch")
                        continue

                    val_mse += float(loss_mse.detach().cpu().item())
                    val_phys += phys_error_display
                    val_total += float(loss.detach().cpu().item())
                    n_val_done += 1

                    bar.set_postfix(
                    mse=f"{loss_mse.item():.4f}", 
                    phys_err=f"{phys_error_display:.4f}",
                    alpha=f"{alpha:.2f}"
                    )

            if n_val_done == 0:
                logging.warning("All validation batches skipped in epoch %d", epoch + 1)
                n_val_done = 1

            avg_val_total = val_total / n_val_done

            val_history['mse'].append(val_mse / n_val_done)
            val_history['phys'].append(val_phys / n_val_done)
            val_history['total'].append(avg_val_total)

            current_metric = avg_val_total

            if current_metric < best_val_loss:
                best_val_loss = current_metric
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
                    print(f'Early stop triggered at epoch {epoch+1}')
                    break

        logging.info(f'Final mu: {self.mu}')
        logging.info(f'Final sigma: {self.sigma}')

        return train_history, val_history

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
        train, valid = trainer.train_benchmark(batch=6, epochs=156, lr=5e-3, patience=5, n_input=4, n_output=1)

        outfile = './outputs/losses_benchmark.dat'
        with open(outfile, 'w') as f:
            f.write('epoch train_loss valid_loss\n')
            maxlen = max(len(train), len(valid))
            for i in range(maxlen):
                t = train[i] if i < len(train) else 0.0
                v = valid[i] if i < len(valid) else 0.0
                f.write(f'{i+1} {t:.6f} {v:.6f}\n')
                
        logging.info('Benchmark training finished. Losses written to %s', outfile)

    elif mode == 2:
        train_hist, val_hist = trainer.train(batch=6, epochs=156, warmup_epochs=15, lr=1e-4, patience=5, n_input=4, n_output=1)

        os.makedirs('./outputs', exist_ok=True)
        outfile = './outputs/losses_hybrid.dat'
        with open(outfile, 'w') as f:
            f.write('epoch train_mse train_phys train_total valid_mse valid_phys valid_total\n')
        
            maxlen = len(train_hist['total'])
            for i in range(maxlen):
                tm = train_hist['mse'][i]
                tp = train_hist['phys'][i]
                tt = train_hist['total'][i]
                    
                vm = val_hist['mse'][i]
                vp = val_hist['phys'][i]
                vt = val_hist['total'][i]
                    
                f.write(f'{i+1} {tm:.6f} {tp:.6f} {tt:.6f} {vm:.6f} {vp:.6f} {vt:.6f}\n')

        logging.info('Training finished. Losses written to %s', outfile)

    print('TRAINING successful')

if __name__ == '__main__':
    main()