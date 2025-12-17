#!/usr/bin/env python3

"""
Physics-motivated loss functions for Project03
Daniel Villarruel-Yanez (2025.11.25)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple, Dict

class PhysicsLoss(nn.Module):
    
    def __init__(self,
                 spatial_dims: Tuple[int, int],
                 dx: float,
                 dt: float, 
                 derivative_scheme: str ='sobel',
                 smooth_before: bool = True,
                 logvar_clamp: Tuple[float, float] = (-10., 10.)):
        
        super().__init__()
        self.h, self.w = spatial_dims
        self.dx = float(dx)
        self.dt = float(dt)

        self.loss_keys = ['continuity', 'divergence', 'strain_consistency', 'symmetry', 'KE']

        init_val = 5.0
        self.log_vars = nn.Parameter(torch.full((len(self.loss_keys),), init_val, dtype=torch.float32))

        if derivative_scheme not in ['central_diff', 'sobel']:
            raise ValueError('Incorrect derivative scheme. Use "central_diff" or "sobel"')
        
        if derivative_scheme == 'central_diff':
            kx = torch.tensor([[[[0, 0, 0], [-0.5, 0, 0.5], [0, 0, 0]]]], dtype=torch.float32) / (2.0 * self.dx)
            ky = torch.tensor([[[[0, -0.5, 0], [0, 0, 0], [0, 0.5, 0]]]], dtype=torch.float32) / (2.0 * self.dx)
            
        else:
            kx = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float32) / (8.0 * self.dx)
            ky = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=torch.float32) / (8.0 * self.dx)
            
        self.register_buffer('kernel_x', kx)
        self.register_buffer('kernel_y', ky)

        self.smooth_before = bool(smooth_before)
        self.logvar_clamp_min = float(logvar_clamp[0])
        self.logvar_clamp_max = float(logvar_clamp[1])

    def spatial_gradient(self, field):
        """
        """

        if field.dim() == 3:
            field = field.unsqueeze(1)

        if self.smooth_before:
            field = F.avg_pool2d(field, kernel_size=3, stride=1, padding=1)

        field_pad = F.pad(field, (1, 1, 1, 1), mode='circular')

        kx = self.kernel_x.to(field.dtype).to(field.device)
        ky = self.kernel_y.to(field.dtype).to(field.device)

        channels = field.shape[1]
        kx_rep = kx.repeat(channels, 1, 1, 1)
        ky_rep = ky.repeat(channels, 1, 1, 1)

        d_dx = F.conv2d(field_pad, kx_rep, groups=channels)
        d_dy = F.conv2d(field_pad, ky_rep, groups=channels)

        return d_dx.squeeze(1), d_dy.squeeze(1)

    def forward(self, ypred, ytrue, xprev=None):

        ypred = ypred.float()
        ytrue = ytrue.float()
        if xprev is not None:
            xprev = xprev.float()

        log_vars = torch.clamp(self.log_vars, min=self.logvar_clamp_min, max=self.logvar_clamp_max)

        losses: Dict[str, torch.Tensor] = {}
        device = ypred.device

        rho = ypred[:, 0]
        vx  = ypred[:, 1]
        vy  = ypred[:, 2]
        Dxy = ypred[:, 4]
        Dyx = ypred[:, 5]
        Exy = ypred[:, 8]
        Eyx = ypred[:, 9]

        # Incompressibility -> zero divergence in velocity field

        dvx_dx, dvx_dy = self.spatial_gradient(vx)
        dvy_dx, dvy_dy = self.spatial_gradient(vy)

        div = dvx_dx + dvy_dy

        losses['divergence'] = torch.mean(div**2)

        # Mass conservation -> continuity equation

        if xprev is not None:
            
            rho_prev = xprev[:, 0]

            drho_dt = (rho - rho_prev) / self.dt

            flux_x = rho * vx
            flux_y = rho * vy

            dfx_dx, _ = self.spatial_gradient(flux_x)
            _, dfy_dy = self.spatial_gradient(flux_y)

            cont_res = drho_dt + (dfx_dx + dfy_dy)

            losses['continuity'] = torch.mean(cont_res**2)

        else:
            losses['continuity'] = torch.tensor(0.0, device=ypred.device)

        # Tensor symmetry

        Dsym = Dxy - Dyx
        Esym = Exy - Eyx

        losses['symmetry'] = torch.mean(Dsym**2) + torch.mean(Esym**2)

        # Strain-velocity consistency

        Dxy_from_v = 0.5 * (dvx_dy + dvy_dx)
        
        strain_consistency = torch.mean((Dxy - Dxy_from_v)**2) + torch.mean((Dyx - Dxy_from_v)**2)
        losses['strain_consistency'] = strain_consistency

        # Kinetic energy

        rho_true = ytrue[:, 0]
        vx_true = ytrue[:, 1]
        vy_true = ytrue[:, 2]

        vpred = vx**2 + vy**2
        vtrue = vx_true**2 + vy_true**2

        kpred = 0.5 * torch.mean(rho * vpred, dim=(1, 2))
        ktrue = 0.5 * torch.mean(rho_true * vtrue, dim=(1, 2))

        losses['KE'] = F.mse_loss(kpred, ktrue)

        final_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        pure_weighted_error = torch.tensor(0.0, device=device, dtype=torch.float32)

        for i, key in enumerate(self.loss_keys):
            log_var = self.log_vars[i]
            precision = torch.exp(-log_var)
            comp = losses[key]

            term_error = 0.5 * precision * comp
            term_reg   = 0.5 * log_var

            final_loss += (term_error + term_reg)
            pure_weighted_error += term_error

        return final_loss, losses, pure_weighted_error