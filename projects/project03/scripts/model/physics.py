#!/usr/bin/env python3

"""
Physics-motivated loss functions for Project03
Daniel Villarruel-Yanez (2025.11.25)
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict
import torch.nn.functional as F

class PhysicsLoss(nn.Module):
    
    def __init__(self, spatial_dims: Tuple[int, int], dx: float, dt: float, derivative_scheme: str ='central_diff'):
        super().__init__()
        self.h, self.w = spatial_dims
        self.dx = float(dx)
        self.dt = float(dt)

        self.loss_keys = ['continuity', 'divergence', 'strain_consistency', 'symmetry', 'KE']

        init_val = 1.0
        self.log_vars = nn.Parameter(torch.full((len(self.loss_keys),), init_val, dtype=torch.float32))

        if derivative_scheme not in ['central_diff', 'sobel']:
            raise ValueError('Incorrect derivative scheme. Use "central_diff" or "sobel"')
        
        if derivative_scheme == 'central_diff':
            self.register_buffer('kernel_x', torch.tensor([[[[0, 0, 0],
                                                             [-0.5, 0, 0.5],
                                                             [0, 0, 0]]]], dtype=torch.float32) / self.dx)
        
            self.register_buffer('kernel_y', torch.tensor([[[[0, -0.5, 0],
                                                             [0, 0, 0],
                                                             [0, 0.5, 0]]]], dtype=torch.float32) / self.dx)
            
        else:
            self.register_buffer('kernel_x', torch.tensor([[[[-1, 0, 1],
                                                             [-2, 0, 2],
                                                             [-1, 0, 1]]]], dtype=torch.float32) / 8.0)
            
            self.register_buffer('kernel_y', torch.tensor([[[[-1, -2, -1],
                                                             [0, 0, 0],
                                                             [1, 2, 1]]]], dtype=torch.float32) / 8.0)

    def spatial_gradient(self, field):
        """
        """
        if field.dim() == 3:
            field = field.unsqueeze(1)
        elif field.dim() == 4 and field.size(1) != 1:
            b, c, h, w = field.shape
            field = field.view(b*c, 1, h, w)

        field_pad = F.pad(field, (1, 1, 1, 1), mode='circular')

        d_dx = F.conv2d(field_pad, self.kernel_x) / self.dx
        d_dy = F.conv2d(field_pad, self.kernel_y) / self.dx

        return d_dx.squeeze(1), d_dy.squeeze(1)

    def forward(self, ypred, ytrue, xprev=None):
        
        assert ypred.dim() == 4 and ytrue.dim() == 4, "ypred and ytrue must be (B, C, H, W)"
        device = ypred.device()
        dtype  = ypred.dtype

        losses: Dict[str, torch.Tensor] = {}

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

        kpred = 0.5 * torch.sum(rho * vpred, dim=(1, 2))
        ktrue = 0.5 * torch.sum(rho_true * vtrue, dim=(1, 2))

        losses['KE'] = F.mse_loss(kpred, ktrue)

        final_loss = torch.tensor(0.0, device=device, dtype=dtype)
        assert len(self.log_vars) == len(self.loss_keys), "log_vars must match loss_keys length"

        for i, key in enumerate(self.loss_keys):
            log_var = self.log_vars[i]
            precision = torch.exp(-log_var)
            comp = losses[key]
            final_loss += 0.5 * precision * comp + 0.5 * log_var

        return final_loss, losses