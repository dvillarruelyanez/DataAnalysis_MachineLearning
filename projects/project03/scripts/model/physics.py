#!/usr/bin/env python3

"""
Physics-motivated loss functions for Project03
Daniel Villarruel-Yanez (2025.11.25)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class PhysicsLoss(nn.Module):
    
    def __init__(self, spatial_dims, dx, dt, device='cuda'):
        super().__init__()
        self.h, self.w = spatial_dims
        self.dx = dx
        self.dt = dt
        self.device = device

        #self.loss_keys = ['MSE', 'continuity', 'divergence', 'symmetry', 'KE']
        #initial_log_vars = torch.tensor([-2.0, 3.0, 3.0, 3.0, 3.0], device=device)

        self.loss_keys = ['continuity', 'divergence', 'symmetry', 'KE']
        initial_log_vars = torch.tensor([ 3.0, 3.0, 3.0, 3.0], device=device)

        self.log_vars = nn.Parameter(initial_log_vars)

        self.register_buffer('kernel_x', torch.tensor([[[[-1, 0, 1],
                                                         [-2, 0, 2],
                                                         [-1, 0, 1]]]], dtype=torch.float32) / 8.0)
        
        self.register_buffer('kernel_y', torch.tensor([[[[-1, -2, -1],
                                                         [0, 0, 0],
                                                         [1, 2, 1]]]], dtype=torch.float32) / 8.0)

    def spatial_gradient(self, field):
        """
        """
        field = field.unsqueeze(1)

        field_pad = F.pad(field, (1, 1, 1, 1), mode='circular')

        d_dx = F.conv2d(field_pad, self.kernel_x) / self.dx
        d_dy = F.conv2d(field_pad, self.kernel_y) / self.dx

        return d_dx.squeeze(1), d_dy.squeeze(1)

    def forward(self, ypred, ytrue, xprev=None):
        losses = {}

        #losses['mse'] = F.mse_loss(ypred, ytrue)

        rho = ypred[:, 0]
        vx  = ypred[:, 1]
        vy  = ypred[:, 2]
        Dxy = ypred[:, 4]
        Dyx = ypred[:, 5]
        Exy = ypred[:, 8]
        Eyx = ypred[:, 9]

        # Incompressibility -> zero divergence in velocity field

        dvx_dx, _ = self.spatial_gradient(vx)
        _, dvy_dy = self.spatial_gradient(vy)

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

        # Kinetic energy

        rho_true = ytrue[:, 0]
        vx_true = ytrue[:, 1]
        vy_true = ytrue[:, 2]

        vpred = vx**2 + vy**2
        vtrue = vx_true**2 + vy_true**2

        kpred = 0.5 * torch.sum(rho * vpred, dim=(1, 2))
        ktrue = 0.5 * torch.sum(rho_true * vtrue, dim=(1, 2))

        losses['KE'] = F.mse_loss(kpred, ktrue)

        final_loss = 0.0
        for i, key in enumerate(self.loss_keys):
            precision = torch.exp(-self.log_vars[i])
            loss_comp = losses[key]
            final_loss += precision * loss_comp + self.log_vars[i]

        return final_loss, losses