#!/usr/bin/env python3

"""
Physics-motivated loss functions for Project03
Daniel Villarruel-Yanez (2025.11.25)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class PhysicsLoss(nn.Module):
    
    def __init__(self, spatial_dims, dx, dt, weights=None):
        super().__init__()
        self.h, self.w = spatial_dims
        self.dx = dx
        self.dt = dt

        default_weights = {
            'mse': 1.0,
            'continuity': 0.1,
            'divergence': 0.1,
            'symmetry': 1.0,
            'KE': 0.01
        }

        self.target_weights = weights if weights else default_weights

        self.physics = ['continuity', 'divergence', 'symmetry', 'KE']

        self.weights = self.target_weights.copy()

        self.register_buffer('kernel_x', torch.tensor([[[[-1, 0, 1],
                                                         [-2, 0, 2],
                                                         [-1, 0, 1]]]], dtype=torch.float32) / 8.0)
        
        self.register_buffer('kernel_y', torch.tensor([[[[-1, -2, -1],
                                                         [0, 0, 0],
                                                         [1, 2, 1]]]], dtype=torch.float32) / 8.0)
        

    def physics_scale(self, alpha):
        """
        """
        for key in self.physics:
            if key in self.target_weights:
                self.weights[key] = self.target_weights[key] * alpha

    def spatial_gradient(self, field):
        """
        """

        field_pad = F.pad(field, (1, 1, 1, 1), mode='replicate')

        d_dx = F.conv2d(field_pad, self.kernel_x) / self.dx
        d_dy = F.conv2d(field_pad, self.kernel_y) / self.dx

        return d_dx, d_dy

    def forward(self, ypred, ytrue, xprev=None):
        losses = {}

        losses['mse'] = F.mse_loss(ypred, ytrue)

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

        kpred = 0.5 * torch.sum(rho * vpred, dim=(1, 2, 3))
        ktrue = 0.5 * torch.sum(rho_true * vtrue, dim=(1, 2, 3))

        losses['KE'] = F.mse_loss(kpred, ktrue)

        total_loss = 0
        for key, val in losses.items():
            total_loss += self.weights[key] * val

        return total_loss, losses