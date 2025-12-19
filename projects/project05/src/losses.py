import torch
import torch.nn as nn
from pytorch_msssim import ssim

def gradient_map(x: torch.Tensor):
    """
    Computes simple finite-difference image gradients along the x (width) and y (height) axes.

    Args:
        x (torch.Tensor): Input tensor of shape (B, C, H, W).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - grad_x: Gradient along the width dimension.
            - grad_y: Gradient along the height dimension.
    """
    # Horizontal gradient (difference along width: col[i+1] - col[i])
    grad_x = x[:, :, :, 1:] - x[:, :, :, :-1]
    
    # Vertical gradient (difference along height: row[i+1] - row[i])
    grad_y = x[:, :, 1:, :] - x[:, :, :-1, :]
    
    return grad_x, grad_y


def combined_loss_fn(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Calculates a combined loss for video prediction consisting of MSE, Gradient Loss, and SSIM.

    Args:
        y_pred (torch.Tensor): The predicted sequence. Expected shape (B, T, C, H, W).
        y_true (torch.Tensor): The ground truth sequence. Expected shape (B, T, C, H, W).

    Returns:
        torch.Tensor: A scalar tensor representing the calculated loss.
    """
    
    # Dimensionality Sanity Check
    # Ensure inputs are 5D: (Batch, Time, Channel, Height, Width)
    
    # If y_true is 4D (B, C, H, W), assume T=1 and add the time dimension.
    if y_true.dim() == 4:
        y_true = y_true.unsqueeze(1) 

    # If y_pred is 4D (B, C, H, W), assume T=1 and add the time dimension.
    if y_pred.dim() == 4:
        y_pred = y_pred.unsqueeze(1)

    # Unpack shapes
    B, T, C, H, W = y_pred.shape

    # Calculate Individual Losses
    
    # Mean Squared Error (Pixel-level fidelity)
    mse = torch.mean((y_pred - y_true) ** 2)

    # Gradient & SSIM Loss (Frame-wise calculation)
    grad_loss = 0.0
    ssim_loss = 0.0
    
    for t in range(T):
        # Extract the frame at time t. Slicing reduces dimension to (B, C, H, W)
        pred_t = y_pred[:, t]
        true_t = y_true[:, t]
        
        # Gradient Loss (Sharpness)
        gx_pred, gy_pred = gradient_map(pred_t)
        gx_true, gy_true = gradient_map(true_t)
        
        # L1 distance between gradients
        current_grad_loss = torch.mean(torch.abs(gx_pred - gx_true)) + \
                            torch.mean(torch.abs(gy_pred - gy_true))
        grad_loss += current_grad_loss

        # SSIM Loss (Structural Similarity)
        ssim_val = ssim(pred_t, true_t, data_range=1.0)
        ssim_loss += (1 - ssim_val)

    # Normalize by the number of time steps (if T > 0)
    if T > 0:
        grad_loss /= T
        ssim_loss /= T

    # Weighted Sum
    total_loss = 0.6 * mse + 0.2 * grad_loss + 0.2 * ssim_loss

    return total_loss