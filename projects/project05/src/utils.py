import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
from skimage.metrics import structural_similarity as ssim_metric
from typing import List, Dict, Union, Optional, Tuple
from datetime import datetime, timedelta

# 1. HELPER FUNCTIONS & CLASSIFICATION

def classify_embryo_types(keys: List[str]) -> Dict[str, List[str]]:
    """
    Classifies embryo identifiers into biological groups based on naming conventions.

    Args:
        keys (List[str]): List of unique embryo identifiers.

    Returns:
        Dict[str, List[str]]: Dictionary mapping 'BMP', 'Nodal', 'Normal' to ID lists.
    """
    groups = {'BMP': [], 'Nodal': [], 'Normal': []}
    for key in keys:
        if 'Bmp' in key:
            groups['BMP'].append(key)
        elif 'Nd' in key:
            groups['Nodal'].append(key)
        elif 'Nr' in key:
            groups['Normal'].append(key)
    return groups

def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Converts tensor (C, H, W) to numpy (H, W) in [0, 1]."""
    img = tensor.detach().cpu().numpy()
    if img.ndim == 3: img = img[0]
    return np.clip(img, 0, 1)

def compute_gradient_loss_np(img1: np.ndarray, img2: np.ndarray) -> float:
    """Computes L1 gradient difference (sharpness metric)."""
    gx1 = img1[:, 1:] - img1[:, :-1]; gy1 = img1[1:, :] - img1[:-1, :]
    gx2 = img2[:, 1:] - img2[:, :-1]; gy2 = img2[1:, :] - img2[:-1, :]
    return np.mean(np.abs(gx1 - gx2)) + np.mean(np.abs(gy1 - gy2))

def get_time_at_frame(frame_idx: int, total_frames: int, start_h: float = 2.0, end_h: float = 16.0) -> str:
    """Calculates biological time string for a given frame index."""
    total_duration_min = (end_h - start_h) * 60
    if total_frames > 1:
        dt = total_duration_min / (total_frames - 1)
    else:
        dt = 0
    current_min = (start_h * 60) + (frame_idx * dt)
    h_disp = int(current_min // 60)
    m_disp = int(current_min % 60)
    return f"{h_disp}h {m_disp:02d}m"

def perform_autoregressive_inference(model, full_sequence, start_idx, end_idx, n_past, device):
    """
    Executes autoregressive inference over a specified window and calculates temporal bounds.
    
    The function derives the temporal resolution assuming the full sequence spans 
    from 2 hours (hpf) to 16 hours (hpf).

    Args:
        model (torch.nn.Module): The trained spatio-temporal prediction model.
        full_sequence (np.array): The complete ground-truth sequence (Time, C, H, W).
        start_idx (int): The index where prediction begins (first predicted frame).
        end_idx (int): The index where prediction ends (exclusive).
        n_past (int): Number of past frames required for the context window.
        device (torch.device): Computation device (CPU/GPU).

    Returns:
        dict: A dictionary containing:
            - 'predictions': np.array of shape (N_pred, C, H, W)
            - 'gt_future': np.array of shape (N_pred, C, H, W) (Ground Truth)
            - 't_start': float, calculated start time in hours.
            - 't_end': float, calculated end time in hours.
    """
    # 1. Temporal Resolution Calculation
    # Assumption: Sequence spans strictly from 2h to 16h
    GLOBAL_START_H = 2.0
    GLOBAL_END_H = 16.0
    total_frames = len(full_sequence)
    
    # Calculate time per frame (hours)
    dt = (GLOBAL_END_H - GLOBAL_START_H) / total_frames
    
    # Calculate specific time bounds for this prediction window
    # t = 2h + (index * dt)
    pred_start_time = GLOBAL_START_H + (start_idx * dt)
    pred_end_time = GLOBAL_START_H + (end_idx * dt)

    # 2. Data Preparation
    # Extract context: the 'n_past' frames immediately preceding start_idx
    if start_idx < n_past:
        raise ValueError(f"Start index {start_idx} is too small for context size {n_past}.")
        
    ctx = full_sequence[start_idx - n_past : start_idx]
    gt_future = full_sequence[start_idx : end_idx]
    
    # Tensor setup (Batch, Time, Channel, H, W)
    current_input = torch.from_numpy(ctx).float().unsqueeze(0).unsqueeze(2).to(device)
    
    # 3. Autoregressive Loop
    num_steps = end_idx - start_idx
    preds_ar = []
    
    model.eval()
    with torch.no_grad():
        for _ in range(num_steps):
            # Forward pass
            y = model(current_input)
            
            # Decode/Store result (assuming helper tensor_to_numpy exists)
            # If tensor_to_numpy isn't imported, use y.cpu().numpy()
            y_np = y.cpu().numpy() if not hasattr(model, 'tensor_to_numpy') else tensor_to_numpy(y) 
            preds_ar.append(y_np) 
            
            # Update Sliding Window (Remove oldest frame, append new prediction)
            current_input = torch.cat([current_input[:, 1:], y], dim=1)

    # Post-processing
    preds_ar = np.array(preds_ar)
    
    # If the model output was (Batch, 1, C, H, W), squeeze batch/time for cleaner numpy array
    # Adjust this squeeze based on your specific model output shape
    if preds_ar.ndim > 4: 
        preds_ar = np.concatenate(preds_ar, axis=0) # Shape: (Time, C, H, W)
        if preds_ar.shape[1] == 1: # Remove channel dim if redundant or squeeze batch
             preds_ar = preds_ar.squeeze(1)

    return {
        "predictions": preds_ar,
        "gt_future": gt_future,
        "t_start": pred_start_time,
        "t_end": pred_end_time
    }

# 2. VIDEO GENERATION 

def save_inspection_video(
    data_module: object, 
    embryo_key: str, 
    save_path: str, 
    fps: int = 15,
    start_h: float = 2.0,
    end_h: float = 16.0
) -> None:
    """
    Generates a 2-column MP4 [Raw | Processed] with biological timestamps.
    """
    # 1. Retrieve Data
    if embryo_key not in data_module.normalized_data or embryo_key not in data_module.raw_paths:
        print(f"Error: Data missing for {embryo_key}")
        return
        
    proc_stack = data_module.normalized_data[embryo_key]
    if proc_stack.ndim == 4: proc_stack = proc_stack.squeeze(1)
    
    raw_path = data_module.raw_paths[embryo_key]
    if not os.path.exists(raw_path): return
    raw_stack = tiff.imread(raw_path)

    # 2. Setup Video
    h, w = proc_stack.shape[1], proc_stack.shape[2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, fps, (w * 2, h))
    
    min_len = min(len(raw_stack), len(proc_stack))
    
    for i in range(min_len):
        # Raw (Left)
        raw = cv2.resize(raw_stack[i], (w, h))
        raw_norm = (raw - raw.min()) / (raw.max() - raw.min() + 1e-6)
        raw_rgb = cv2.cvtColor((raw_norm * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        
        # Processed (Right)
        proc_rgb = cv2.cvtColor((np.clip(proc_stack[i], 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        
        combined = np.hstack((raw_rgb, proc_rgb))
        
        # Labels & Time
        time_str = get_time_at_frame(i, min_len, start_h, end_h)
        font = cv2.FONT_HERSHEY_SIMPLEX
       # cv2.putText(combined, "Raw Data", (10, 20), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
       # cv2.putText(combined, "Processed", (w + 10, 20), font, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(combined, time_str, (10, h - 10), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        out.write(combined)
    out.release()
    print(f"Inspection video saved: {save_path}")


def save_prediction_video(
    gt_seq: np.ndarray,
    pred_seq: np.ndarray,
    save_path: str,
    fps: int = 10,
    start_h: float = 2.0,
    end_h: float = 16.0, # Uses full range logic or specific segment
    is_autoregressive: bool = False,
    n_past: int = 5
) -> None:
    """
    Generates 3-column MP4 [GT | Pred | Diff] with timestamps.
    
    If is_autoregressive is True, the time calculation adjusts to show 
    future time relative to the start of prediction.
    """
    if isinstance(gt_seq, torch.Tensor): gt_seq = tensor_to_numpy(gt_seq)
    if isinstance(pred_seq, torch.Tensor): pred_seq = tensor_to_numpy(pred_seq)
    if gt_seq.ndim == 4: gt_seq = gt_seq.squeeze(1)
    if pred_seq.ndim == 4: pred_seq = pred_seq.squeeze(1)

    t_steps = gt_seq.shape[0]
    h, w = gt_seq[0].squeeze().shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, fps, (w * 3, h))
    
    duration_min = (end_h - start_h) * 60
    dt = duration_min / max(1, t_steps - 1)

    for i in range(t_steps):
        # 2D
        gt_frame = gt_seq[i].squeeze()
        pred_frame = pred_seq[i].squeeze()

        gt_uint8 = (np.clip(gt_frame, 0, 1) * 255).astype(np.uint8)
        pred_uint8 = (np.clip(pred_frame, 0, 1) * 255).astype(np.uint8)
        # GT & Pred
        gt_c = cv2.cvtColor(gt_uint8, cv2.COLOR_GRAY2BGR)
        pred_c = cv2.cvtColor(pred_uint8, cv2.COLOR_GRAY2BGR)
        
        # Diff Heatmap
        diff = np.abs(gt_frame - pred_frame)
        diff_vis = (np.clip(diff * 3, 0, 1) * 255).astype(np.uint8)
        diff_c = cv2.applyColorMap(diff_vis, cv2.COLORMAP_INFERNO)
        
        combined = np.hstack((gt_c, pred_c, diff_c))
        
        # Time Label
        curr_min = (start_h * 60) + (i * dt)
        h_d, m_d = int(curr_min // 60), int(curr_min % 60)
        time_label = f"{h_d}h {m_d:02d}m"
        if is_autoregressive:
             # Indicate this is a forecast step
             time_label += f" (+{i+1})"

        font = cv2.FONT_HERSHEY_SIMPLEX
       # cv2.putText(combined, "Ground Truth", (10, 20), font, 0.5, (255, 255, 255), 1)
       # cv2.putText(combined, "Prediction", (w + 10, 20), font, 0.5, (255, 255, 255), 1)
       # cv2.putText(combined, "|Difference|", (2*w + 10, 20), font, 0.5, (0, 255, 255), 1)
        cv2.putText(combined, time_label, (10, h - 10), font, 0.5, (255, 255, 255), 1)

        out.write(combined)
    out.release()
    print(f"Prediction video saved: {save_path}")

# 3. QUANTITATIVE METRICS (AVERAGING)

def compute_autoregressive_metrics(
    model, keys, data_module, n_past, n_future, device, 
    start_h=2.0, end_h=16.0
) -> Dict:
    """
    Computes averaged degradation curves (MSE/SSIM/Grad) over future steps.
    Returns x_axis in 'Hours into Future' estimated from dataset average.
    """
    mse_list, ssim_list, grad_list = [], [], []
    
    # Calculate average dt from dataset to map steps to hours
    # Assume standard length for all keys to estimate dt
    sample_seq = data_module.normalized_data[keys[0]]
    total_duration = (end_h - start_h)
    dt_hours = total_duration / len(sample_seq) 
    
    for key in keys:
        if key not in data_module.normalized_data: continue
        seq = data_module.normalized_data[key]
        if seq.ndim == 4: seq = seq.squeeze(1)
        if len(seq) < n_past + n_future: continue

        start_idx = len(seq) // 2 
        context = torch.from_numpy(seq[start_idx:start_idx+n_past]).float().unsqueeze(0).unsqueeze(2).to(device)
        gt_future = seq[start_idx+n_past : start_idx+n_past+n_future]
        
        preds = []
        model.eval()
        with torch.no_grad():
            curr = context
            for _ in range(n_future):
                y = model(curr)
                preds.append(tensor_to_numpy(y[0,0]))
                curr = torch.cat([curr[:,1:], y], dim=1)
        
        preds = np.array(preds)
        
        # Step-wise metrics
        curr_mse, curr_ssim, curr_grad = [], [], []
        for t in range(n_future):
            gt, p = gt_future[t], preds[t]
            curr_mse.append(np.mean((gt - p)**2))
            curr_ssim.append(1 - ssim_metric(gt, p, data_range=1.0))
            curr_grad.append(compute_gradient_loss_np(gt, p))
            
        mse_list.append(curr_mse)
        ssim_list.append(curr_ssim)
        grad_list.append(curr_grad)
    # Arrays
    mse_arr = np.array(mse_list)
    ssim_arr = np.array(ssim_list)
    grad_arr = np.array(grad_list)
    # Sample Size
    n_samples = mse_arr.shape[0]

    # Time axis for plotting (Hours into future)
    time_axis = np.arange(1, n_future + 1) * dt_hours

    return {
        # Mean Values
        'mse': np.mean(mse_list, axis=0).tolist(),
        'ssim_error': np.mean(ssim_list, axis=0).tolist(),
        'grad': np.mean(grad_list, axis=0).tolist(),
        # Standard Error of the MEAN (SEM) = std / sqrt(N)
        'mse_sem': (np.std(mse_arr, axis=0) / np.sqrt(n_samples)).tolist(),
        'ssim_error_sem': (np.std(ssim_arr, axis=0) / np.sqrt(n_samples)).tolist(),
        'grad_sem': (np.std(grad_arr, axis=0) / np.sqrt(n_samples)).tolist(),
        
        'time_axis': time_axis.tolist()
    }

def compute_onestep_metrics(
    model, keys, data_module, n_past, device,
    start_h=2.0, end_h=16.0
) -> Dict:
    """
    Computes metrics over the biological lifecycle (One-Step-Ahead).
    Averages metrics at each absolute time point and calculates SEM for error bars.
    """
    # Determine maximum sequence length to initialize storage
    max_len = 0
    for k in keys:
        max_len = max(max_len, len(data_module.normalized_data[k]))

    # Initialize storage: List of lists to store raw values per time step
    # Structure: raw_mse[time_step] = [error_embryo_1, error_embryo_2, ...]
    # This preserves variance data required for SEM calculation.
    raw_mse = [[] for _ in range(max_len)]
    raw_ssim = [[] for _ in range(max_len)]
    raw_grad = [[] for _ in range(max_len)]

    for key in keys:
        seq = data_module.normalized_data[key]
        if seq.ndim == 4: seq = seq.squeeze(1)
        T = len(seq)

        # Run One-Step Inference
        # We iterate through the sequence, using ground truth context at every step.
        with torch.no_grad():
            for t in range(n_past, T):
                # Prepare input context
                ctx = seq[t-n_past : t]
                inp = torch.from_numpy(ctx).float().unsqueeze(0).unsqueeze(2).to(device)
                
                # Model prediction
                y = model(inp)
                pred = tensor_to_numpy(y[0,0])
                gt = seq[t]
                
                # Compute and store raw metrics
                # MSE
                val_mse = np.mean((gt - pred)**2)
                raw_mse[t].append(val_mse)
                
                # SSIM (1 - SSIM to represent error)
                val_ssim = 1 - ssim_metric(gt, pred, data_range=1.0)
                raw_ssim[t].append(val_ssim)
                
                # Gradient Difference
                val_grad = compute_gradient_loss_np(gt, pred)
                raw_grad[t].append(val_grad)

    # Aggregate results (Mean and SEM)
    avg_mse, sem_mse = [], []
    avg_ssim, sem_ssim = [], []
    avg_grad, sem_grad = [], []

    # Identify valid time steps (steps where at least one embryo had data)
    # Typically indices from [n_past] up to [max_len]
    valid_indices = [i for i in range(max_len) if len(raw_mse[i]) > 0]

    for t in valid_indices:
        # Convert list to array for statistical operations
        arr_mse = np.array(raw_mse[t])
        arr_ssim = np.array(raw_ssim[t])
        arr_grad = np.array(raw_grad[t])
        
        # Calculate Mean
        avg_mse.append(np.mean(arr_mse))
        avg_ssim.append(np.mean(arr_ssim))
        avg_grad.append(np.mean(arr_grad))
        
        # Calculate SEM: Standard Deviation / sqrt(N)
        # This provides the magnitude for the error bars/shading
        n_samples = len(arr_mse)
        if n_samples > 1:
            sem_mse.append(np.std(arr_mse) / np.sqrt(n_samples))
            sem_ssim.append(np.std(arr_ssim) / np.sqrt(n_samples))
            sem_grad.append(np.std(arr_grad) / np.sqrt(n_samples))
        else:
            sem_mse.append(0.0)
            sem_ssim.append(0.0)
            sem_grad.append(0.0)

    # Generate Time Axis (Biological Hours)
    # Map valid indices back to biological time
    dt_hours = (end_h - start_h) / max_len
    time_axis = [start_h + (t * dt_hours) for t in valid_indices]

    return {
        'mse': avg_mse,
        'mse_sem': sem_mse,
        
        'ssim_error': avg_ssim,
        'ssim_error_sem': sem_ssim,
        
        'grad': avg_grad,
        'grad_sem': sem_grad,

        'time_axis': time_axis
    }

# 4. PLOTTING

def plot_curves(
    metrics_by_type: Dict, 
    metric_key: str,
    y_label: str, 
    x_label: str,
    title: str,
    save_path: str
) -> None:
    """
    Generates a publication-quality plot with statistical error bands.
    
    Visualizes the mean metric trend over time. If Standard Error of the Mean (SEM) 
    data is available in the input dictionary (suffixed with '_sem'), it renders 
    a shaded region representing the confidence interval (Mean ± SEM).
    """

    # Global plot configuration for readability and publication standards
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['svg.fonttype'] = 'none'
    
    fig, ax = plt.subplots(figsize=(6, 5)) # Single column width (approx 8.9cm)
    
    # Define color palette and markers for consistency across figures
    colors = {'BMP': '#E64B35', 'Nodal': '#4DBBD5', 'Normal': '#00A087'} 
    markers = {'BMP': 'o', 'Nodal': 's', 'Normal': '^'}
    
    for group, data in metrics_by_type.items():
        # Ensure the primary metric exists for this group
        if metric_key in data and data[metric_key]:
            y_vals = np.array(data[metric_key])
            x_vals = np.array(data['time_axis'])
            
            # Subsample markers to avoid clutter in dense time series
            mark_every = max(1, len(y_vals) // 10)
            
            # Plot the Mean Line
            ax.plot(x_vals, y_vals, 
                    color=colors.get(group, 'k'),
                    label=group,
                    linewidth=1.5,
                    marker=markers.get(group, None),
                    markersize=4,
                    markevery=mark_every,
                    alpha=0.9)

            # Plot Error Bands (Shaded Area)
            # Check if statistical dispersion data (SEM) exists for this metric
            sem_key = f"{metric_key}_sem"
            
            if sem_key in data and len(data[sem_key]) == len(y_vals):
                sem_vals = np.array(data[sem_key])
                
                # Calculate upper and lower bounds
                lower_bound = y_vals - sem_vals
                upper_bound = y_vals + sem_vals
                
                # Render the shaded region
                # alpha=0.2 ensures the shade is subtle and allows overlapping groups to be seen
                ax.fill_between(
                    x_vals, 
                    lower_bound, 
                    upper_bound, 
                    color=colors.get(group, 'k'), 
                    alpha=0.2, 
                    edgecolor=None # Remove border from the shaded area for a cleaner look
                )

    # Axis formatting
    ax.set_xlabel(x_label, fontsize=9)
    ax.set_ylabel(y_label, fontsize=9)
    
    # Use scientific notation for Y-axis if values are very small (e.g., MSE)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2), useMathText=True)
    
    # Minimalist style: Remove top and right spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.grid(False) 
    
    # Legend configuration
    ax.legend(frameon=False, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, format=None)
    plt.close()
    print(f"Plot saved: {save_path}")