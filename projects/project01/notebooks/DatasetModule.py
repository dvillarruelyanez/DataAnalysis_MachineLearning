import torch
from torch.utils.data import Dataset
from the_well.data import WellDataset
from einops import rearrange
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec


class MyDataset(Dataset):
    def __init__(self, mysplit, xlen=4, stride = 1, base_path = "./datasets", ylen=1):
        self.raw = WellDataset(
            well_base_path=f"{base_path}/datasets",
            well_dataset_name="active_matter",
            well_split_name=mysplit,
            n_steps_input=xlen,
            n_steps_output=ylen,
            min_dt_stride=stride,
            use_normalization=True,
        )
    
    def __len__(self):
        return len(self.raw)

    def __getitem__(self, idx):
        D = self.raw[idx]
        x = D["input_fields"]
        try:
            y = D["output_fields"]
        except:
            return x
        else:
            return x, y


def plot_datapoint(x, y):
    field_names = ['Concentration', 'Velocity_x', 'Velocity_y', 'D_xx', 'D_xy', 'D_yx', 'D_yy', 'E_xx', 'E_xy', 'E_yx', 'E_yy']
    x = rearrange(x, "T Lx Ly F -> F T Lx Ly")
    y = rearrange(y, "T Lx Ly F -> F T Lx Ly")


    len_x = x.shape[1]
    len_y = y.shape[1]
    FF = x.shape[0]

    nrows, ncols = 11, 7

    fig = plt.figure(figsize=3*plt.figaspect(11/6), constrained_layout=True)
    outer = gridspec.GridSpec(nrows, 1, figure=fig, height_ratios=[1.1,1,1,1,1,1,1,1,1,1,1])

    # fig, axs = plt.subplots(FF, 1, figsize=((len_x + len_y + len_pred) * 1.3, FF * 1.2))
    
    for field in range(FF):
        vmin_x = np.nanmin(x[field])
        vmax_x = np.nanmax(x[field])
        vmin_y = np.nanmin(y[field])
        vmax_y = np.nanmax(y[field])
        vmin = min(vmin_x, vmin_y)
        vmax = max(vmax_x, vmax_y)

        inner = gridspec.GridSpecFromSubplotSpec(1, 6, subplot_spec=outer[field], wspace=0.1, hspace=0, \
            width_ratios=[1,1,1,1,1, 0.05])


        for t in range(len_x):
            ax = fig.add_subplot(inner[0, t])
            ax.imshow(x[field, t], cmap="viridis", interpolation="none", vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if field == 0:
                ax.set_title(f"$t_{t+1}$​")
            if t == 0:
                ax.set_ylabel(f"{field_names[field]}")

        for t in range(len_y):
            ax = fig.add_subplot(inner[0, len_x + t])
            imy = ax.imshow(y[field, t], cmap="viridis", interpolation="none", vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if field == 0:
                ax.set_title(f"$tp_{len_x+t+1}$​")
        
        # --- Colorbar for error ------------------
        cax7 = fig.add_subplot(inner[0, 5])
        fig.colorbar(imy, cax=cax7, format="%.2f")

def vrmse(target, pred, eps=1e-7, field_wise=False):
    
    # Mean squared error ⟨|u - v|²⟩
    mse = torch.mean((pred - target)**2)

    # Variance of target: ⟨|u - ū|²⟩
    mean_target = torch.mean(target)
    var = torch.mean((target - mean_target)**2)

    # VRMSE = sqrt(mse / (var + eps))
    VRMSE = torch.sqrt(mse / (var + eps))

    if field_wise:
        # Mean squared error ⟨|u - v|²⟩
        msef = torch.mean((pred - target)**2, dim=(0, 1, 2, 3))

        # Variance of target: ⟨|u - ū|²⟩
        mean_targetf = torch.mean(target, dim=(0, 1, 2, 3))
        mean_targetf = mean_targetf.view(1, 1, 1, 1, 11)
        varf = torch.mean((target - mean_targetf)**2, dim=(0, 1, 2, 3))

        VRMSEf = torch.sqrt(msef / (varf + eps))

        # VRMSE = sqrt(mse / (var + eps))
        return VRMSE, VRMSEf
    return VRMSE

def plot_pred_datapoint(x, y, pred):
    field_names = ['Concentration', 'Velocity_x', 'Velocity_y', 'D_xx', 'D_xy', 'D_yx', 'D_yy', 'E_xx', 'E_xy', 'E_yx', 'E_yy']
    x = rearrange(x, "T Lx Ly F -> F T Lx Ly")
    y = rearrange(y, "T Lx Ly F -> F T Lx Ly")
    pred  = rearrange(pred, "T Lx Ly F -> F T Lx Ly")

    error = torch.abs(y - pred)

    len_x = x.shape[1]
    len_y = y.shape[1]
    len_pred = pred.shape[1]
    FF = x.shape[0]

    nrows, ncols = 11, 7

    fig = plt.figure(figsize=(15, 30), constrained_layout=True)
    outer = gridspec.GridSpec(nrows, 1, figure=fig)

    # fig, axs = plt.subplots(FF, 1, figsize=((len_x + len_y + len_pred) * 1.3, FF * 1.2))
    
    for field in range(FF):
        vmin_x = np.nanmin(x[field])
        vmax_x = np.nanmax(x[field])
        vmin_y = np.nanmin(y[field])
        vmax_y = np.nanmax(y[field])
        vmin_pred = np.nanmin(pred[field])
        vmax_pred = np.nanmax(pred[field])
        vmin = min(vmin_x, vmin_y, vmin_pred)
        vmax = max(vmax_x, vmax_y, vmax_pred)

        inner = gridspec.GridSpecFromSubplotSpec(1, 9, subplot_spec=outer[field], wspace=0.1, hspace=0, \
            width_ratios=[1,1,1,1,1,1, 0.05, 1, 0.05])


        for t in range(len_x):
            ax = fig.add_subplot(inner[0, t])
            ax.imshow(x[field, t], cmap="viridis", interpolation="none", vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if field == 0:
                ax.set_title(f"$t_{t+1}$​")
            if t == 0:
                ax.set_ylabel(f"{field_names[field]}")

        for t in range(len_y):
            ax = fig.add_subplot(inner[0, len_x + t])
            ax.imshow(y[field, t], cmap="viridis", interpolation="none", vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if field == 0:
                ax.set_title(f"$tp_{t+len_x+1}$​")

        for t in range(len_pred):
            ax = fig.add_subplot(inner[0, len_x + len_y + t])
            im = ax.imshow(pred[field, t], cmap="viridis", interpolation="none", vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if field == 0:
                ax.set_title(f"$pred_{t+len_x+1}$​")

        # --- Colorbar for first 6 images ------------------
        # cax1 = fig.add_subplot(inner[0, 6])
        # fig.colorbar(im, cax=cax1, format="%.2f")

        for t in range(len_pred):
            ax = fig.add_subplot(inner[0, len_x + len_y + 2 + t])
            # im7 = ax.imshow(error[field, t], cmap="magma", interpolation="none")
            im7 = ax.imshow(error[field, t], cmap="viridis", interpolation="none", vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if field == 0:
                ax.set_title(r"$E_{abs}$, VRMSE:"+ str(float(vrmse(pred, y)))[:5])
        
        # --- Colorbar for error ------------------
        cax7 = fig.add_subplot(inner[0, 8])
        fig.colorbar(im7, cax=cax7, format="%.2f")

def plot_rollout(x, y, error):
    ts = []
    field_names = ['Concentration', 'Velocity_x', 'Velocity_y']
    field_names2 = ['Pred Concentration', 'Pred Velocity_x', 'Pred Velocity_y']
    x = rearrange(x, "T Lx Ly F -> F T Lx Ly")
    y = rearrange(y, "T Lx Ly F -> F T Lx Ly")


    len_x = x.shape[1]
    len_y = y.shape[1]
    FF = x.shape[0]

    nrows, ncols = 6, 5

    fig = plt.figure(figsize=2*plt.figaspect(6/4), constrained_layout=True)
    outer = gridspec.GridSpec(nrows, 1, figure=fig, height_ratios=[1.1,1,1,1,1,1])

    x_ys=[0,2,4]
    for field in range(FF):
        vmin_x = np.nanmin(x[field])
        vmax_x = np.nanmax(x[field])
        vmin_y = np.nanmin(y[field])
        vmax_y = np.nanmax(y[field])
        vmin = min(vmin_x, vmin_y)
        vmax = max(vmax_x, vmax_y)

        inner = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=outer[x_ys[field]], wspace=0.1, hspace=0, \
            width_ratios=[1,1,1,1, 0.05])

        my_ts = [4,9,14,19]
        for t in range(4):
            ax = fig.add_subplot(inner[0, t])
            im = ax.imshow(x[field, my_ts[t]], cmap="viridis", interpolation="none", vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if field == 0:
                a = my_ts[t]+1
                ax.set_title(f"$t + {a}$​")
            if t == 0:
                ax.set_ylabel(f"{field_names[field]}")
                
        # --- Colorbar for error ------------------
        cax5 = fig.add_subplot(inner[0, 4])
        fig.colorbar(im, cax=cax5, format="%.2f")
    
    y_ys=[1,3,5]
    for field in range(FF):
        vmin_x = np.nanmin(x[field])
        vmax_x = np.nanmax(x[field])
        vmin_y = np.nanmin(y[field])
        vmax_y = np.nanmax(y[field])
        vmin = min(vmin_x, vmin_y)
        vmax = max(vmax_x, vmax_y)

        inner = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=outer[y_ys[field]], wspace=0.1, hspace=0, \
            width_ratios=[1,1,1,1, 0.05])

        my_ts = [4,9,14,19]
        for t in range(4):
            ax = fig.add_subplot(inner[0, t])
            im = ax.imshow(y[field, my_ts[t]], cmap="viridis", interpolation="none", vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if t == 0:
                ax.set_ylabel(f"{field_names2[field]}")
                
        # --- Colorbar for error ------------------
        cax5 = fig.add_subplot(inner[0, 4])
        fig.colorbar(im, cax=cax5, format="%.2f")