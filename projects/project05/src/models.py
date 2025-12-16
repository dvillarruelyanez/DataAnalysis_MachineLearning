import torch
import torch.nn as nn
from typing import Tuple, List, Union

class ConvLSTMCell(nn.Module):
    """
    A single Convolutional LSTM Cell.
    
    Args:
        input_dim (int): Number of channels in the input tensor.
        hidden_dim (int): Number of channels in the hidden state.
        kernel_size (tuple): Size of the convolutional kernel (height, width).
        bias (bool): Whether to add a bias term to the convolutions.
    """

    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: Tuple[int, int], bias: bool):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Ensure padding keeps the spatial dimensions constant
        self.padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.kernel_size = kernel_size
        self.bias = bias

        # We combine the input and hidden state concatenation into a single convolution
        # Output channels = 4 * hidden_dim because we have 4 gates (i, f, o, g).
        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )

    def forward(self, input_tensor: torch.Tensor, cur_state: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for a single time step.

        Args:
            input_tensor (torch.Tensor): Current input. Shape (B, C_in, H, W).
            cur_state (tuple): Previous states (h_prev, c_prev). 
                               Each with shape (B, C_hidden, H, W).

        Returns:
            tuple: (h_next, c_next)
        """
        h_cur, c_cur = cur_state

        # Concatenate along channel axis: (B, C_in + C_hidden, H, W)
        combined = torch.cat([input_tensor, h_cur], dim=1)
        
        # Convolution operation
        combined_conv = self.conv(combined)
        
        # Split the result into the 4 gates: Input, Forget, Output, Cell Candidate
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        # Apply activation functions
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        # Update cell state
        c_next = f * c_cur + i * g
        
        # Update hidden state
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size: int, image_size: Tuple[int, int], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initializes the Hidden (h) and Cell (c) states with zeros.
        """
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=device))


class ConvLSTM(nn.Module):
    """
    A Multi-layer ConvLSTM Network.
    
    This module manages a stack of ConvLSTMCells and handles the unrolling of the 
    sequence over time.

    Args:
        input_dim (int): Number of input channels.
        hidden_dim (list[int]): Number of hidden channels for each layer.
        kernel_size (tuple or list[tuple]): Kernel size for convolutions.
        num_layers (int): Number of stacked ConvLSTM layers.
        batch_first (bool): If True, input format is (B, T, C, H, W). Defaults to True.
        return_all_layers (bool): If True, returns outputs from all layers. Defaults to False.
    """

    def __init__(self, input_dim: int, hidden_dim: List[int], kernel_size: Union[Tuple, List[Tuple]], 
                 num_layers: int, batch_first: bool = True, bias: bool = True, return_all_layers: bool = False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Ensure parameters are lists matching the number of layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length for hidden_dim or kernel_size.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            
            cell_list.append(ConvLSTMCell(
                input_dim=cur_input_dim,
                hidden_dim=self.hidden_dim[i],
                kernel_size=self.kernel_size[i],
                bias=self.bias
            ))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor: torch.Tensor, hidden_state: List = None):
        """
        Args:
            input_tensor: (B, T, C, H, W) if batch_first=True.
            hidden_state: List of (h, c) tuples for initialization.

        Returns:
            layer_output_list: Output sequence from the last layer (or all layers).
            last_state_list: Final (h, c) states for all layers.
        """
        if not self.batch_first:
            # Permute to (B, T, C, H, W) for internal processing
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, seq_len, _, h, w = input_tensor.size()

        # Auto-initialize hidden states if None, on the same device as input
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w), device=input_tensor.device)

        layer_output_list = []
        last_state_list = []

        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            
            for t in range(seq_len):
                # Input to cell: (B, C, H, W) at time t
                # h, c are updated in place for the loop
                h, c = self.cell_list[layer_idx](
                    input_tensor=cur_layer_input[:, t, :, :, :],
                    cur_state=(h, c)
                )
                output_inner.append(h)

            # Stack time steps: (B, T, C, H, W)
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output # Output of layer i is input to layer i+1

            layer_output_list.append(layer_output)
            last_state_list.append((h, c))

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size, device):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size, device))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class Forecaster(nn.Module):
    """
    The final video prediction model.
    
    Architecture:
        1. ConvLSTM Encoder-Decoder (Sequence Processing).
        2. Final Convolutional Layer (Projection to Image Space).
        3. Sigmoid Activation (Normalization to 0-1).

    Args:
        input_dim (int): Input channels (e.g., 1 for grayscale).
        hidden_dims (list[int]): Channels in hidden layers.
        kernel_size (tuple): Convolution kernel size.
        num_layers (int): Number of LSTM layers.
    """
    def __init__(self, input_dim: int, hidden_dims: List[int], kernel_size: Tuple[int, int], num_layers: int):
        super(Forecaster, self).__init__()
        
        self.convlstm = ConvLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dims,
            kernel_size=kernel_size,
            num_layers=num_layers,
            batch_first=True,
            bias=True,
            return_all_layers=False
        )
        
        # Project back to input_dim (1 channel)
        self.final_conv = nn.Conv2d(
            in_channels=hidden_dims[-1],
            out_channels=input_dim,
            kernel_size=(3, 3), # 3x3 is standard for refinement
            padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input sequence (B, T, C, H, W).
            
        Returns:
            torch.Tensor: Prediction for the NEXT time step.
                          Shape (B, 1, 1, H, W).
        """
        # Run the ConvLSTM over the sequence
        # layer_output is a list, we want the last layer -> [0]
        layer_output, _ = self.convlstm(x)
        
        # Get the output of the very last time step from the sequence
        # Shape: (B, T, C, H, W) -> (B, C, H, W)
        last_time_step_output = layer_output[0][:, -1, :, :, :]
        
        # Project to image space
        prediction = self.final_conv(last_time_step_output)
        
        # Sigmoid to ensure range [0, 1] (images)
        prediction = torch.sigmoid(prediction)
        
        # Add time dimension back -> (B, 1, 1, H, W) for consistency with loss_fn
        return prediction.unsqueeze(1)