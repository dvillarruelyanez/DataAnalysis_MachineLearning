import torch
import torch.nn as nn

device = 'cuda'

class ConvLSTMCell(nn.Module):
    """
    Basic ConvLSTM cell.
    
    Parameters:
    input_dim (int): Number of channels of input tensor.
    hidden_dim (int): Number of channels of hidden state.
    kernel_size (int, int): Size of the convolutional kernel.
    bias (bool): Whether or not to add the bias.
    """
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        # Calculate padding to keep spatial dimensions the same ("same" padding)
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        # This single convolution layer computes all 4 gates at once: (input, forget, output, cell) 
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)
    
    def forward(self, input_tensor, cur_state):
        """
        Updating the cell and hidden state for a single time step.
        
        Args:
            input_tensor (torch.Tensor): Input tensor for the current time step. 
                                         Shape: (B, C_in, H, W)
            cur_state (tuple): (h_cur, c_cur)
                h_cur (torch.Tensor): Hidden state from previous time step. 
                                      Shape: (B, C_hidden, H, W)
                c_cur (torch.Tensor): Cell state from previous time step.
                                      Shape: (B, C_hidden, H, W)
        
        Returns:
            (h_next, c_next): Next hidden and cell states.
        """
        h_cur, c_cur = cur_state

        # Concatenate input and hidden state along the channel dimension
        # (B, C_in, H, W) + (B, C_hidden, H, W) -> (B, C_in + C_hidden, H, W)
        combined = torch.cat([input_tensor, h_cur], dim=1)
        
        # Apply the convolution
        combined_conv = self.conv(combined)
        
        # Split the result into the 4 gates (i, f, o, g)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        # Apply activation functions
        i = torch.sigmoid(cc_i)  # Input gate
        f = torch.sigmoid(cc_f)  # Forget gate
        o = torch.sigmoid(cc_o)  # Output gate
        g = torch.tanh(cc_g)     # Cell gate

        # Calculate the next cell state and hidden state
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        """
        Initializes the (h, c) states with zeros.
        
        Args:
            batch_size (int): Batch size.
            image_size (tuple): (height, width) of the image.
            
        Returns:
            (torch.Tensor, torch.Tensor): A tuple of zero-filled Tensors 
                                          for (h_0, c_0).
        """
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=device))

class ConvLSTM(nn.Module):
    """
    Full ConvLSTM module that wraps ConvLSTMCell. It takes ConvLSTMCell and unroll the sequence in time and stack multiple layers.

    Parameters:
        input_dim (int): Number of channels in input (monochrome/RGB)
        hidden_dim (int/list of ints): Number of hidden channels (can be a list for multiple layers)
        kernel_size (int/list of ints): Size of kernel in convolutions (can be a list for multiple layers)
        num_layers int: Number of LSTM layers stacked on each other
        batch_first (bool): Whether or not dimension 0 is the batch (True) or not (False)
        bias (bool): Bias or no bias in Convolution
        return_all_layers (bool): Return the list of computations for all layers or just the last output
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that kernel_size and hidden_dim are lists of len num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        # Create the list of cell layers
        cell_list = []
        for i in range(0, self.num_layers):
            # Input dim for the first layer is input_dim, 
            # for subsequent layers it's the hidden_dim of the previous layer.
                if i == 0:
                    cur_input_dim = self.input_dim 
                else: 
                    cur_input_dim = self.hidden_dim[i - 1]

                cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,hidden_dim=self.hidden_dim[i], kernel_size=self.kernel_size[i],bias=self.bias))

        # nn.ModuleList holds all the layers, making them visible to PyTorch
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        Forward pass for a sequence.
        
        Args:
            input_tensor (torch.Tensor): Input sequence. 
                                         Shape: (B, T, C_in, H, W) if batch_first=True
            hidden_state (list, optional): List of (h, c) tuples for each layer.
        
        Returns:
            (layer_output_list, last_state_list)
        """
        # (T, B, C, H, W) -> (B, T, C, H, W) if batch_first=True
        if not self.batch_first:
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Initialize hidden state if not provided
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        # Loop over layers
        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            
            # Loop over time (sequence length)
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h) # Store the hidden state (output) for this time step

            # Stack all time step outputs
            layer_output = torch.stack(output_inner, dim=1)
            
            # This layer's output becomes the next layer's input
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c]) # Store the final (h, c) of the layer

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:] # Only return the last layer's output
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        """Initializes hidden state for all layers."""
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        """Helper function to check kernel_size type."""
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        """Helper function to ensure a param is a list of len num_layers."""
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

class Forecaster(nn.Module):
    """
    Final Forecaster model.
    It wraps the ConvLSTM and adds a final 2D Conv layer for prediction.
    
    Parameters:
        input_dim (int): Number of channels in input (e.g., 1 for grayscale)
        hidden_dims (int/list of ints): List of hidden channels for each ConvLSTM layer (e.g., [16, 32])
        kernel_size ([int,int]): Kernel size for ConvLSTM (e.g., (3, 3))
        num_layers (int): Number of ConvLSTM layers.
    """
    def __init__(self, input_dim, hidden_dims, kernel_size):
        super(Forecaster, self).__init__()
        
        # The ConvLSTM engine 
        self.convlstm = ConvLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dims,
            kernel_size=kernel_size,
            num_layers=len(hidden_dims),
            batch_first=True,  
            bias=True,
            return_all_layers=False # Only last layer's output
        )
        
        # The final prediction layer
        # This layer takes the output from the last ConvLSTM layer
        # back to our 'input_dim' (1 channel).
        self.final_conv = nn.Conv2d(
            in_channels=hidden_dims[-1], # Output channels from the last LSTM layer
            out_channels=input_dim,      # 1 output channel (our prediction)
            kernel_size=(3, 3),          # 3x3 kernel
            padding=1                    # 'Same padding' to keep size
        )

    def forward(self, x):
        """
        Forward pass for the complete model.
        
        Args:
            x (torch.Tensor): Input sequence tensor.
                              Shape: (B, T, C, H, W)
                              
        Returns:
            torch.Tensor: The predicted frame.
                          Shape: (B, C, H, W)
        """
        x = x.permute(0, 1, 4, 2, 3)

        # Pass the sequence through the ConvLSTM engine
        layer_output, last_states = self.convlstm(x)
        
        # Take the LAST time step
        last_time_step_output = layer_output[0][:, -1, :, :, :]
        
        # Pass through the final layer
        prediction = self.final_conv(last_time_step_output)

        prediction = (prediction.permute(0, 2, 3, 1)).unsqueeze(1)
        
        # Use Sigmoid
        return torch.sigmoid(prediction)