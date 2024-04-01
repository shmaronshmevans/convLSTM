"""
Code taken from:
https://github.com/ndrplz/ConvLSTM_pytorch/tree/master 
"""

import torch.nn as nn
import torch


class ConvLSTMCell(nn.Module):
    """
    Convolutional LSTM Cell.

    Parameters
    ----------
    input_dim: int
        Number of channels of the input tensor.
    hidden_dim: int
        Number of channels of the hidden state.
    kernel_size: (int, int)
        Size of the convolutional kernel.
    bias: bool
        Whether or not to add the bias.
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()

        # Store parameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        # Define convolutional layer
        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

    def forward(self, input_tensor, cur_state):
        """
        Forward pass of the ConvLSTM cell.

        Parameters
        ----------
        input_tensor: torch.Tensor
            Input tensor of shape (batch_size, input_channels, height, width).
        cur_state: tuple
            Tuple containing the current hidden and cell states.

        Returns
        -------
        h_next, c_next: torch.Tensor, torch.Tensor
            Next hidden and cell states.
        """
        # Unpack current states
        h_cur, c_cur = cur_state

        # Concatenate input tensor with previous hidden state along channel axis
        combined = torch.cat([input_tensor, h_cur], dim=1)

        # Apply convolutional operation
        combined_conv = self.conv(combined)

        # Split convolutional output into gates
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        # Apply activation functions
        i = torch.sigmoid(cc_i)  # Input gate
        f = torch.sigmoid(cc_f)  # Forget gate
        o = torch.sigmoid(cc_o)  # Output gate
        g = torch.tanh(cc_g)  # Candidate cell state

        # Compute the next cell state
        c_next = f * c_cur + i * g
        # Compute the next hidden state
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        """
        Initialize the hidden state.

        Parameters
        ----------
        batch_size: int
            Size of the batch.
        image_size: tuple
            Tuple containing height and width of the input image.

        Returns
        -------
        tuple
            Tuple containing the initialized hidden and cell states.
        """
        height, width = image_size

        # Initialize hidden and cell states with zeros
        return (
            torch.zeros(
                batch_size,
                self.hidden_dim,
                height,
                width,
                device=self.conv.weight.device,
            ),
            torch.zeros(
                batch_size,
                self.hidden_dim,
                height,
                width,
                device=self.conv.weight.device,
            ),
        )


class ConvLSTM(nn.Module):
    """
    Convolutional LSTM network.

    Parameters
    ----------
    input_dim: int
        Number of input channels.
    hidden_dim: list
        List of integers specifying the number of channels in hidden states for each layer.
    kernel_size: list
        List of tuples specifying the size of convolutional kernels for each layer.
    num_layers: int
        Number of ConvLSTM layers stacked on top of each other.
    batch_first: bool, optional
        Whether dimension 0 is the batch size or not. Default is True.
    bias: bool, optional
        Whether or not to add the bias in convolutional layers. Default is True.
    return_all_layers: bool, optional
        Whether to return the output and states for all layers. Default is False.

    Input
    -----
    A tensor of size B, T, C, H, W or T, B, C, H, W

    Output
    ------
    A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
        0 - layer_output_list is the list of lists of length T of each output
        1 - last_state_list is the list of last states
                each element of the list is a tuple (h, c) for hidden state and memory

    Example
    -------
    >> x = torch.rand((32, 10, 64, 128, 128))
    >> convlstm = ConvLSTM(64, [16], [(3, 3)], 1, True, True, False)
    >> _, last_states = convlstm(x)
    >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        kernel_size,
        num_layers,
        batch_first=True,
        bias=False,
        return_all_layers=False,
    ):
        super(ConvLSTM, self).__init__()

        # Ensure consistency of kernel size and hidden dimensions lists
        self._check_kernel_size_consistency(kernel_size)
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError("Inconsistent list length.")

        # Store parameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        # Create ConvLSTM cells for each layer
        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            cell_list.append(
                ConvLSTMCell(
                    input_dim=cur_input_dim,
                    hidden_dim=self.hidden_dim[i],
                    kernel_size=self.kernel_size[i],
                    bias=self.bias,
                )
            )
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        Forward pass of the ConvLSTM network.

        Parameters
        ----------
        input_tensor: torch.Tensor
            Input tensor of shape (batch_size, sequence_length, input_channels, height, width) or
            (sequence_length, batch_size, input_channels, height, width).
        hidden_state: None
            Not used in this implementation. Stateful behavior is not implemented.

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # Permute input tensor if batch_first is False
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        # Get dimensions of input tensor
        b, t, c, h, w = input_tensor.size()

        # Initialize hidden states
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = t
        cur_layer_input = input_tensor

        # Iterate through each layer
        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []

            # Iterate through each time step
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](
                    input_tensor=cur_layer_input[:, t, :, :, :], cur_state=[h, c]
                )
                output_inner.append(h)

            # Stack outputs along time dimension
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        # Return output and last states
        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        # Reshape output to match target output shape
        output = layer_output_list[-1]
        output = output.squeeze().permute(0, 1, 3, 2)
        output = output.transpose(1, 2)[:, :, :, 0]

        return output, last_state_list

    def _init_hidden(self, batch_size, image_size):
        """
        Initialize hidden states for all layers.

        Parameters
        ----------
        batch_size: int
            Batch size.
        image_size: tuple
            Tuple specifying the height and width of the input image.

        Returns
        -------
        init_states: list
            List of initial hidden states for all layers.
        """
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        """
        Check if kernel size is consistent.

        Parameters
        ----------
        kernel_size: tuple or list
            Size of convolutional kernels.

        Raises
        ------
        ValueError
            If kernel size is not tuple or list of tuples.
        """
        if not (
            isinstance(kernel_size, tuple)
            or (
                isinstance(kernel_size, list)
                and all([isinstance(elem, tuple) for elem in kernel_size])
            )
        ):
            raise ValueError("`kernel_size` must be tuple or list of tuples")

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        """
        Extend a parameter for multiple layers if necessary.

        Parameters
        ----------
        param: int, tuple, or list
            Parameter to be extended.
        num_layers: int
            Number of layers.

        Returns
        -------
        param: list
            Extended parameter list for all layers.
        """
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
