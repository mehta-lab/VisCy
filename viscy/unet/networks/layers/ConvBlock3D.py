from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock3D(nn.Module):
    """3D convolutional building block for volumetric neural networks.

    A flexible 3D convolutional block designed for processing volumetric data
    such as medical imaging, microscopy, and video sequences. Supports residual
    connections, various normalization schemes, activation functions, and
    configurable layer ordering for deep 3D U-Net architectures.

    The block processes tensors in [..., z, x, y] or [..., z, y, x] format
    and provides dynamic layer configuration with support for transpose
    convolutions, dropout, and multiple padding strategies optimized for
    volumetric convolution operations.

    Parameters
    ----------
    in_filters : int
        Number of input feature channels.
    out_filters : int
        Number of output feature channels.
    dropout : float or bool, default=False
        Dropout probability. If False, no dropout is applied.
    norm : {"batch", "instance"}, default="batch"
        Normalization type to apply.
    residual : bool, default=True
        Whether to include residual connections.
    activation : {"relu", "leakyrelu", "elu", "selu", "linear"}, default="relu"
        Activation function type.
    transpose : bool, default=False
        Whether to use transpose convolution layers.
    kernel_size : int or tuple of int, default=(3, 3, 3)
        3D convolutional kernel size.
    num_repeats : int, default=3
        Number of convolutional layers in the block.
    filter_steps : {"linear", "first", "last"}, default="first"
        Strategy for channel dimension changes across layers.
    layer_order : str, default="can"
        Order of conv (c), activation (a), normalization (n) layers.
    padding : str, int, tuple or None, default=None
        Padding strategy for convolutions.
    """

    def __init__(
        self,
        in_filters: int,
        out_filters: int,
        dropout: float | bool = False,
        norm: Literal["batch", "instance"] = "batch",
        residual: bool = True,
        activation: Literal["relu", "leakyrelu", "elu", "selu", "linear"] = "relu",
        transpose: bool = False,
        kernel_size: int | tuple[int, int, int] = (3, 3, 3),
        num_repeats: int = 3,
        filter_steps: Literal["linear", "first", "last"] = "first",
        layer_order: str = "can",
        padding: str | int | tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()
        self.in_filters = in_filters
        self.out_filters = out_filters
        self.dropout = dropout
        self.norm = norm
        self.residual = residual
        self.activation = activation
        self.transpose = transpose
        self.num_repeats = num_repeats
        self.filter_steps = filter_steps
        self.layer_order = layer_order

        # ---- Handle Kernel ----#
        ks = kernel_size
        if isinstance(ks, int):
            assert ks % 2 == 1, "Kernel dims must be odd"
        elif isinstance(ks, tuple):
            for i in range(len(ks)):
                assert ks[i] % 2 == 1, "Kernel dims must be odd"
            assert i == 2, "kernel_size length must be 3"
        else:
            raise AttributeError("'kernel_size' must be either int or tuple")
        self.kernel_size = kernel_size

        # ---- Handle Padding ----#
        self.pad_type = "same"
        self.padding = (ks[2] // 2, ks[1] // 2, ks[0] // 2)
        if padding == "valid":
            self.padding = (0, 0, 0)
        elif self.padding == "valid_stack":  # note: deprecated
            ks = kernel_size
            self.padding = (0, 0, ks[0] // 2)
        elif isinstance(padding, tuple):
            self.padding = padding
        self.padding = tuple(self.padding[i // 2] for i in range(6)) + (0,) * 4

        # ----- Init Dropout -----#
        if self.dropout:
            self.drop_list = []
            for i in range(self.num_repeats):
                self.drop_list.append(nn.Dropout3d(self.dropout))
            self.register_modules(self.drop_list, "dropout")

        # ---- Init linear filter steps ----#
        steps = np.linspace(in_filters, out_filters, num_repeats + 1).astype(int)

        # ----- Init Normalization Layers -----#
        self.norm_list = [None for i in range(num_repeats)]
        if self.norm == "batch":
            for i in range(self.num_repeats):
                if self.filter_steps == "linear":
                    self.norm_list[i] = nn.BatchNorm3d(steps[i + 1])
                elif self.filter_steps == "first":
                    self.norm_list[i] = nn.BatchNorm3d(steps[-1])
                elif self.filter_steps == "last":
                    if i < self.num_repeats - 1:
                        self.norm_list[i] = nn.BatchNorm3d(steps[0])
                    else:
                        self.norm_list[i] = nn.BatchNorm3d(steps[-1])
        elif self.norm == "instance":
            for i in range(self.num_repeats):
                if self.filter_steps == "linear":
                    self.norm_list[i] = nn.InstanceNorm3d(steps[i + 1])
                elif self.filter_steps == "first":
                    self.norm_list[i] = nn.InstanceNorm3d(steps[-1])
                elif self.filter_steps == "last":
                    if i < self.num_repeats - 1:
                        self.norm_list[i] = nn.InstanceNorm3d(steps[0])
                    else:
                        self.norm_list[i] = nn.InstanceNorm3d(steps[-1])
        self.register_modules(self.norm_list, f"{norm}_norm")

        # ----- Init Conv Layers -----#
        #
        # init conv layers and determine transposition during convolution
        # The parameters governing the initiation logic flow are:
        #                 self.filter_steps
        #                 self.transpose
        #                 self.num_repeats
        # See above for definitions.
        # -------#

        self.conv_list = []
        if self.filter_steps == "linear":
            for i in range(self.num_repeats):
                depth_pair = (
                    (steps[i], steps[i + 1])
                    if i + 1 < num_repeats
                    else (steps[i], steps[-1])
                )
                if self.transpose:
                    self.conv_list.append(
                        nn.ConvTranspose3d(
                            depth_pair[0], depth_pair[1], kernel_size=kernel_size
                        )
                    )
                else:
                    self.conv_list.append(
                        nn.Conv3d(depth_pair[0], depth_pair[1], kernel_size=kernel_size)
                    )

        elif self.filter_steps == "first":
            if self.transpose:
                for i in range(self.num_repeats):
                    if i == 0:
                        self.conv_list.append(
                            nn.ConvTranspose3d(
                                in_filters, out_filters, kernel_size=kernel_size
                            )
                        )
                    else:
                        self.conv_list.append(
                            nn.ConvTranspose3d(
                                out_filters, out_filters, kernel_size=kernel_size
                            )
                        )
            else:
                for i in range(self.num_repeats):
                    if i == 0:
                        self.conv_list.append(
                            nn.Conv3d(in_filters, out_filters, kernel_size=kernel_size)
                        )
                    else:
                        self.conv_list.append(
                            nn.Conv3d(out_filters, out_filters, kernel_size=kernel_size)
                        )

        elif self.filter_steps == "last":
            if self.transpose:
                for i in range(self.num_repeats):
                    if i == self.num_repeats - 1:
                        self.conv_list.append(
                            nn.ConvTranspose3d(
                                in_filters, out_filters, kernel_size=kernel_size
                            )
                        )
                    else:
                        self.conv_list.append(
                            nn.ConvTranspose3d(
                                in_filters, in_filters, kernel_size=kernel_size
                            )
                        )
            else:
                for i in range(self.num_repeats):
                    if i == self.num_repeats - 1:
                        self.conv_list.append(
                            nn.Conv3d(in_filters, out_filters, kernel_size=kernel_size)
                        )
                    else:
                        self.conv_list.append(
                            nn.Conv3d(in_filters, in_filters, kernel_size=kernel_size)
                        )
        self.register_modules(self.conv_list, "Conv3d")

        # ----- Init Residual Layer -----#
        # Note that convolution is only used in residual layer
        # when block is shrinking feature space
        # Unregistered -- Not a learnable parameter
        self.resid_conv = nn.Conv3d(
            self.in_filters, self.out_filters, kernel_size=1, padding=0
        )

        # ----- Init Activation Layers -----#
        self.act_list = []
        if self.activation == "relu":
            for i in range(self.num_repeats):
                self.act_list.append(nn.ReLU())
        elif self.activation == "leakyrelu":
            for i in range(self.num_repeats):
                self.act_list.append(nn.LeakyReLU())
        elif self.activation == "elu":
            for i in range(self.num_repeats):
                self.act_list.append(nn.ELU())
        elif self.activation == "selu":
            for i in range(self.num_repeats):
                self.act_list.append(nn.SELU())
        elif self.activation != "linear":
            raise NotImplementedError(
                f"Activation type {self.activation} not supported."
            )
        self.register_modules(self.act_list, f"{self.activation}_act")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward call of convolutional block.

        Order of layers within the block is defined by the 'layer_order' parameter,
        which is a string of 'c's, 'a's and 'n's in reference to
        convolution, activation, and normalization layers.
        This sequence is repeated num_repeats times.

        Recommended layer order:   convolution -> activation -> normalization

        Regardless of layer order,
        the final layer sequence in the block always ends in activation.
        This allows for usage of passthrough layers
        or a final output activation function determined separately.

        Residual blocks:
            if input channels are greater than output channels,
            we use a 1x1 convolution on input to get desired feature channels
            if input channels are less than output channels,
            we zero-pad input channels to output channel size

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        x_0 = x
        for i in range(self.num_repeats):
            order = list(self.layer_order)
            while len(order) > 0:
                layer = order.pop(0)
                if layer == "c":
                    x = F.pad(x, self.padding, "constant", 0)
                    x = self.conv_list[i](x)
                    if self.dropout:
                        x = self.drop_list[i](x)
                elif layer == "a":
                    if i < self.num_repeats - 1 or self.activation != "linear":
                        x = self.act_list[i](x)
                elif layer == "n" and self.norm_list[i]:
                    x = self.norm_list[i](x)

        # residual summation comes after final activation/normalization
        if self.residual:
            # pad/collapse feature dimension
            if self.in_filters > self.out_filters:
                x_0 = self.resid_conv(x_0)
            elif self.in_filters < self.out_filters:
                x_0 = F.pad(
                    x_0,
                    (*[0] * 6, self.out_filters - self.in_filters, *[0] * 3),
                    mode="constant",
                    value=0,
                )

            # fit xy dimensions
            if self.pad_type == "valid_stack":
                lost = [dim // 2 * self.num_repeats for dim in self.kernel_size[1:]]
                x_0 = x_0[
                    ...,
                    lost[0] : x_0.shape[-2] - lost[0],
                    lost[1] : x_0.shape[-1] - lost[1],
                ]

            x = torch.add(x, x_0)

        return x

    def model(self) -> nn.Sequential:
        """
        Create sequential model from ConvBlock parameters.

        Layer order: convolution -> normalization -> activation

        We can make a list of layer modules and unpack them into nn.Sequential.
        Note: this is distinct from the forward call
            because we want to use the forward call with addition,
            since this is a residual block.
            The forward call performs the residual calculation,
            and all the parameters can be seen by the optimizer when given this model.

        Returns
        -------
        nn.Sequential
            Sequential model containing all layers in the block.
        """
        layers = []

        for i in range(self.num_repeats):
            layers.append(self.conv_list[i])
            if self.dropout:
                layers.append(self.drop_list[i])
            if self.norm[i]:
                layers.append(self.norm_list[i])
            if i < len(self.act_list):
                layers.append(self.act_list[i])

        return nn.Sequential(*layers)

    def register_modules(self, module_list: list[nn.Module], name: str) -> None:
        """
        Register modules for PyTorch optimizer visibility.

        Used to enable model graph creation
        with non-sequential model types and dynamic layer numbers

        Parameters
        ----------
        module_list : list[torch.nn.Module]
            List of modules to register.
        name : str
            Name of module type.
        """
        for i, module in enumerate(module_list):
            self.add_module(f"{name}_{str(i)}", module)
