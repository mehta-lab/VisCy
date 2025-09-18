"""2D U-Net implementation for image-to-image translation tasks."""

import torch
import torch.nn as nn

from viscy.unet.networks.layers.ConvBlock2D import ConvBlock2D


class Unet2d(nn.Module):
    """2D U-Net neural network for image-to-image translation.

    A convolutional neural network following the U-Net architecture for 2D images.
    Supports both segmentation and regression tasks with configurable depth and filters.

    Follows 2D UNet Architecture:

    References
    ----------
    1) U-Net: https://arxiv.org/pdf/1505.04597.pdf
    2) Residual U-Net: https://arxiv.org/pdf/1711.10684.pdf

    Parameters
    ----------
    in_channels : int, optional
        Number of feature channels in, by default 1.
    out_channels : int, optional
        Number of feature channels out, by default 1.
    kernel_size : int or tuple of int, optional
        Size of x and y dimensions of conv kernels in blocks, by default (3, 3).
    residual : bool, optional
        Whether to use residual connections, by default False.
    dropout : float, optional
        Probability of dropout, between 0 and 0.5, by default 0.2.
    num_blocks : int, optional
        Number of convolutional blocks on encoder and decoder, by default 4.
    num_block_layers : int, optional
        Number of layers per block, by default 2.
    num_filters : list of int, optional
        List of filters/feature levels at each conv block depth, by default [].
    task : str, optional
        Network task (for virtual staining this is regression),
        one of 'seg','reg', by default "seg".
    """

    def __name__(self):
        """Return the name of the network architecture."""
        return "Unet2d"

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        kernel_size=(3, 3),
        residual=False,
        dropout=0.2,
        num_blocks=4,
        num_block_layers=2,
        num_filters=[],
        task="seg",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.residual = residual
        self.dropout = dropout
        self.num_blocks = num_blocks
        self.num_block_layers = num_block_layers
        self.task = task

        # ----- set static parameters -----#
        self.block_padding = "same"
        down_mode = "avgpool"  # TODO set static avgpool
        up_mode = "bilinear"  # TODO set static bilinear
        activation = "relu"  # TODO set static relu
        self.bottom_block_spatial = False  # TODO set static

        # ----- Standardize Filter Sequence -----#
        if len(num_filters) != 0:
            assert len(num_filters) == num_blocks + 1, (
                "Length of num_filters must be equal to num_blocks + 1 "
                "(number of convolutional blocks per path)."
            )
            self.num_filters = num_filters
        else:
            self.num_filters = [pow(2, i) * 16 for i in range(num_blocks + 1)]
        downsampling_filters = [in_channels] + self.num_filters
        upsampling_filters = [
            self.num_filters[-(i + 1)] + self.num_filters[-(i + 2)]
            for i in range(len(self.num_filters))
            if i < len(self.num_filters) - 1
        ] + [out_channels]

        # ----- Downsampling steps -----#
        self.down_list = []
        if down_mode == "maxpool":
            for i in range(num_blocks):
                self.down_list.append(nn.MaxPool2d(kernel_size=2))
        elif down_mode == "avgpool":
            for i in range(num_blocks):
                self.down_list.append(nn.AvgPool2d(kernel_size=2))
        elif down_mode == "conv":
            raise NotImplementedError("Not yet implemented!")
            # TODO: implement.
        self.register_modules(self.down_list, "down_samp")

        # ----- Upsampling steps -----#
        self.up_list = []
        if up_mode == "bilinear":
            for i in range(num_blocks):
                self.up_list.append(
                    nn.Upsample(mode=up_mode, scale_factor=2, align_corners=False)
                )
        elif up_mode == "conv":
            raise NotImplementedError("Not yet implemented!")
            # TODO: implement
        elif up_mode == "tconv":
            raise NotImplementedError("Not yet implemented!")
            # TODO: implement
        else:
            raise NotImplementedError(f"Upsampling mode '{up_mode}' not supported.")

        # ----- Convolutional blocks -----#
        self.down_conv_blocks = []
        for i in range(num_blocks):
            self.down_conv_blocks.append(
                ConvBlock2D(
                    downsampling_filters[i],
                    downsampling_filters[i + 1],
                    dropout=self.dropout,
                    residual=self.residual,
                    activation=activation,
                    kernel_size=self.kernel_size,
                    num_repeats=self.num_block_layers,
                )
            )
        self.register_modules(self.down_conv_blocks, "down_conv_block")

        self.bottom_transition_block = ConvBlock2D(
            self.num_filters[-2],
            self.num_filters[-1],
            dropout=self.dropout,
            residual=self.residual,
            activation=activation,
            kernel_size=self.kernel_size,
            num_repeats=self.num_block_layers,
        )

        self.up_conv_blocks = []
        for i in range(num_blocks):
            self.up_conv_blocks.append(
                ConvBlock2D(
                    upsampling_filters[i],
                    downsampling_filters[-(i + 2)],
                    dropout=self.dropout,
                    residual=self.residual,
                    activation=activation,
                    kernel_size=self.kernel_size,
                    num_repeats=self.num_block_layers,
                )
            )
        self.register_modules(self.up_conv_blocks, "up_conv_block")

        # ----- Terminal Block and Activation Layer -----#
        if self.task == "reg":
            self.terminal_block = ConvBlock2D(
                downsampling_filters[1],
                out_channels,
                dropout=self.dropout,
                residual=False,
                activation="linear",
                num_repeats=1,
                norm="none",
                kernel_size=self.kernel_size,
            )
        else:
            self.terminal_block = ConvBlock2D(
                downsampling_filters[1],
                out_channels,
                dropout=self.dropout,
                residual=False,
                activation=activation,
                num_repeats=1,
                norm="none",
                kernel_size=self.kernel_size,
            )

    def forward(self, x: torch.Tensor, validate_input: bool = False) -> torch.Tensor:
        """Forward pass through the 2D U-Net.

        Call order:
        => num_block 2D convolutional blocks, with downsampling in between (encoder)
        => num_block 2D convolutional blocks, with upsampling between them (decoder)
        => skip connections between corresponding blocks on encoder and decoder
        => terminal block collapses to output dimensions

        Parameters
        ----------
        x : torch.tensor
            Input image stack.
        validate_input : bool, optional
            Deactivates assertions which are redundant if forward pass is being
            traced by tensorboard writer, by default False.

        Returns
        -------
        torch.tensor
            Network output with same spatial dimensions as input.
        """
        # handle input exceptions
        if validate_input:
            assert x.shape[-1] == x.shape[-2], "Input must be square in xy"
            assert x.shape[-3] == self.in_channels, (
                f"Input channels must equal network input channels: {self.in_channels}"
            )
        x = x.squeeze(2)
        # encoder
        skip_tensors = []
        for i in range(self.num_blocks):
            x = self.down_conv_blocks[i](x, validate_input=validate_input)
            skip_tensors.append(x)
            x = self.down_list[i](x)

        # transition block
        x = self.bottom_transition_block(x)

        # decoder
        for i in range(self.num_blocks):
            x = self.up_list[i](x)
            x = torch.cat([x, skip_tensors[-1 * (i + 1)]], 1)
            x = self.up_conv_blocks[i](x, validate_input=validate_input)

        # output channel collapsing layer
        x = self.terminal_block(x)

        return x.unsqueeze(2)

    def register_modules(self, module_list: list[nn.Module], name: str) -> None:
        """Register modules stored in a list to the model object.

        So that they can be seen by PyTorch optimizer.

        Used to enable model graph creation with
        non-sequential model types and dynamic layer numbers.

        Parameters
        ----------
        module_list : list[torch.nn.module]
            List of modules to register
        name : str
            Name of module type
        """
        for i, module in enumerate(module_list):
            self.add_module(f"{name}_{str(i)}", module)
