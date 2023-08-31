import torch
import torch.nn as nn

from viscy.unet.networks.layers.ConvBlock3D import ConvBlock3D


class Unet25d(nn.Module):
    def __name__(self):
        return "Unet25d"

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        in_stack_depth=5,
        out_stack_depth=1,
        xy_kernel_size=(3, 3),
        residual=False,
        dropout=0.2,
        num_blocks=4,
        num_block_layers=2,
        num_filters=[],
        task="seg",
    ):
        """
        Instance of 2.5D Unet.
        1.) https://elifesciences.org/articles/55502

        Architecture takes in stack of 2d inputs given as a 3d tensor
        and returns a 2d interpretation.
        Learns 3d information based upon input stack,
        but speeds up training by compressing 3d information before the decoding path.
        Uses interruption conv layers in the Unet skip paths to
        compress information with z-channel convolution.

        :param int in_channels: number of feature channels in (1 or more)
        :param int out_channels: number of feature channels out (1 or more)
        :param int input_stack_depth: depth of input stack in z
        :param int output_stack_depth: depth of output stack
        :param int/tuple(int, int) xy_kernel_size: size of x and y dimensions
            of conv kernels in blocks
        :param bool residual: see name
        :param float dropout: probability of dropout, between 0 and 0.5
        :param int num_blocks: number of convolutional blocks
            on encoder and decoder paths
        :param int num_block_layers: number of layer sequences repeated per block
        :param list[int] num_filters: list of filters/feature levels
            at each conv block depth
        :param str task: network task (for virtual staining this is regression),
            one of 'seg','reg'
        :param str debug_mode: if true logs features at each step of architecture,
            must be manually set
        """
        super(Unet25d, self).__init__()
        self.in_channels = in_channels
        self.num_blocks = num_blocks
        self.kernel_size = xy_kernel_size
        self.residual = residual
        assert (
            dropout >= 0 and dropout <= 0.5
        ), f"Dropout {dropout} not in allowed range: [0, 0.5]"
        self.dropout = dropout
        self.task = task
        self.debug_mode = False

        # ----- set static parameters ----- #
        self.block_padding = "same"
        down_mode = "avgpool"  # TODO set static avgpool
        up_mode = "trilinear"  # TODO set static trilinear
        activation = "relu"  # TODO set static relu
        self.bottom_block_spatial = False  # TODO set static
        # TODO set conv_block layer order variable

        # ----- Standardize Filter Sequence ----- #
        if len(num_filters) != 0:
            assert len(num_filters) == num_blocks + 1, (
                "Length of num_filters must be equal to num_"
                "blocks + 1 (number of convolutional blocks per path)."
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
                self.down_list.append(
                    nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
                )
        elif down_mode == "avgpool":
            for i in range(num_blocks):
                self.down_list.append(
                    nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
                )
        elif down_mode == "conv":
            raise NotImplementedError("Not yet implemented!")
            # TODO: implement.
        self.register_modules(self.down_list, "down_samp")

        # ----- Upsampling steps ----- #
        self.up_list = []
        for i in range(num_blocks):
            self.up_list.append(
                nn.Upsample(scale_factor=(1, 2, 2), mode=up_mode, align_corners=False)
            )

        # ----- Convolutional blocks ----- #
        self.down_conv_blocks = []
        for i in range(num_blocks):
            self.down_conv_blocks.append(
                ConvBlock3D(
                    downsampling_filters[i],
                    downsampling_filters[i + 1],
                    dropout=self.dropout,
                    residual=self.residual,
                    activation=activation,
                    kernel_size=(3, self.kernel_size[0], self.kernel_size[1]),
                    num_repeats=num_block_layers,
                )
            )
        self.register_modules(self.down_conv_blocks, "down_conv_block")

        if self.bottom_block_spatial:
            # TODO: residual must be false or dimensionality breaks. Fix later
            self.bottom_transition_block = ConvBlock3D(
                self.num_filters[-2],
                self.num_filters[-1],
                num_repeats=1,
                residual=False,
                kernel_size=(
                    1 + in_stack_depth - out_stack_depth,
                    self.kernel_size[0],
                    self.kernel_size[1],
                ),
                padding=(0, 1, 1),
            )
        else:
            self.bottom_transition_block = nn.Conv3d(
                self.num_filters[-2],
                self.num_filters[-1],
                kernel_size=(1 + in_stack_depth - out_stack_depth, 1, 1),
                padding=0,
            )

        self.up_conv_blocks = []
        for i in range(num_blocks):
            self.up_conv_blocks.append(
                ConvBlock3D(
                    upsampling_filters[i],
                    downsampling_filters[-(i + 2)],
                    dropout=self.dropout,
                    residual=self.residual,
                    activation=activation,
                    kernel_size=(1, self.kernel_size[0], self.kernel_size[1]),
                    num_repeats=num_block_layers,
                )
            )
        self.register_modules(self.up_conv_blocks, "up_conv_block")

        # ----- Skip Interruption Conv Blocks ----- #
        self.skip_conv_layers = []
        for i in range(num_blocks):
            self.skip_conv_layers.append(
                nn.Conv3d(
                    downsampling_filters[i + 1],
                    downsampling_filters[i + 1],
                    kernel_size=(1 + in_stack_depth - out_stack_depth, 1, 1),
                )
            )
        self.register_modules(self.skip_conv_layers, "skip_conv_layer")

        # ----- Terminal Block and Activation Layer ----- #
        if self.task == "reg":
            self.terminal_block = ConvBlock3D(
                downsampling_filters[1],
                out_channels,
                dropout=False,
                residual=False,
                activation="linear",
                kernel_size=(1, 3, 3),
                norm="none",
                num_repeats=1,
            )
        else:
            self.terminal_block = ConvBlock3D(
                downsampling_filters[1],
                out_channels,
                dropout=self.dropout,
                residual=False,
                activation=activation,
                kernel_size=(1, 3, 3),
                num_repeats=1,
            )

        # ----- Feature Logging ----- #
        self.log_save_folder = None

    def forward(self, x):
        """
        Forward call of network.

        Call order:
            => num_block 3D convolutional blocks, with downsampling in between (encoder)
            => skip connections between corresponding blocks in encoder and decoder
            => num_block 2D (3d with 1 z-channel) convolutional blocks, with upsampling
                between them (decoder)
            => terminal block collapses to output dimensions

        :param torch.tensor x: input image
        """

        # encoder
        skip_tensors = []
        for i in range(self.num_blocks):
            x = self.down_conv_blocks[i](x)
            skip_tensors.append(x)
            x = self.down_list[i](x)

        # transition block
        x = self.bottom_transition_block(x)

        # skip interruptions
        for i in range(self.num_blocks):
            skip_tensors[i] = self.skip_conv_layers[i](skip_tensors[i])

        # decoder
        for i in range(self.num_blocks):
            x = self.up_list[i](x)
            x = torch.cat([x, skip_tensors[-1 * (i + 1)]], 1)
            x = self.up_conv_blocks[i](x)

        # output channel collapsing layer
        x = self.terminal_block(x)
        return x

    def register_modules(self, module_list, name):
        """
        Helper function that registers modules stored in a list to the model object
        so that the can be seen by PyTorch optimizer.

        Used to enable model graph creation with
        non-sequential model types and dynamic layer numbers

        :param list(torch.nn.module) module_list: list of modules to register
        :param str name: name of module type
        """
        for i, module in enumerate(module_list):
            self.add_module(f"{name}_{str(i)}", module)
