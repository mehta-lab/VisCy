import logging

import timm
import torch.nn as nn
import torch.nn.functional as F

from viscy.unet.networks.unext2 import StemDepthtoChannels

_logger = logging.getLogger("lightning.pytorch")


class ContrastiveEncoder(nn.Module):
    def __init__(
        self,
        backbone: str = "convnext_tiny",
        in_channels: int = 2,
        in_stack_depth: int = 12,
        stem_kernel_size: tuple[int, int, int] = (5, 4, 4),
        embedding_len: int = 256,
        stem_stride: int = 2,
        predict: bool = False,
        drop_path_rate: float = 0.2,
    ):
        """ContrastiveEncoder network that uses
        ConvNext and ResNet backbons from timm.

        :param str backbone: Backbone architecture for the encoder,
            defaults to "convnext_tiny"
        :param int in_channels: Number of input channels, defaults to 2
        :param int in_stack_depth: Number of input slices in z-stack, defaults to 12
        :param tuple[int, int, int] stem_kernel_size: 3D kernel size for the stem.
            Input stack depth must be divisible by the kernel depth,
            defaults to (5, 3, 3)
        :param int embedding_len: Length of the embedding vector, defaults to 256
        :param int stem_stride: stride of the stem, defaults to 2
        :param bool predict: prediction mode, defaults to False
        :param float drop_path_rate: probability that residual connections
            are dropped during training, defaults to 0.2
        """
        super().__init__()
        self.predict = predict
        self.backbone = backbone

        encoder = timm.create_model(
            backbone,
            pretrained=True,
            features_only=False,
            drop_path_rate=drop_path_rate,
            num_classes=3 * embedding_len,
        )

        if "convnext" in backbone:
            _logger.debug(f"Using ConvNeXt backbone for {type(self).__name__}.")

            in_channels_encoder = encoder.stem[0].out_channels

            # Remove the convolution layer of stem, but keep the layernorm.
            encoder.stem[0] = nn.Identity()

            # Save projection head separately and erase the projection head contained within the encoder.
            projection = nn.Sequential(
                nn.Linear(encoder.head.fc.in_features, 3 * embedding_len),
                nn.ReLU(inplace=True),
                nn.Linear(3 * embedding_len, embedding_len),
            )

            encoder.head.fc = nn.Identity()

        elif "resnet" in backbone:
            _logger.debug(f"Using ResNet backbone for {type(self).__name__}")
            # Adapt stem and projection head of resnet here.
            # replace the stem designed for RGB images with a stem designed to handle 3D multi-channel input.

            in_channels_encoder = encoder.conv1.out_channels
            encoder.conv1 = nn.Identity()

            projection = nn.Sequential(
                nn.Linear(encoder.fc.in_features, 3 * embedding_len),
                nn.ReLU(inplace=True),
                nn.Linear(3 * embedding_len, embedding_len),
            )
            encoder.fc = nn.Identity()

        # Create a new stem that can handle 3D multi-channel input.
        _logger.debug(f"Stem kernel size: {stem_kernel_size}")
        self.stem = StemDepthtoChannels(
            in_channels, in_stack_depth, in_channels_encoder, stem_kernel_size
        )

        # Append modified encoder.
        self.encoder = encoder
        # Append modified projection head.
        self.projection = projection

    def forward(self, x):
        x = self.stem(x)
        embedding = self.encoder(x)
        projections = self.projection(embedding)
        projections = F.normalize(projections, p=2, dim=1)
        return (
            embedding,
            projections,
        )  # Compute the loss on projections, analyze the embeddings.
