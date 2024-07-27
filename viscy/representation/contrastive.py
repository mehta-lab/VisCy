import timm
import torch.nn as nn
import torch.nn.functional as F

from viscy.unet.networks.unext2 import UNeXt2Stem
from viscy.unet.networks.unext2 import StemDepthtoChannels

class ContrastiveEncoder(nn.Module):
    def __init__(
        self,
        backbone: str = "convnext_tiny",
        in_channels: int = 2,
        in_stack_depth: int = 12,
        stem_kernel_size: tuple[int, int, int] = (5, 3, 3),
        embedding_len: int = 256,
        stem_stride: int = 2,
        predict: bool = False,
    ):
        super().__init__()

        self.predict = predict
        self.backbone = backbone 

        """
        ContrastiveEncoder network that uses ConvNext and ResNet backbons from timm.

        Parameters:
        - backbone (str): Backbone architecture for the encoder. Default is "convnext_tiny".
        - in_channels (int): Number of input channels. Default is 2.
        - in_stack_depth (int): Number of input slices in z-stack. Default is 15.
        - stem_kernel_size (tuple[int, int, int]): 3D kernel size for the stem. Input stack depth must be divisible by the kernel depth. Default is (5, 3, 3).
        - embedding_len (int): Length of the embedding. Default is 1000.
        """

        # if in_stack_depth % stem_kernel_size[0] != 0:
        #     raise ValueError(
        #         f"Input stack depth {in_stack_depth} is not divisible "
        #         f"by stem kernel depth {stem_kernel_size[0]}."
        #     )

        # encoder
        # self.model = timm.create_model(
        #     backbone,
        #     pretrained=True,
        #     features_only=False,
        #     drop_path_rate=0.2,
        #     num_classes=4 * embedding_len,
        # )

        encoder = timm.create_model(
            backbone,
            pretrained=True,
            features_only=False,
            drop_path_rate=0.2,
            num_classes=3 * embedding_len,
        )

        if "convnext" in backbone:
            print("Using ConvNext backbone.")
            # replace the stem designed for RGB images with a stem designed to handle 3D multi-channel input.
            # in_channels_encoder = self.model.stem[0].out_channels 

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

            # stem = UNeXt2Stem(
            #     in_channels=in_channels,
            #     out_channels=in_channels_encoder,
            #     kernel_size=stem_kernel_size,
            #     in_stack_depth=in_stack_depth,
            # )
            # self.model.stem = stem

            # self.model.head.fc = nn.Sequential(
            #     self.model.head.fc,
            #     nn.ReLU(inplace=True),
            #     nn.Linear(4 * embedding_len, embedding_len),
            # )

        elif "resnet" in backbone:
            print("Using ResNet backbone.")
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
        print("using stem kernel size", stem_kernel_size)
        self.stem = StemDepthtoChannels(in_channels, in_stack_depth, in_channels_encoder, stem_kernel_size) 
        # in_channels and in_stack_depth can be changed, in_channels_encoder is computed. 
        # Using fixed stride and stem_kernel_size.


        # Append modified encoder.
        self.encoder = encoder
        # Append modified projection head.
        self.projection = projection


            # in_channels_encoder = self.model.conv1.out_channels
            # print("in_channels_encoder", in_channels_encoder)

            # stem = UNeXt2Stem(
            #     in_channels=in_channels,
            #     out_channels=in_channels_encoder,
            #     kernel_size=stem_kernel_size,
            #     in_stack_depth=in_stack_depth,
            # )
            # self.model.conv1 = stem

            # self.model.fc = nn.Sequential(
            #     nn.Linear(self.model.fc.in_features, 4 * embedding_len),
            #     nn.ReLU(inplace=True),
            #     nn.Linear(4 * embedding_len, embedding_len),
            # )

    def forward(self, x):
        x = self.stem(x)
        embedding = self.encoder(x)
        projections = self.projection(embedding)
        projections = F.normalize(projections, p=2, dim=1)
        return (
            embedding,
            projections,
        )  # Compute the loss on projections, analyze the embeddings.

        # if self.predict:
        #     print("running predict forward!")
        #     # for resnet
        #     x = self.model.conv1(x)
        #     x = self.model.bn1(x)
        #     x = self.model.act1(x)
        #     x = self.model.maxpool(x)
        #     x = self.model.layer1(x)
        #     x = self.model.layer2(x) 
        #     x = self.model.layer3(x)
        #     x = self.model.layer4(x)
        #     x = self.model.global_pool(x)
        #     features_before_projection = x.flatten(1)
        #     projections = self.model.fc(features_before_projection)
        #     projections = F.normalize(projections, p=2, dim=1)  # L2 normalization
        #     return features_before_projection, projections

        #     # for convnext code
        #     # x = self.model.stem(x)
        #     # x = self.model.stages[0](x)
        #     # x = self.model.stages[1](x)
        #     # x = self.model.stages[2](x)
        #     # x = self.model.stages[3](x)
        #     # x = self.model.head.global_pool(x)
        #     # x = self.model.head.norm(x)
        #     # x = self.model.head.flatten(x)
        #     # features_before_projection = self.model.head.drop(x)
        #     # projections = self.model.head.fc(features_before_projection)
        #     # # features_before_projection = F.normalize(
        #     # #     features_before_projection, p=2, dim=1
        #     # # )
        #     # projections = F.normalize(projections, p=2, dim=1)  # L2 normalization
        #     # print(features_before_projection.shape, projections.shape)
        #     # return features_before_projection, projections
        # # feature is without projection head
        # else:
        #     # for resnet code 
        #     # print("Running forward without predict!")
        #     # x = self.model.conv1(x)
        #     # print(f'After conv1: {x.shape}')  
        #     # x = self.model.bn1(x)
        #     # print(f'After bn1: {x.shape}')  
        #     # x = self.model.act1(x)
        #     # print(f'After act1: {x.shape}')  
        #     # x = self.model.maxpool(x)
        #     # print(f'After maxpool: {x.shape}')  
        #     # x = self.model.layer1(x)
        #     # print(f'After layer1: {x.shape}')  
        #     # x = self.model.layer2(x)
        #     # print(f'After layer2: {x.shape}')  
        #     # x = self.model.layer3(x)
        #     # print(f'After layer3: {x.shape}') 
        #     # x = self.model.layer4(x)
        #     # print(f'After layer4: {x.shape}')  
        #     # x = self.model.global_pool(x)
        #     # print(f'After global_pool: {x.shape}')  
        #     # x = x.flatten(1)
        #     # x = self.model.fc(x)
        #     # print(f'After fc: {x.shape}')  
        #     # x = F.normalize(x, p=2, dim=1)  # L2 normalization
        #     # return x
        #     print("running train forward!")
        #     projections = self.model(x)
        #     projections = F.normalize(projections, p=2, dim=1)  # L2 normalization
        #     return projections
