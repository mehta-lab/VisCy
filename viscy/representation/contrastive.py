import timm
import torch.nn as nn

# from viscy.unet.networks.resnet import resnetStem
# Currently identical to resnetStem, but could be different in the future.

from viscy.unet.networks.unext2 import UNeXt2Stem

class ContrastiveEncoder(nn.Module):
    def __init__(
        self,
        backbone: str = "convnext_tiny",
        in_channels: int = 2,
        in_stack_depth: int = 15,
        stem_kernel_size: tuple[int, int, int] = (5, 3, 3),
        embedding_len: int = 256,
    ):
        super().__init__()

        """
        ContrastiveEncoder network that uses ConvNext and ResNet backbons from timm.

        Parameters:
        - backbone (str): Backbone architecture for the encoder. Default is "convnext_tiny".
        - in_channels (int): Number of input channels. Default is 2.
        - in_stack_depth (int): Number of input slices in z-stack. Default is 15.
        - stem_kernel_size (tuple[int, int, int]): 3D kernel size for the stem. Input stack depth must be divisible by the kernel depth. Default is (5, 3, 3).
        - embedding_len (int): Length of the embedding. Default is 1000.
        """

        if in_stack_depth % stem_kernel_size[0] != 0:
            raise ValueError(
                f"Input stack depth {in_stack_depth} is not divisible "
                f"by stem kernel depth {stem_kernel_size[0]}."
            )

        # encoder
        self.model = timm.create_model(
            backbone,
            pretrained=True,
            features_only=False,
            drop_path_rate=0.2,
            num_classes=4 * embedding_len,
        )

        if "convnext_tiny" in backbone:
            print("Using ConvNext backbone.")
            # replace the stem designed for RGB images with a stem designed to handle 3D multi-channel input.
            in_channels_encoder = self.model.stem[0].out_channels
            stem = UNeXt2Stem(
                in_channels=in_channels,
                out_channels=in_channels_encoder,
                kernel_size=stem_kernel_size,
                in_stack_depth=in_stack_depth,
            )
            self.model.stem = stem

            self.model.head.fc = nn.Sequential(
                self.model.head.fc,
                nn.ReLU(inplace=True),
                nn.Linear(4 * embedding_len, embedding_len),
            )

            """ 
            head of convnext
            -------------------
            (head): NormMlpClassifierHead(
            (global_pool): SelectAdaptivePool2d(pool_type=avg, flatten=Identity())
            (norm): LayerNorm2d((768,), eps=1e-06, elementwise_affine=True)
            (flatten): Flatten(start_dim=1, end_dim=-1)
            (pre_logits): Identity()
            (drop): Dropout(p=0.0, inplace=False)
            (fc): Linear(in_features=768, out_features=1024, bias=True)


            head of convnext for contrastive learning
            ----------------------------
            (head): NormMlpClassifierHead(
            (global_pool): SelectAdaptivePool2d(pool_type=avg, flatten=Identity())
            (norm): LayerNorm2d((768,), eps=1e-06, elementwise_affine=True)
            (flatten): Flatten(start_dim=1, end_dim=-1)
            (pre_logits): Identity()
            (drop): Dropout(p=0.0, inplace=False)
            (fc): Sequential(
            (0): Linear(in_features=768, out_features=1024, bias=True)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=1024, out_features=256, bias=True)
            )
            """

        # TO-DO: need to debug further
        elif "resnet" in backbone:
            print("Using ResNet backbone.")
            # Adapt stem and projection head of resnet here.
            # replace the stem designed for RGB images with a stem designed to handle 3D multi-channel input.
            in_channels_encoder = self.model.conv1.out_channels
            stem = UNeXt2Stem(
                in_channels=in_channels,
                out_channels=in_channels_encoder,
                kernel_size=stem_kernel_size,
                in_stack_depth=in_stack_depth,
            )
            self.model.conv1 = stem

            self.model.head.fc = nn.Sequential(
                self.model.head.fc,
                nn.ReLU(inplace=True),
                nn.Linear(4 * embedding_len, embedding_len),
            )

            """ 
            head of resnet
            -------------------
            (global_pool): SelectAdaptivePool2d(pool_type=avg, flatten=Flatten(start_dim=1, end_dim=-1))
            (fc): Linear(in_features=2048, out_features=1024, bias=True)


            head of resnet for contrastive learning
            ----------------------------
            (global_pool): SelectAdaptivePool2d(pool_type=avg, flatten=Flatten(start_dim=1, end_dim=-1))
            (fc): Sequential(
            (0): Linear(in_features=2048, out_features=1024, bias=True)
            (1): ReLU(inplace=True)
            (2): Linear(in_features=1024, out_features=256, bias=True)
            """

    def forward(self, x):
        features = self.model.forward_features(x)  # extract features
        intermediate_embeddings = self.model.head.global_pool(features)  # apply global pooling
        intermediate_embeddings = self.model.head.flatten(intermediate_embeddings)  # flatten features
        projected_embeddings = self.model.head.fc(intermediate_embeddings)  # apply projection head
        return intermediate_embeddings, projected_embeddings