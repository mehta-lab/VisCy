import timm
import torch.nn as nn
import torch.nn.functional as F

# from viscy.unet.networks.resnet import resnetStem
# Currently identical to resnetStem, but could be different in the future.
from viscy.unet.networks.unext2 import UNeXt2Stem
from viscy.unet.networks.unext2 import UNeXt2StemResNet

class ContrastiveEncoder(nn.Module):
    def __init__(
        self,
        backbone: str = "convnext_tiny",
        in_channels: int = 2,
        in_stack_depth: int = 15,
        stem_kernel_size: tuple[int, int, int] = (5, 3, 3),
        embedding_len: int = 256,
        predict: bool = False,
    ):
        super().__init__()

        self.predict = predict

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
            print("in_channels_encoder", in_channels_encoder)

            out_channels_encoder = self.model.bn1.num_features
            print("out_channels_bn", out_channels_encoder)

            stem = UNeXt2StemResNet(
                in_channels=in_channels,
                out_channels=out_channels_encoder,
                kernel_size=stem_kernel_size,
                in_stack_depth=in_stack_depth,
            )
            self.model.conv1 = stem

            self.model.bn1 = nn.BatchNorm2d(out_channels_encoder)

            print(f'Updated out_channels_encoder: {out_channels_encoder}')

            self.model.fc = nn.Sequential(
                nn.Linear(self.model.fc.in_features, 4 * embedding_len),
                nn.ReLU(inplace=True),
                nn.Linear(4 * embedding_len, embedding_len),
            )

            # self.model.fc = nn.Sequential(
            # nn.Linear(self.model.fc.in_features, 1024),
            # nn.ReLU(inplace=True),
            # nn.Linear(1024, embedding_len),
            # )

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
        if self.predict:
            print("running predict forward!")
            x = self.model.stem(x)
            x = self.model.stages[0](x)
            x = self.model.stages[1](x)
            x = self.model.stages[2](x)
            x = self.model.stages[3](x)
            x = self.model.head.global_pool(x)
            x = self.model.head.norm(x)
            x = self.model.head.flatten(x)
            features_before_projection = self.model.head.drop(x)
            projections = self.model.head.fc(features_before_projection)
            features_before_projection = F.normalize(
                features_before_projection, p=2, dim=1
            )
            projections = F.normalize(projections, p=2, dim=1)  # L2 normalization
            print(features_before_projection.shape, projections.shape)
            return features_before_projection, projections
        # feature is without projection head
        else:
            print("running forward without predict!")
            print("Running forward without predict!")
            x = self.model.conv1(x)
            print(f'After conv1: {x.shape}')  # Debugging statement
            x = self.model.bn1(x)
            print(f'After bn1: {x.shape}')  # Debugging statement
            x = self.model.act1(x)
            print(f'After act1: {x.shape}')  # Debugging statement
            x = self.model.maxpool(x)
            print(f'After maxpool: {x.shape}')  # Debugging statement
            x = self.model.layer1(x)
            print(f'After layer1: {x.shape}')  # Debugging statement
            x = self.model.layer2(x)
            print(f'After layer2: {x.shape}')  # Debugging statement
            x = self.model.layer3(x)
            print(f'After layer3: {x.shape}')  # Debugging statement
            x = self.model.layer4(x)
            print(f'After layer4: {x.shape}')  # Debugging statement
            x = self.model.global_pool(x)
            print(f'After global_pool: {x.shape}')  # Debugging statement
            x = x.flatten(1)
            x = self.model.fc(x)
            print(f'After fc: {x.shape}')  # Debugging statement
            x = F.normalize(x, p=2, dim=1)  # L2 normalization
            return x
            
            # projections = self.model(x)
            # projections = F.normalize(projections, p=2, dim=1)  # L2 normalization
            # return projections
