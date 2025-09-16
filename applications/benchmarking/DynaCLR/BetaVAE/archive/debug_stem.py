#!/usr/bin/env python3

import torch
from viscy.representation.vae import VaeEncoder

# Test the stem layer computation
z_stack_depth = 32
encoder = VaeEncoder(
    backbone="resnet50",
    in_channels=1,
    in_stack_depth=z_stack_depth,
    embedding_dim=256,
    stem_kernel_size=(8, 4, 4),
    stem_stride=(8, 4, 4),
)

# Create test input
x = torch.randn(1, 1, z_stack_depth, 192, 192)
print(f"Input shape: {x.shape}")

# Test stem output
stem_output = encoder.stem(x)
print(f"Stem output shape: {stem_output.shape}")

# Check what the ResNet expects
import timm
resnet50 = timm.create_model("resnet50", pretrained=True, features_only=True)
print(f"ResNet50 conv1 expects input channels: {resnet50.conv1.in_channels}")
print(f"ResNet50 conv1 produces output channels: {resnet50.conv1.out_channels}")

# Test if we can pass stem output to ResNet
try:
    # Remove conv1 like in the encoder
    resnet50.conv1 = torch.nn.Identity()
    resnet_output = resnet50(stem_output)
    print(f"ResNet output shapes: {[f.shape for f in resnet_output]}")
    print("SUCCESS: No channel mismatch!")
except Exception as e:
    print(f"ERROR: {e}")