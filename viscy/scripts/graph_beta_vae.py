# %% Imports and paths.
import torch
import torchview

from viscy.representation.engine import BetaVaeModule
from viscy.representation.vae import BetaVaeMonai

# %% Initialize the model and log the graph.
beta_vae_model = BetaVaeMonai(
    spatial_dims=3,
    in_shape=[1, 32, 192, 192],
    out_channels=1,
    latent_size=768,
    channels=[64, 128, 256, 512],
    strides=[[2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]],
    norm="instance",
)

model_graph = torchview.draw_graph(
    beta_vae_model,
    torch.randn(1, 1, 32, 192, 192),
    depth=3,  # adjust depth to zoom in.
    device="cpu",
)
# Print the image of the model.
model_graph.resize_graph(scale=2.5)
model_graph.visual_graph


# %% Initiatlize the lightning module and view the model.
vae_module = BetaVaeModule(
    vae=beta_vae_model,
    beta=5e-1,
    beta_schedule="cosine",
    beta_min=1e-4,
    beta_warmup_epochs=200,
    use_temporal_loss=True,
    temporal_weight=1e5,
    temporal_weight_schedule="cosine",
    temporal_weight_min=5e2,
    temporal_weight_warmup_epochs=400,
    lr=5e-5,
    log_batches_per_epoch=3,
    log_samples_per_batch=3,
    example_input_array_shape=[1, 1, 32, 192, 192],
    reconstruction_loss_fn=torch.nn.MSELoss(reduction="sum"),
)
print(vae_module.model)

# %%
model_graph = torchview.draw_graph(
    vae_module.model,
    torch.randn(1, 1, 32, 192, 192),
    depth=3,  # adjust depth to zoom in.
    device="cpu",
)
# Print the image of the model.
model_graph.visual_graph
