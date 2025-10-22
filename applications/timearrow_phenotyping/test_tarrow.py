# %% Imports
import torch
import torchview
from viscy.data.tarrow import TarrowDataModule
from viscy.representation.timearrow import TarrowModule


# %% Load minimal config
config = {
    'data': {
        'init_args': {
            'ome_zarr_path': '/hpc/projects/organelle_phenotyping/ALFI_models_data/datasets/zarr_datasets/float_phase_ome_zarr_output_valtrain.zarr',  # Replace with actual path
            'channel_name': 'DIC',
            'patch_size': [256, 256],
            'batch_size': 32,
            'num_workers': 4,
            'train_split': 0.8
        }
    },
    'model': {
        'init_args': {
            'backbone': 'unet',
            'projection_head': 'minimal_batchnorm',
            'classification_head': 'minimal',
        }
    }
}

# # Optionally load config from file
# config_path = "/hpc/projects/organelle_phenotyping/models/ALFI/tarrow_test/tarrow.yml"
# with open(config_path) as f:
#     config = yaml.safe_load(f)

# %% Initialize data and model
data_module = TarrowDataModule(**config['data']['init_args'])
model = TarrowModule(**config['model']['init_args'])
# %% Construct a batch of data from the data module
data_module.setup('fit')
batch = next(iter(data_module.train_dataloader()))
images, labels = batch
print(model)
# %% Print model graph.
try:
    # Try constructing the graph
    model_graph = torchview.draw_graph(
        model, 
        input_data=images,
        save_graph=False,  # Don't save, just display
        expand_nested=True,
        device='cpu'  # specify CPU device
    )
except Exception as e:
    print(f"Error generating model graph: {e}")

model_graph.visual_graph  # Display the graph

# %%
