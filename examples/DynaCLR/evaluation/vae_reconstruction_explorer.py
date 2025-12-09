# %%

import logging

import anndata as ad
import numpy as np
import plotly.graph_objects as go
import torch
from dash import Dash, Input, Output, dcc, html

from viscy.representation.engine import BetaVaeModule

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION - Edit these paths for your data
# =============================================================================

# Path to the embeddings zarr file (output from EmbeddingWriter)
EMBEDDINGS_PATH = "/hpc/projects/intracellular_dashboard/viral-sensor/2024_08_14_ZIKV_pal17_48h/6-phenotype/predictions/vae/timeaware/tvae_timeaware_beta_0_200epoch_tmatching_2e5_400epoch_lr_2e-5_nofc_ckpt121.zarr"

# Path to the model checkpoint
CHECKPOINT_PATH = "/hpc/projects/organelle_phenotyping/models/Phase/vae/timeaware/lightning_logs/timeaware_beta_0_200epoch_tmatching_2e5_400epoch_lr_2e-5_nofc/20251119-023041/checkpoints/epoch=121-loss=4205462.500.ckpt"

# Number of samples to visualize (subset for interactive plotting)
N_SAMPLES = 500

# Which dimensionality reduction to use for visualization ("umap", "phate", "pca")
PROJECTION_TYPE = "phate"

# Z-slice to show in reconstruction (middle slice if None)
Z_SLICE = None

# Random seed for reproducibility
RANDOM_SEED = 42

# Port for Dash server
PORT = 8050

# Host (use 0.0.0.0 to allow external connections)
HOST = "0.0.0.0"

# =============================================================================
# MAIN SCRIPT
# =============================================================================


def load_embeddings(embeddings_path: str) -> ad.AnnData:
    """Load embeddings from zarr store."""
    _logger.info(f"Loading embeddings from {embeddings_path}")
    adata = ad.read_zarr(embeddings_path)
    _logger.info(f"Loaded {adata.n_obs} samples with {adata.n_vars} dimensions")
    return adata


def load_model(checkpoint_path: str) -> BetaVaeModule:
    """Load VAE model from checkpoint."""
    from viscy.representation.vae import BetaVaeMonai

    _logger.info(f"Loading model from {checkpoint_path}")

    # Instantiate the VAE architecture (must match the training config)
    vae = BetaVaeMonai(
        spatial_dims=3,
        in_shape=[1, 32, 192, 192],
        out_channels=1,
        latent_size=768,
        channels=[64, 128, 256, 512],
        strides=[[2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]],
        kernel_size=3,
        up_kernel_size=3,
        num_res_units=0,
        use_sigmoid=False,
        norm="instance",
        inter_channels=None,
        inter_dilations=None,
        num_inter_units=2,
        act="PRELU",
        dropout=None,
        bias=True,
    )

    # Load checkpoint with the VAE model
    model = BetaVaeModule.load_from_checkpoint(
        checkpoint_path, vae=vae, map_location="cpu"
    )
    model.eval()
    return model


def generate_reconstructions(model: BetaVaeModule, z_codes: np.ndarray) -> np.ndarray:
    """Generate reconstructions from latent codes."""
    _logger.info(f"Generating reconstructions for {len(z_codes)} samples")

    with torch.inference_mode():
        z_tensor = torch.from_numpy(z_codes).float()

        # Use the MONAI VarAutoEncoder's decode_forward method
        # model.model is the BetaVaeMonai wrapper, model.model.model is the MONAI VarAutoEncoder
        # use_sigmoid should match training config (False in your case)
        reconstructions = model.model.model.decode_forward(z_tensor, use_sigmoid=False)

        # Move to CPU and convert to numpy
        reconstructions = reconstructions.cpu().numpy()

    _logger.info(f"Reconstructions shape: {reconstructions.shape}")
    return reconstructions


def create_dash_app(
    adata: ad.AnnData,
    reconstructions: np.ndarray,
    projection_type: str = "phate",
    z_slice: int | None = None,
):
    """Create Dash app with interactive hover-to-see reconstructions."""

    # Get available projections
    available_projections = []
    for key in adata.obsm.keys():
        if key.startswith("X_"):
            proj_name = key[2:]  # Remove "X_" prefix
            available_projections.append(proj_name)

    _logger.info(f"Available projections: {available_projections}")

    # Validate initial projection
    if projection_type not in available_projections:
        _logger.warning(
            f"Projection '{projection_type}' not found. Using '{available_projections[0]}'"
        )
        projection_type = available_projections[0]

    # Select z-slice for visualization
    if z_slice is None:
        z_slice = reconstructions.shape[2] // 2

    _logger.info(f"Using z-slice {z_slice} for visualization")

    # Extract 2D slices from reconstructions
    recon_slices = reconstructions[:, 0, z_slice, :, :]  # [N, H, W]

    # Normalize reconstructions for display
    recon_min = recon_slices.min(axis=(1, 2), keepdims=True)
    recon_max = recon_slices.max(axis=(1, 2), keepdims=True)
    recon_slices_norm = (recon_slices - recon_min) / (recon_max - recon_min + 1e-8)

    # Create hover text with metadata
    hover_text = []
    for i in range(len(adata)):
        text = f"<b>Sample {i}</b><br>"
        for col in adata.obs.columns[:5]:  # Show first 5 metadata columns
            text += f"{col}: {adata.obs.iloc[i][col]}<br>"
        hover_text.append(text)

    # Initialize Dash app
    app = Dash(__name__)

    # Function to create scatter plot for a given projection
    def create_scatter_plot(proj_type, x_component=0, y_component=1):
        coords = adata.obsm[f"X_{proj_type}"]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=coords[:, x_component],
                y=coords[:, y_component],
                mode="markers",
                marker=dict(
                    size=8,
                    color=np.arange(len(coords)),
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Sample Index"),
                ),
                text=hover_text,
                hovertemplate="%{text}<extra></extra>",
                customdata=np.arange(len(coords)),
            )
        )
        fig.update_layout(
            title=f"VAE Latent Space ({proj_type.upper()} projection)",
            xaxis_title=f"{proj_type.upper()} {x_component + 1}",
            yaxis_title=f"{proj_type.upper()} {y_component + 1}",
            hovermode="closest",
            height=600,
        )
        return fig

    # Get number of components for each projection
    projection_n_components = {
        proj: adata.obsm[f"X_{proj}"].shape[1] for proj in available_projections
    }

    # Create initial scatter plot
    scatter_fig = create_scatter_plot(projection_type)

    # Create initial reconstruction heatmap
    recon_fig = go.Figure()
    recon_fig.add_trace(
        go.Heatmap(
            z=recon_slices_norm[0],
            colorscale="Gray",
            showscale=False,
            hoverinfo="skip",
        )
    )
    recon_fig.update_layout(
        title="Reconstruction (hover over points)",
        xaxis=dict(showticklabels=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(showticklabels=False, constrain="domain"),
        height=600,
    )

    # App layout
    app.layout = html.Div(
        [
            html.H1(
                "VAE Reconstruction Explorer",
                style={"textAlign": "center", "marginBottom": 20},
            ),
            html.Div(
                [
                    html.Label(
                        "Projection Type:",
                        style={"fontWeight": "bold", "marginRight": 10},
                    ),
                    dcc.Dropdown(
                        id="projection-dropdown",
                        options=[
                            {"label": proj.upper(), "value": proj}
                            for proj in available_projections
                        ],
                        value=projection_type,
                        clearable=False,
                        style={
                            "width": "150px",
                            "display": "inline-block",
                            "marginRight": 20,
                        },
                    ),
                    html.Label(
                        "X-axis:", style={"fontWeight": "bold", "marginRight": 10}
                    ),
                    dcc.Dropdown(
                        id="x-component-dropdown",
                        options=[
                            {"label": f"PC{i + 1}", "value": i}
                            for i in range(projection_n_components[projection_type])
                        ],
                        value=0,
                        clearable=False,
                        style={
                            "width": "100px",
                            "display": "inline-block",
                            "marginRight": 20,
                        },
                    ),
                    html.Label(
                        "Y-axis:", style={"fontWeight": "bold", "marginRight": 10}
                    ),
                    dcc.Dropdown(
                        id="y-component-dropdown",
                        options=[
                            {"label": f"PC{i + 1}", "value": i}
                            for i in range(projection_n_components[projection_type])
                        ],
                        value=1,
                        clearable=False,
                        style={"width": "100px", "display": "inline-block"},
                    ),
                ],
                style={"textAlign": "center", "marginBottom": 20},
            ),
            html.Div(
                [
                    html.Div(
                        [dcc.Graph(id="scatter-plot", figure=scatter_fig)],
                        style={"width": "60%", "display": "inline-block"},
                    ),
                    html.Div(
                        [dcc.Graph(id="reconstruction-plot", figure=recon_fig)],
                        style={"width": "40%", "display": "inline-block"},
                    ),
                ]
            ),
        ]
    )

    # Callback to update component dropdowns when projection changes
    @app.callback(
        [
            Output("x-component-dropdown", "options"),
            Output("y-component-dropdown", "options"),
            Output("x-component-dropdown", "value"),
            Output("y-component-dropdown", "value"),
        ],
        Input("projection-dropdown", "value"),
    )
    def update_component_options(selected_projection):
        n_components = projection_n_components[selected_projection]
        options = [
            {"label": f"{selected_projection.upper()}{i + 1}", "value": i}
            for i in range(n_components)
        ]
        # Reset to default components 0 and 1
        return options, options, 0, 1

    # Callback to update scatter plot when projection or components change
    @app.callback(
        Output("scatter-plot", "figure"),
        [
            Input("projection-dropdown", "value"),
            Input("x-component-dropdown", "value"),
            Input("y-component-dropdown", "value"),
        ],
    )
    def update_scatter(selected_projection, x_comp, y_comp):
        return create_scatter_plot(selected_projection, x_comp, y_comp)

    # Callback for hover interaction
    @app.callback(
        Output("reconstruction-plot", "figure"), Input("scatter-plot", "hoverData")
    )
    def update_reconstruction(hover_data):
        if hover_data is None:
            # Return initial reconstruction
            idx = 0
        else:
            # Get the index of the hovered point
            idx = hover_data["points"][0]["customdata"]

        # Create updated reconstruction figure
        fig = go.Figure()
        fig.add_trace(
            go.Heatmap(
                z=recon_slices_norm[idx],
                colorscale="Gray",
                showscale=False,
                hoverinfo="skip",
            )
        )
        fig.update_layout(
            title=f"Reconstruction - Sample {idx}",
            xaxis=dict(showticklabels=False, scaleanchor="y", scaleratio=1),
            yaxis=dict(showticklabels=False, constrain="domain"),
            height=600,
        )
        return fig

    return app


def main():
    """Main execution function."""

    # Set random seed
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    # Load embeddings
    adata = load_embeddings(EMBEDDINGS_PATH)

    # Subsample if needed
    if len(adata) > N_SAMPLES:
        _logger.info(f"Subsampling to {N_SAMPLES} samples")
        indices = np.random.choice(len(adata), N_SAMPLES, replace=False)
        indices = np.sort(indices)
        adata = adata[indices].copy()

    # Check that z codes are available
    if "X_z" not in adata.obsm:
        raise ValueError(
            "No latent codes (X_z) found in embeddings. "
            "Make sure the embeddings were generated with the updated EmbeddingWriter."
        )

    # Load model
    model = load_model(CHECKPOINT_PATH)

    # Generate reconstructions
    z_codes = adata.obsm["X_z"]
    reconstructions = generate_reconstructions(model, z_codes)

    # Create Dash app
    _logger.info("Creating Dash app")
    app = create_dash_app(
        adata, reconstructions, projection_type=PROJECTION_TYPE, z_slice=Z_SLICE
    )

    # Run the server
    _logger.info(f"Starting Dash server on http://{HOST}:{PORT}")
    _logger.info("Hover over points to see their reconstructions update in real-time!")
    _logger.info("Press Ctrl+C to stop the server")
    app.run(host=HOST, port=PORT, debug=True)


if __name__ == "__main__":
    main()

# %%
