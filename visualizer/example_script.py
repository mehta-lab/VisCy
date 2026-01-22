# %%
from pathlib import Path

from visualizer import DatasetConfig, MultiDatasetConfig, create_app, run_app

config = MultiDatasetConfig(
    datasets=[
        DatasetConfig(
            adata_path=Path(
                "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_07_A549_SEC61_DENV/4-phenotyping/2-predictions/DynaCLR-2D-BagOfChannels-timeaware/v3/timeaware_phase_160patch_104ckpt.zarr"
            ),
            data_path=Path(
                "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_07_A549_SEC61_DENV/4-phenotyping/1-train-test/2024_11_07_A549_SEC61_DENV.zarr"
            ),
            fov_filter=["B/1", "B/3"],
            annotation_csv=None,
            annotation_column="infection_status",
            categories={0: "uninfected", 1: "infected", 2: "unknown"},
            dataset_id=None,  # Will auto-detect from zarr filename
            channels=("Phase3D",),
            z_range=(0, 1),
        ),
        DatasetConfig(
            adata_path=Path(
                "/hpc/projects/intracellular_dashboard/organelle_dynamics/2025_07_22_A549_SEC61_TOMM20_G3BP1_ZIKV/4-phenotyping/predictions/DynaCLR-2D-BagOfChannels-timeaware/v3/timeaware_phase_160patch_104ckpt.zarr"
            ),
            data_path=Path(
                "/hpc/projects/intracellular_dashboard/organelle_dynamics/2025_07_22_A549_SEC61_TOMM20_G3BP1_ZIKV/4-phenotyping/train-test/2025_07_22_A549_SEC61_TOMM20_G3BP1_ZIKV.zarr"
            ),
            fov_filter=["C/2"],
            annotation_csv=None,
            annotation_column="infection_status",
            categories={0: "uninfected", 1: "infected", 2: "unknown"},
            dataset_id=None,  # Will auto-detect from zarr filename
            channels=("Phase3D",),
            z_range=(0, 1),
        ),
        DatasetConfig(
            adata_path=Path(
                "/hpc/projects/intracellular_dashboard/organelle_box/2025_09_24_Organelle_box_NCLN/6.1-phenotyping/anndata/phase_3hpi_160patch_104ckpt_ver3max.zarr"
            ),
            data_path=Path(
                "/hpc/projects/intracellular_dashboard/organelle_box/2025_09_24_Organelle_box_NCLN/5-concatenate/3hpi.zarr"
            ),
            fov_filter=["G3BP1/uninfected"],
            annotation_csv=None,
            annotation_column="infection_status",
            categories={0: "uninfected", 1: "infected", 2: "unknown"},
            dataset_id=None,  # Will auto-detect from zarr filename
            channels=("Phase3D",),
            z_range=(0, 1),
        ),
    ],
    # PHATE parameters: None = use existing (single dataset only), or dict with params
    phate_kwargs={"n_components": 2, "knn": 5, "decay": 40, "scale_embeddings": False},
    # Image settings
    yx_patch_size=(160, 160),
    # Server settings
    port=8050,
    debug=False,
    default_color_mode="time",
)

app = create_app(config)
run_app(app)
