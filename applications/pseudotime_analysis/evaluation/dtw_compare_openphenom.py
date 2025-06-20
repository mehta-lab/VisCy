# %%
from pathlib import Path

import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Load model directly
from transformers import AutoModel

from viscy.data.triplet import TripletDataModule
from viscy.representation.evaluation.dimensionality_reduction import compute_phate
from viscy.transforms import ScaleIntensityRangePercentilesd


# %% OpenPhenom Wrapper
class OpenPhenomWrapper:
    def __init__(self):
        try:
            self.model = AutoModel.from_pretrained(
                "recursionpharma/OpenPhenom", trust_remote_code=True
            )
            self.model.eval()
            self.model.to("cuda")
        except ImportError:
            raise ImportError(
                "Please install the OpenPhenom dependencies: "
                "pip install git+https://github.com/recursionpharma/maes_microscopy.git"
            )

    def extract_features(self, x):
        """Extract features from the input images.

        Args:
            x: Input tensor of shape [B, C, D, H, W] or [B, C, H, W]

        Returns:
            Features of shape [B, 384]
        """
        # OpenPhenom expects [B, C, H, W] but our data might be [B, C, D, H, W]
        # If 5D input, take middle slice or average across D
        if x.dim() == 5:
            # Take middle slice or average across D dimension
            d = x.shape[2]
            x = x[:, :, d // 2, :, :]

        # Convert to uint8 as OpenPhenom expects uint8 inputs
        if x.dtype != torch.uint8:
            # Normalize to 0-1 range if not already
            x = (x - x.min()) / (x.max() - x.min() + 1e-10)
            x = (x * 255).clamp(0, 255).to(torch.uint8)

        # Get embeddings
        self.model.return_channelwise_embeddings = False
        with torch.no_grad():
            embeddings = self.model.predict(x)

        return embeddings


# %% Initialize OpenPhenom model
print("Loading OpenPhenom model...")
openphenom = OpenPhenomWrapper()
# For infection dataset with phase and RFP
print("Setting up data module...")
dm = TripletDataModule(
    data_path="/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_07_A549_SEC61_ZIKV_DENV/2-assemble/2024_11_07_A549_SEC61_DENV.zarr",
    tracks_path="/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_07_A549_SEC61_ZIKV_DENV/1-preprocess/label-free/4-track-gt/2024_11_07_A549_SEC61_ZIKV_DENV_2_cropped.zarr",
    source_channel=["Phase3D", "raw GFP EX488 EM525-45"],
    batch_size=32,  # Lower batch size for OpenPhenom which is larger
    num_workers=10,
    z_range=(25, 40),
    initial_yx_patch_size=(192, 192),
    final_yx_patch_size=(192, 192),
    normalizations=[
        ScaleIntensityRangePercentilesd(
            keys=["raw GFP EX488 EM525-45"], lower=50, upper=99, b_min=0.0, b_max=1.0
        ),
        ScaleIntensityRangePercentilesd(
            keys=["Phase3D"], lower=50, upper=99, b_min=0.0, b_max=1.0
        ),
    ],
)
dm.prepare_data()
dm.setup("predict")
# %%
print("Extracting features...")
features = []
indices = []

with torch.inference_mode():
    for batch in tqdm(dm.predict_dataloader()):
        # Get both channels and handle dimensions properly
        phase = batch["anchor"][:, 0]  # Phase channel
        rfp = batch["anchor"][:, 1]  # RFP channel
        rfp = torch.max(rfp, dim=1).values
        Z = phase.shape[-3]
        phase = phase[:, Z // 2]
        img = torch.stack([phase, rfp], dim=1).to("cuda")

        # Extract features using OpenPhenom
        batch_features = openphenom.extract_features(img)
        features.append(batch_features.cpu())
        indices.append(batch["index"])

# %%
print("Processing features...")
pooled = torch.cat(features).numpy()
tracks = pd.concat([pd.DataFrame(idx) for idx in indices])
print("Computing PCA and PHATE...")
scaled_features = StandardScaler().fit_transform(pooled)
pca = PCA(n_components=8)
pca_features = pca.fit_transform(scaled_features)

phate_embedding = compute_phate(
    embeddings=pooled,
    n_components=2,
    knn=5,
    decay=40,
    n_jobs=15,
)
# %% Add features to dataframe
for i, feature in enumerate(pooled.T):
    tracks[f"feature_{i}"] = feature
# Add PCA features to dataframe
for i, feature in enumerate(pca_features.T):
    tracks[f"pca_{i}"] = feature
for i, feature in enumerate(phate_embedding.T):
    tracks[f"phate_{i}"] = feature

# %% Save the extracted features

output_path = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_07_A549_SEC61_ZIKV_DENV/4-phenotyping/figure/SEC61B/openphenom_pretrained_analysis"
)
output_path.mkdir(parents=True, exist_ok=True)
output_embeddings_file = (
    output_path / "openphenom_pretrained_features_SEC61B_n_Phase.csv"
)
print(f"Saving features to {output_embeddings_file}")
tracks.to_csv(output_embeddings_file, index=False)

# %%
