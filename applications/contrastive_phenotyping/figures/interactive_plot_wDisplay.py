
# This is a simple example of an interactive plot using Dash.
from pathlib import Path
import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import dash.dependencies as dd

from viscy.representation.embedding_writer import read_embedding_dataset
from viscy.representation.evaluation import dataset_of_tracks

# Initialize Dash app
app = dash.Dash(__name__)

# Sample DataFrame for demonstration
features_path = Path(
    "/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/time_sampling_strategies/time_interval/predict/jun_time_interval_1_epoch_178.zarr"
)
embedding_dataset = read_embedding_dataset(features_path)
features = embedding_dataset["features"]
scaled_features = StandardScaler().fit_transform(features.values)
pca = PCA(n_components=3)
embedding = pca.fit_transform(scaled_features)
features = (
    features.assign_coords(PCA1=("sample", embedding[:, 0]))
    .assign_coords(PCA2=("sample", embedding[:, 1]))
    .assign_coords(PCA3=("sample", embedding[:, 2]))
    .set_index(sample=["PCA1", "PCA2", "PCA3"], append=True)
)

df = pd.DataFrame({k: v for k, v in features.coords.items() if k != "features"})

# Image paths for each track and time

data_path = Path(
    "/hpc/projects/organelle_phenotyping/2024_06_13_SEC61_TOMM20_ZIKV_DENGUE_1/registered_chunked.zarr"
)
tracks_path = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_06_13_SEC61_TOMM20_ZIKV_DENGUE_1/4.2-tracking/track.zarr"
)

# Create scatter plot with hover data (track_id, t, fov_name)
fig = px.scatter(
    df,
    x="PCA1",
    y="PCA2",
    color="PCA1",
    hover_name="fov_name",
    hover_data=["id", "t", "track_id"],  # Include track_id and t for image lookup
)

# Layout of the app
app.layout = html.Div([
    dcc.Graph(
        id="scatter-plot",
        figure=fig,
    ),
    html.Div([
        html.Img(id="hover-image", src="", style={"width": "150px", "height": "150px"})
    ])
])

# Helper function to convert numpy array to base64 image
def numpy_to_base64(img_array):
    # Clip, normalize, and scale to the range [0, 255]
    img_array = np.clip(img_array, img_array.min(), img_array.max()) # Clip values to the expected range
    img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min())  # Normalize to [0, 1]
    img_array = (img_array * 255).astype(np.uint8)  # Scale to [0, 255] and convert to uint8
    
    img = Image.fromarray(img_array)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")


# Callback to update the image when a point is hovered over
@app.callback(
    dd.Output("hover-image", "src"),
    [dd.Input("scatter-plot", "hoverData")]
)
def update_image(hoverData):
    if hoverData is None:
        return ""  # Return empty if no hover

    # Extract the necessary information from hoverData
    fov_name = hoverData['points'][0]['hovertext']  # fov_name is in hovertext
    track_id = hoverData['points'][0]['customdata'][2]  # track_id from hover_data
    t = hoverData['points'][0]['customdata'][1]  # t from hover_data

    print(f"Hovering over: fov_name={fov_name}, track_id={track_id}, t={t}")

    # Lookup the image path based on fov_name, track_id, and t
    # image_key = (fov_name, track_id, t)

    # Get the image URL if it exists
    # return image_paths.get(image_key, "")  # Return empty string if no match
    source_channel = ["Phase3D"]
    z_range = (33,34)
    predict_dataset = dataset_of_tracks(
        data_path,
        tracks_path,
        [fov_name],
        [track_id],
        z_range=z_range,
        source_channel=source_channel,
    )
    # image_patch = np.stack([p["anchor"][0, 7].numpy() for p in predict_dataset])

    # Check if the dataset was retrieved successfully
    if not predict_dataset:
        print(f"No dataset found for fov_name={fov_name}, track_id={track_id}, t={t}")
        return ""  # Return empty if no dataset is found

    # Extract the image patch (assuming it's a numpy array)
    try:
        image_patch = np.stack([p["anchor"][0].numpy() for p in predict_dataset])
        image_patch = image_patch[0,0]
        print(f"Image patch shape: {image_patch.shape}")
    except Exception as e:
        print(f"Error extracting image patch: {e}")
        return ""

    # Check if the image is valid (this step is just a safety check)
    if image_patch.ndim != 2:
        print(f"Invalid image data: image_patch is not 2D.")
        return ""

    return numpy_to_base64(image_patch)

if __name__ == '__main__':
    app.run_server(debug=True)