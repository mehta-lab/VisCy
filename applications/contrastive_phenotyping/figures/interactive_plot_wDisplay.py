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
import atexit

from viscy.representation.embedding_writer import read_embedding_dataset
# from viscy.representation.evaluation import dataset_of_tracks
from viscy.data.triplet import TripletDataModule

# Initialize Dash app
app = dash.Dash(__name__)

# Sample DataFrame for demonstration
fov_name = "/0/6/000000"
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

# Filter data for FOVs starting with '/0/6/000000'
mask = features.coords['fov_name'].str.startswith(fov_name)
features = features.sel(sample=mask)

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
        html.Img(id="hover-image", src="", style={"width": "300px", "height": "150px"})
    ])
])

def normalize_image(img_array):
    """Normalize a single image array to [0, 255]"""
    img_array = np.clip(img_array, img_array.min(), img_array.max())
    img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min())
    return (img_array * 255).astype(np.uint8)

def numpy_to_base64(img_array):
    """Convert numpy array to base64 string"""
    # Ensure img_array is uint8 before creating Image
    if not isinstance(img_array, np.uint8):
        img_array = img_array.astype(np.uint8)
    img = Image.fromarray(img_array)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")

# Create an image cache before initializing the Dash app
image_cache = {}

def cleanup_cache():
    """Clear the image cache when the program exits"""
    print("Cleaning up image cache...")
    image_cache.clear()

# Register the cleanup function
atexit.register(cleanup_cache)

def preload_images(df):
    """Preload all images into memory"""
    print("Preloading images into cache...")
    
    # groups = df.groupby(['fov_name', 'track_id'])
    track_id_list = df['track_id'].unique().tolist()
    
    # for (fov_name, track_id), group in groups:

    data_module = TripletDataModule(
        data_path=data_path,
        tracks_path=tracks_path,
        include_fov_names=[fov_name],
        include_track_ids=track_id_list,
        source_channel=["Phase3D", "MultiCam_GFP_mCherry_BF-Prime BSI Express"],
        z_range=(31,36),
        initial_yx_patch_size=(128, 128),
        final_yx_patch_size=(128, 128),
        batch_size=1,
        num_workers=16,
        normalizations=None,
        predict_cells=True,
    )
    data_module.setup("predict")

    for batch in data_module.predict_dataloader():
        images = batch["anchor"].numpy()
        indices = batch["index"]
        track_id = indices["track_id"].tolist()
        t = indices["t"].tolist()
        # print(track_id, t)

        try:
            img = np.stack(images)
            cache_key = (fov_name, track_id[0], t[0])
            
            # Extract and normalize each channel independently
            channel1 = normalize_image(img[0, 0, 2])  # First channel at z=2
            channel2 = normalize_image(np.max(img[0,1], axis=0))  # Max projection of second channel
            
            # Ensure both channels are uint8
            channel1 = channel1.astype(np.uint8)
            channel2 = channel2.astype(np.uint8)
            
            # Concatenate the normalized channels horizontally
            combined_img = np.hstack((channel1, channel2))
            
            # Store the base64 string in the cache
            try:
                image_cache[cache_key] = numpy_to_base64(combined_img)
            except Exception as e:
                print(f"Error converting image to base64 for {cache_key}: {e}")
                continue
                
        except Exception as e:
            print(f"Error processing images for {fov_name}, {track_id}: {e}")
            continue
    
    print(f"Cached {len(image_cache)} images")

# Call preload before creating the Dash app
preload_images(df)

# Modify the callback to use cached images
@app.callback(
    dd.Output("hover-image", "src"),
    [dd.Input("scatter-plot", "hoverData")]
)
def update_image(hoverData):
    if hoverData is None:
        return ""

    # Extract the necessary information from hoverData
    fov_name = hoverData['points'][0]['hovertext']
    track_id = hoverData['points'][0]['customdata'][2]
    t = hoverData['points'][0]['customdata'][1]
    
    # Get image from cache
    cache_key = (fov_name, track_id, t)
    return image_cache.get(cache_key, "")

if __name__ == '__main__':
    app.run_server(debug=True)