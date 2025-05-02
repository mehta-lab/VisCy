# %%

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from viscy.representation.embedding_writer import read_embedding_dataset
from viscy.representation.evaluation import dataset_of_tracks
from viscy.representation.evaluation.dimensionality_reduction import compute_phate

# %% Paths and parameters.

features_path = Path(
    "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/trainng_logs/SEC61/rev6_NTXent_sensorPhase_infection/2chan_160patch_94ckpt_rev6_2.zarr"
)

embedding_dataset = read_embedding_dataset(features_path)
embedding_dataset

features = embedding_dataset["features"]

# %% compute and store phate components

phate_embedding = compute_phate(
    embedding_dataset=embedding_dataset,
    n_components=2,
    knn=5,
    decay=40,
    n_jobs=15,
)

# %% OVERLAY INFECTION ANNOTATION

# load the infection state annotation
def load_annotation(da, path, name, categories: dict | None = None):
    annotation = pd.read_csv(path)
    annotation["fov_name"] = "/" + annotation["fov_name"]
    annotation = annotation.set_index(["fov_name", "id"])
    mi = pd.MultiIndex.from_arrays(
        [da["fov_name"].values, da["id"].values], names=["fov_name", "id"]
    )
    selected = annotation.reindex(mi)[name]
    if categories:
        selected = selected.astype("category").cat.rename_categories(categories)
    return selected
        
ann_root = Path("/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_02_04_A549_DENV_ZIKV_timelapse/8-train-test-split/supervised_inf_pred")

infection = load_annotation(
    features,
    ann_root / "extracted_inf_state.csv",
    "infection_state",
    {0:"background", 1: "uninfected", 2: "infected"},
)
data_path_Feb = Path("/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_02_04_A549_DENV_ZIKV_timelapse/8-train-test-split/registered_test.zarr")
tracks_path_Feb = Path("/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_02_04_A549_DENV_ZIKV_timelapse/8-train-test-split/track_test.zarr")

# %% filter background class from the data

infection_npy = infection.cat.codes.values
infection_npy_filtered = infection_npy[infection_npy != 0]

feature_npy = features.values
feature_npy_filtered = feature_npy[infection_npy != 0]

# add time and well info into dataframe
time_npy = features["t"].values
time_npy_filtered = time_npy[infection_npy != 0]

phate1_npy = phate_embedding[1][:,0]
phate1_npy_filtered = phate1_npy[infection_npy != 0]

phate2_npy = phate_embedding[1][:,1]
phate2_npy_filtered = phate2_npy[infection_npy != 0]

fov_name_list = features["fov_name"].values
fov_name_list_filtered = fov_name_list[infection_npy != 0]

track_id_list = features["track_id"].values
track_id_list_filtered = track_id_list[infection_npy != 0]

data = pd.DataFrame(
    {
        "infection": infection_npy_filtered,
        "time": time_npy_filtered,
        "fov_name": fov_name_list_filtered,
        "track_id": track_id_list_filtered,
        "PHATE1": phate1_npy_filtered,
        "PHATE2": phate2_npy_filtered,
    }
)
# Add all 768 features to the dataframe
feature_columns = pd.DataFrame(feature_npy_filtered, columns=[f"feature_{i+1}" for i in range(768)])
data = pd.concat([data, feature_columns], axis=1)

# %% plot phatemap of the data
colormap = {
    2: 'orange',
    1: 'steelblue',
}
# plot phatemap with infection prediction hue
fig = plt.figure(figsize=(10, 10))
plt.scatter(data["PHATE1"], data["PHATE2"], 
           c=data["infection"].map(colormap), 
           s=25,  
           edgecolor='white', linewidth=0.5)
plt.savefig(
    "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/arXiv_rev2/infection/Phate_Feb_sensor_infection_phatemap_timeawarentxent.png",
    format="png",
    dpi=300,
    bbox_inches='tight'
)

# %% plot phatemap of the data with cell tracks overlayed

infected_fov = '/B/4/9'
infected_track = 42
uninfected_fov = '/A/3/9'
uninfected_track = 19 # or 23


cell_uninfected = data[
    (data["fov_name"] == uninfected_fov) &
    (data["track_id"] == uninfected_track)
][["PHATE1", "PHATE2"]].reset_index(drop=True).iloc[::2]

cell_infected = data[
    (data["fov_name"] == infected_fov) &
    (data["track_id"] == infected_track)
][["PHATE1", "PHATE2"]].reset_index(drop=True).iloc[::2]


sns.scatterplot(
    x=data["PHATE1"],
    y=data["PHATE2"],
    hue=data["infection"],
    palette={1: "steelblue", 2: "orange"},
    s=7,
    alpha=0.5,
)

from matplotlib.patches import FancyArrowPatch
def add_arrows(df, color):
    for i in range(df.shape[0] - 1):
        start = df.iloc[i]
        end = df.iloc[i + 1]
        arrow = FancyArrowPatch(
            (start["PHATE1"], start["PHATE2"]),
            (end["PHATE1"], end["PHATE2"]),
            color=color,
            arrowstyle="-",
            mutation_scale=5,  # reduce the size of arrowhead by half
            lw=1,
            shrinkA=0,
            shrinkB=0,
        )
        plt.gca().add_patch(arrow)

# Apply arrows to the trajectories
add_arrows(cell_uninfected, color="blue")
add_arrows(cell_infected, color="red")

plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.xlabel("PHATE1", fontsize=14)
plt.ylabel("PHATE2", fontsize=14)
plt.legend([])

plt.savefig("/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/arXiv_rev2/infection/Phate_Feb_sensor_infection_phatemap_timeawarentxent_track.png", dpi=300)


# %% compute PCA components

color_map = {
    2: 'red',
    1: 'blue',
}

pca = PCA(n_components=2)
pca.fit(data.drop(columns=["infection", "fov_name", "track_id", "time", "PHATE1", "PHATE2"]))
pca_embedding = pca.transform(data.drop(columns=["infection", "fov_name", "track_id", "time", "PHATE1", "PHATE2"]))

# plot the PCA components with infection state
fig = plt.figure(figsize=(10, 8))
scatter = plt.scatter(pca_embedding[:,0], pca_embedding[:,1], 
           c=data["infection"].map(color_map),
           s=25,  
           edgecolor='white',
           linewidth=0.5)

# Add legend to group points
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                            markerfacecolor=color, label=label, markersize=10,
                            markeredgecolor='white', markeredgewidth=0.5)
                  for label, color in color_map.items()]
plt.legend([])
plt.xlabel("PCA component 1")
plt.ylabel("PCA component 2")
plt.title("PCA of cell features colored by infection state")
plt.savefig(
    "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/arXiv_rev2/infection/PCA_sensor_infection.svg",
    format="svg",
)
# save as png
plt.savefig(
    "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/arXiv_rev2/infection/PCA_sensor_infection.png",
    format="png",
    dpi=300,
)

# %% manually data the dataset into training and testing set by well name

# dataframe for training set, fov names starts with "/B/4/6" or "/B/4/7" or "/A/3/"
data_train_val = data[
    data["fov_name"].str.contains("/B/4/6")
    | data["fov_name"].str.contains("/B/4/7")
    | data["fov_name"].str.contains("/A/3/")
]

# dataframe for testing set, fov names starts with "/B/4/8" or "/B/4/9" or "/A/4/"
data_test = data[
    data["fov_name"].str.contains("/B/4/8")
    | data["fov_name"].str.contains("/B/4/9")
    | data["fov_name"].str.contains("/B/3/")
]

# %% train a linear classifier to predict infection state from PCA components

x_train = data_train_val.drop(
    columns=[
        "infection",
        "fov_name",
        "track_id",
        "time",
        "PHATE1",
        "PHATE2",
    ]
)
y_train = data_train_val["infection"]

# train a logistic regression model
clf = LogisticRegression(random_state=0).fit(x_train, y_train)

# test the trained classifer on the other half of the data

x_test = data_test.drop(
    columns=[
        "infection",
        "fov_name",
        "track_id",
        "time",
        "PHATE1",
        "PHATE2",
    ]
)
x_data = data.drop(
    columns=[
        "infection",
        "fov_name",
        "track_id",
        "time",
        "PHATE1",
        "PHATE2",
    ]
)
y_test = data_test["infection"]

# predict the infection state for the testing set
y_pred = clf.predict(x_test)
data_pred = clf.predict(x_data)

# %% use the trained classifier to perform prediction on the entire dataset

data_test["predicted_infection"] = y_pred
data["predicted_infection"] = data_pred

# plot the predicted infection state over time for /B/3 well and /B/4 well
time_points_test = np.unique(data_test["time"])

infected_test_cntrl = []
infected_test_infected = []

for time in time_points_test:
    infected_cell = data_test[
        (data_test["fov_name"].str.startswith("/B/3"))
        & (data_test["time"] == time)
        & (data_test["predicted_infection"] == 2)
    ].shape[0]
    total_cell = data_test[
        (data_test["fov_name"].str.startswith("/B/3")) & (data_test["time"] == time)
    ].shape[0]
    infected_test_cntrl.append(infected_cell * 100 / total_cell)
    infected_cell = data_test[
        (data_test["fov_name"].str.startswith("/B/4"))
        & (data_test["time"] == time)
        & (data_test["predicted_infection"] == 2)
    ].shape[0]
    total_cell = data_test[
        (data_test["fov_name"].str.startswith("/B/4")) & (data_test["time"] == time)
    ].shape[0]
    infected_test_infected.append(infected_cell * 100 / total_cell)


infected_true_cntrl = []
infected_true_infected = []

for time in time_points_test:
    infected_cell = data_test[
        (data_test["fov_name"].str.startswith("/B/3"))
        & (data_test["time"] == time)
        & (data_test["infection"] == 2)
    ].shape[0]
    total_cell = data_test[
        (data_test["fov_name"].str.startswith("/B/3")) & (data_test["time"] == time)
    ].shape[0]
    infected_true_cntrl.append(infected_cell * 100 / total_cell)
    infected_cell = data_test[
        (data_test["fov_name"].str.startswith("/B/4"))
        & (data_test["time"] == time)
        & (data_test["infection"] == 2)
    ].shape[0]
    total_cell = data_test[
        (data_test["fov_name"].str.startswith("/B/4")) & (data_test["time"] == time)
    ].shape[0]
    infected_true_infected.append(infected_cell * 100 / total_cell)


# %% perform prediction on the mantis dataset

#  Paths and parameters.
features_path = Path(
    "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/trainng_logs/SEC61/2024_11_07_NTXent_phase_sensor/sensor_phase_160patch_98ckpt_rev5.zarr"
)
data_path_mantis = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_07_A549_SEC61_ZIKV_DENV/2-assemble/2024_11_07_A549_SEC61_ZIKV_DENV.zarr"
)
tracks_path_mantis = Path(
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/2024_11_07_A549_SEC61_ZIKV_DENV/1-preprocess/label-free/4-track-gt/2024_11_07_A549_SEC61_ZIKV_DENV_2_cropped.zarr"
)

embedding_dataset = read_embedding_dataset(features_path)
embedding_dataset

mantis_features = embedding_dataset["features"]

# %% plot mantis and Feb test combined UMAP

# add time and well info into dataframe
mantis_time_npy = mantis_features["t"].values
mantis_fov_npy = mantis_features["fov_name"].values
mantis_track_id_npy = mantis_features["track_id"].values
mantis_phate1_npy = mantis_features["PHATE1"].values
mantis_phate2_npy = mantis_features["PHATE2"].values

mantis_data = pd.DataFrame(
    {
        "time": mantis_time_npy,
        "fov_name": mantis_fov_npy,
        "track_id": mantis_track_id_npy,
        "PHATE1": mantis_phate1_npy,
        "PHATE2": mantis_phate2_npy,
    }
)

# Add all 768 features to the dataframe
mantis_features_npy = mantis_features.values
for i in range(768):
    mantis_data[f"feature_{i+1}"] = mantis_features_npy[:, i]

# use wells with mCherry sensor only
mantis_data = mantis_data[
    mantis_data["fov_name"].str.contains("/B/3")
    | mantis_data["fov_name"].str.contains("/C/2")
]

# add the predicted infection state
mantis_pred = clf.predict(
    mantis_data.drop(
        columns=[
            "fov_name",
            "time",
            "track_id",
            "PHATE1",
            "PHATE2",
        ]
    )
)
mantis_data["predicted_infection"] = mantis_pred

# %% plot % infected over time

time_points_mantis = np.unique(mantis_data["time"])

infected_mantis_cntrl = []
infected_infected = []
mock_wells = '/B/3' # Create regex pattern for mock wells
infected_wells = '/C/2'  # Create regex pattern for dengue infected wells

for time in time_points_mantis:
    infected_mantis = mantis_data[
        (mantis_data["fov_name"].str.contains(mock_wells))
        & (mantis_data["time"] == time)
        & (mantis_data["predicted_infection"] == 2)
    ].shape[0]
    total_mantis = mantis_data[
        (mantis_data["fov_name"].str.contains(mock_wells)) & (mantis_data["time"] == time)
    ].shape[0]
    if total_mantis!=0:
        infected_mantis_cntrl.append(infected_mantis * 100 / total_mantis)
    else:
        infected_mantis_cntrl.append(0)

    infected_mantis = mantis_data[
        (mantis_data["fov_name"].str.contains(infected_wells))
        & (mantis_data["time"] == time)
        & (mantis_data["predicted_infection"] == 2)
    ].shape[0]
    total_mantis = mantis_data[
        (mantis_data["fov_name"].str.contains(infected_wells)) & (mantis_data["time"] == time)
    ].shape[0]
    if total_mantis!=0:
        infected_infected.append(infected_mantis * 100 / total_mantis)
    else:
        infected_infected.append(0)


# %% plot infected percentage over time for both wells
plt.plot(
    time_points_test * 0.5 + 3,
    infected_true_cntrl,
    label="mock true",
    color="steelblue",
    linestyle="--",
)
plt.plot(
    time_points_test * 0.5 + 3,
    infected_test_cntrl,
    label="mock predicted",
    color="blue",
    marker="+",
)
plt.plot(
    time_points_test * 0.5 + 3,
    infected_true_infected,
    label="MOI true",
    color="orange",
    linestyle="--",
)
plt.plot(
    time_points_test * 0.5 + 3,
    infected_test_infected,
    label="MOI predicted",
    color="red",
    marker="+",
)
plt.plot(
    time_points_mantis * 0.167 + 4,
    infected_mantis_cntrl,
    label="mock new predicted",
    color="blue",
    marker="o",
)
plt.plot(
    time_points_mantis * 0.167 + 4,
    infected_infected,
    label="MOI infected predicted",
    color="red",
    marker="o",
)
plt.xlabel("HPI")
plt.ylabel("Infected percentage")
plt.legend()
plt.savefig(
    "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/arXiv_rev2/infection/infected_percentage_withmantis.svg",
    format="svg",
)

# %% appendix video for infection dynamics phatemap, Feb test data, colored by human revised annotation
# add image patches of one track to the phatemap

fov_name_mock = "/B/3/9"
track_id_mock = [[100]]
fov_name_inf = "/B/4/9"
track_id_inf = [[44]]
z_range = (24,29)
source_channel = ["Phase3D", "RFP"]

all_whole = []  # Initialize empty list to store arrays
for i, no_labels in enumerate(track_id_mock):
    prediction_dataset = dataset_of_tracks(
        data_path_Feb,
        tracks_path_Feb,
        [fov_name_mock],
        track_id_mock[i],
        z_range=z_range,
        source_channel=source_channel,
        initial_yx_patch_size=(128,128),
        final_yx_patch_size=(128,128),
    )
    whole = np.stack([p["anchor"] for p in prediction_dataset])
    all_whole.append(whole)  # Add current whole array to list

# Concatenate all arrays along first dimension
whole_combined_mock = np.concatenate(all_whole, axis=0)

phase_mock = whole_combined_mock[:, 0, 2]
fluor_mock = np.max(whole_combined_mock[:, 1], axis=1)

all_whole = []  # Initialize empty list to store arrays
for i, no_labels in enumerate(track_id_inf):
    prediction_dataset = dataset_of_tracks(
        data_path_Feb,
        tracks_path_Feb,
        [fov_name_inf],
        track_id_inf[i],
        z_range=z_range,
        source_channel=source_channel,
        initial_yx_patch_size=(128,128),
        final_yx_patch_size=(128,128),
    )
    whole = np.stack([p["anchor"] for p in prediction_dataset])
    all_whole.append(whole)  # Add current whole array to list

# Concatenate all arrays along first dimension
whole_combined_inf = np.concatenate(all_whole, axis=0)

phase_inf = whole_combined_inf[:, 0, 2]
fluor_inf = np.max(whole_combined_inf[:, 1], axis=1)

for time in range(48):
    plt.clf()
    # Create figure with 1x3 layout (scatterplot on left, 2x2 image grid on right)
    fig = plt.figure(figsize=(16, 8))  # Reduced overall width from 20 to 16
    gs = fig.add_gridspec(2, 3, width_ratios=[2, 1, 1], height_ratios=[1, 1], 
                         wspace=0.05,  # Reduced horizontal spacing between subplots
                         hspace=0.2)   # Reduced vertical spacing between subplots
    
    # PHATE scatter plot (spans both rows on the left)
    ax1 = fig.add_subplot(gs[:, 0])
    # Background scatter plot with more transparency
    sns.scatterplot(
        data=data_test[(data_test["time"] == time)],
        x="PHATE1",
        y="PHATE2",
        hue="infection",
        palette={1: "steelblue", 2: "orange"},
        hue_order=[1, 2],
        s=80,
        alpha=0.3,  # Increased transparency for background points
        ax=ax1
    )
    
    # Highlight tracks with larger markers
    for fov_name, track_ids in [(fov_name_mock, track_id_mock), (fov_name_inf, track_id_inf)]:
        highlight_data = data_test[
            (data_test["time"] == time) & 
            (data_test["fov_name"] == fov_name) & 
            (data_test["track_id"].isin([item for sublist in track_ids for item in sublist] if isinstance(track_ids[0], list) else track_ids))
        ]
        if not highlight_data.empty:
            # Use the same color as the original points but larger size
            ax1.scatter(
                highlight_data["PHATE1"],
                highlight_data["PHATE2"],
                s=400,  # Increased size
                c=["steelblue" if pred == 1 else "orange" for pred in highlight_data["infection"]],
                alpha=1.0,  # Full opacity for highlighted points
                edgecolor="black",
                linewidth=2,
                zorder=5
            )
    
    handles, _ = ax1.get_legend_handles_labels()
    ax1.legend(handles=handles, labels=["uninfected", "infected"], fontsize=20)
    ax1.set_ylim(-0.05, 0.05)
    ax1.set_xlim(-0.05, 0.05)
    ax1.tick_params(axis='both', labelsize=18)
    ax1.set_xlabel('PHATE1', fontsize=20)
    ax1.set_ylabel('PHATE2', fontsize=20)
    
    # Mock condition images
    # Phase image (top middle)
    ax2 = fig.add_subplot(gs[0, 1])
    if time < len(phase_mock):
        ax2.imshow(phase_mock[time], cmap='gray')
    ax2.set_title('Mock Phase', fontsize=20, pad=5)  # Reduced padding
    ax2.axis('off')
    
    # Fluorescence image (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    if time < len(fluor_mock):
        ax3.imshow(fluor_mock[time], cmap='gray')
    ax3.set_title('Mock sensor', fontsize=20, pad=5)  # Reduced padding
    ax3.axis('off')
    
    # Infected condition images
    # Phase image (bottom middle)
    ax4 = fig.add_subplot(gs[1, 1])
    if time < len(phase_inf):
        ax4.imshow(phase_inf[time], cmap='gray')
    ax4.set_title('Infected Phase', fontsize=20, pad=5)  # Reduced padding
    ax4.axis('off')
    
    # Fluorescence image (bottom right)
    ax5 = fig.add_subplot(gs[1, 2])
    if time < len(fluor_inf):
        ax5.imshow(fluor_inf[time], cmap='gray')
    ax5.set_title('Infected sensor', fontsize=20, pad=5)  # Reduced padding
    ax5.axis('off')
    
    plt.suptitle(f"Time: {(time*0.5+3):.2f} HPI", fontsize=20)
    plt.tight_layout()
    plt.savefig(
        "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/arXiv_rev2/infection/video/Phate_Feb_true_wImages/phate_feb_true_infection_"
        + str(time).zfill(3)
        + ".png",
        format="png",
        dpi=300,
    )

# %% appendix video for infection dynamics umap, Feb test data, colored by predicted infection

for time in range(48):
    plt.clf()
    # Create figure with 1x3 layout (scatterplot on left, 2x2 image grid on right)
    fig = plt.figure(figsize=(16, 8))  # Reduced overall width from 20 to 16
    gs = fig.add_gridspec(2, 3, width_ratios=[2, 1, 1], height_ratios=[1, 1], 
                         wspace=0.05,  # Reduced horizontal spacing between subplots
                         hspace=0.2)   # Reduced vertical spacing between subplots
    
    # PHATE scatter plot (spans both rows on the left)
    ax1 = fig.add_subplot(gs[:, 0])
    # Background scatter plot with more transparency
    sns.scatterplot(
        data=data_test[(data_test["time"] == time)],
        x="PHATE1",
        y="PHATE2",
        hue="predicted_infection",
        palette={1: "blue", 2: "red"},
        hue_order=[1, 2],
        s=80,
        alpha=0.3,  # Increased transparency for background points
        ax=ax1
    )
    
    # Highlight tracks with larger markers
    for fov_name, track_ids in [(fov_name_mock, track_id_mock), (fov_name_inf, track_id_inf)]:
        highlight_data = data_test[
            (data_test["time"] == time) & 
            (data_test["fov_name"] == fov_name) & 
            (data_test["track_id"].isin([item for sublist in track_ids for item in sublist] if isinstance(track_ids[0], list) else track_ids))
        ]
        if not highlight_data.empty:
            # Use the same color as the original points but larger size
            ax1.scatter(
                highlight_data["PHATE1"],
                highlight_data["PHATE2"],
                s=400,  # Increased size
                c=["blue" if pred == 1 else "red" for pred in highlight_data["predicted_infection"]],
                alpha=1.0,  # Full opacity for highlighted points
                edgecolor="black",
                linewidth=2,
                zorder=5
            )
    
    handles, _ = ax1.get_legend_handles_labels()
    ax1.legend(handles=handles, labels=["uninfected", "infected"], fontsize=20)
    ax1.set_ylim(-0.05, 0.03)
    ax1.set_xlim(-0.05, 0.03)
    ax1.tick_params(axis='both', labelsize=18)
    ax1.set_xlabel('PHATE1', fontsize=20)
    ax1.set_ylabel('PHATE2', fontsize=20)
    
    # Mock condition images
    # Phase image (top middle)
    ax2 = fig.add_subplot(gs[0, 1])
    if time < len(phase_mock):
        ax2.imshow(phase_mock[time], cmap='gray')
    ax2.set_title('Mock Phase', fontsize=20, pad=5)  # Reduced padding
    ax2.axis('off')
    
    # Fluorescence image (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    if time < len(fluor_mock):
        ax3.imshow(fluor_mock[time], cmap='gray')
    ax3.set_title('Mock sensor', fontsize=20, pad=5)  # Reduced padding
    ax3.axis('off')
    
    # Infected condition images
    # Phase image (bottom middle)
    ax4 = fig.add_subplot(gs[1, 1])
    if time < len(phase_inf):
        ax4.imshow(phase_inf[time], cmap='gray')
    ax4.set_title('Infected Phase', fontsize=20, pad=5)  # Reduced padding
    ax4.axis('off')
    
    # Fluorescence image (bottom right)
    ax5 = fig.add_subplot(gs[1, 2])
    if time < len(fluor_inf):
        ax5.imshow(fluor_inf[time], cmap='gray')
    ax5.set_title('Infected sensor', fontsize=20, pad=5)  # Reduced padding
    ax5.axis('off')
    
    plt.suptitle(f"Time: {(time*0.5+3):.2f} HPI", fontsize=20)
    plt.tight_layout()
    plt.savefig(
        "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/arXiv_rev2/infection/video/Phate_Feb_test_wImages/phate_feb_predicted_infection_"
        + str(time).zfill(3)
        + ".png",
        format="png",
        dpi=300,
    )

# %% appendix video for infection dynamics umap, mantis data, colored by predicted infection
# add image patch to the side of the plot and highlight the scatter point in phatemap over time

fov_name_mock = "/B/3/000000"
track_id_mock = [[130],[131],[132]]
fov_name_inf = "/C/2/000000"
track_id_inf = [[110],[112],[114],[116]]
z_range = (16, 21)
source_channel = ["Phase3D", "raw GFP EX488 EM525-45"]

all_whole = []  # Initialize empty list to store arrays
for i, no_labels in enumerate(track_id_mock):
    prediction_dataset = dataset_of_tracks(
        data_path_mantis,
        tracks_path_mantis,
        [fov_name_mock],
        track_id_mock[i],
        z_range=z_range,
        source_channel=source_channel,
        initial_yx_patch_size=(192,192),
        final_yx_patch_size=(192,192),
    )
    whole = np.stack([p["anchor"] for p in prediction_dataset])
    all_whole.append(whole)  # Add current whole array to list

# Concatenate all arrays along first dimension
whole_combined_mock = np.concatenate(all_whole, axis=0)

phase_mock = whole_combined_mock[:, 0, 2]
fluor_mock = np.max(whole_combined_mock[:, 1], axis=1)

all_whole = []  # Initialize empty list to store arrays
for i, no_labels in enumerate(track_id_inf):
    prediction_dataset = dataset_of_tracks(
        data_path_mantis,
        tracks_path_mantis,
        [fov_name_inf],
        track_id_inf[i],
        z_range=z_range,
        source_channel=source_channel,
        initial_yx_patch_size=(192,192),
        final_yx_patch_size=(192,192),
    )
    whole = np.stack([p["anchor"] for p in prediction_dataset])
    all_whole.append(whole)  # Add current whole array to list

# Concatenate all arrays along first dimension
whole_combined_inf = np.concatenate(all_whole, axis=0)

phase_inf = whole_combined_inf[:, 0, 2]
fluor_inf = np.max(whole_combined_inf[:, 1], axis=1)

for time in range(len(time_points_mantis)):
    plt.clf()
    # Create figure with 1x3 layout (scatterplot on left, 2x2 image grid on right)
    fig = plt.figure(figsize=(16, 8))  # Reduced overall width from 20 to 16
    gs = fig.add_gridspec(2, 3, width_ratios=[2, 1, 1], height_ratios=[1, 1], 
                         wspace=0.05,  # Reduced horizontal spacing between subplots
                         hspace=0.2)   # Reduced vertical spacing between subplots
    
    # PHATE scatter plot (spans both rows on the left)
    ax1 = fig.add_subplot(gs[:, 0])
    # Background scatter plot with more transparency
    sns.scatterplot(
        data=mantis_data[(mantis_data["time"] == time)],
        x="PHATE1",
        y="PHATE2",
        hue="predicted_infection",
        palette={1: "blue", 2: "red"},
        hue_order=[1, 2],
        s=80,
        alpha=0.3,  # Increased transparency for background points
        ax=ax1
    )
    
    # Highlight tracks with larger markers
    for fov_name, track_ids in [(fov_name_mock, track_id_mock), (fov_name_inf, track_id_inf)]:
        highlight_data = mantis_data[
            (mantis_data["time"] == time) & 
            (mantis_data["fov_name"] == fov_name) & 
            (mantis_data["track_id"].isin([item for sublist in track_ids for item in sublist] if isinstance(track_ids[0], list) else track_ids))
        ]
        if not highlight_data.empty:
            # Use the same color as the original points but larger size
            ax1.scatter(
                highlight_data["PHATE1"],
                highlight_data["PHATE2"],
                s=400,  # Increased size
                c=["blue" if pred == 1 else "red" for pred in highlight_data["predicted_infection"]],
                alpha=1.0,  # Full opacity for highlighted points
                edgecolor="black",
                linewidth=2,
                zorder=5
            )
    
    handles, _ = ax1.get_legend_handles_labels()
    ax1.legend(handles=handles, labels=["uninfected", "infected"], fontsize=20)
    ax1.set_ylim(-0.04, 0.04)
    ax1.set_xlim(-0.04, 0.04)
    ax1.tick_params(axis='both', labelsize=18)
    ax1.set_xlabel('PHATE1', fontsize=20)
    ax1.set_ylabel('PHATE2', fontsize=20)
    
    # Mock condition images
    # Phase image (top middle)
    ax2 = fig.add_subplot(gs[0, 1])
    if time < len(phase_mock):
        ax2.imshow(phase_mock[time], cmap='gray')
    ax2.set_title('Mock Phase', fontsize=20, pad=5)  # Reduced padding
    ax2.axis('off')
    
    # Fluorescence image (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    if time < len(fluor_mock):
        ax3.imshow(fluor_mock[time], cmap='gray')
    ax3.set_title('Mock SEC61', fontsize=20, pad=5)  # Reduced padding
    ax3.axis('off')
    
    # Infected condition images
    # Phase image (bottom middle)
    ax4 = fig.add_subplot(gs[1, 1])
    if time < len(phase_inf):
        ax4.imshow(phase_inf[time], cmap='gray')
    ax4.set_title('Infected Phase', fontsize=20, pad=5)  # Reduced padding
    ax4.axis('off')
    
    # Fluorescence image (bottom right)
    ax5 = fig.add_subplot(gs[1, 2])
    if time < len(fluor_inf):
        ax5.imshow(fluor_inf[time], cmap='gray')
    ax5.set_title('Infected SEC61', fontsize=20, pad=5)  # Reduced padding
    ax5.axis('off')
    
    plt.suptitle(f"Time: {(time*0.167+4):.2f} HPI", fontsize=20)
    plt.tight_layout()
    plt.savefig(
        "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/arXiv_rev2/infection/video/Phate_Mantis_wImage/phate_mantis_predicted_infection_"
        + str(time).zfill(3)
        + ".png",
        format="png",
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()

# %% plot phatemaps for Feb data and mantis 10 minute data with infection prediction hue
colormap = {1: "blue", 2: "red"}
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(1, 1, 1)
ax.scatter(data["PHATE1"], data["PHATE2"], 
           c=data["predicted_infection"].map(colormap), 
           s=25,  
           edgecolor='white',
           linewidth=0.5)
# Add legend to group points
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                            markerfacecolor=color, label=label, markersize=10,
                            markeredgecolor='white', markeredgewidth=0.5)
                  for label, color in color_map.items()]
ax.legend([])
ax.set_xlabel("PHATE1")
ax.set_ylabel("PHATE2")

plt.savefig(
    "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/arXiv_rev2/infection/Phate_Feb_sensor_infection.png",
    format="png",
    dpi=300,
)

# %% plot phatemaps for mantis data with infection prediction hue

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(1, 1, 1)
ax.scatter(mantis_data["PHATE1"], mantis_data["PHATE2"], 
           c=mantis_data["predicted_infection"].map(colormap), 
           s=25,  
           edgecolor='white',
           linewidth=0.5)
# Add legend to group points
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                            markerfacecolor=color, label=label, markersize=10,
                            markeredgecolor='white', markeredgewidth=0.5)
                  for label, color in color_map.items()]
ax.legend([])
ax.set_xlabel("PHATE1")
ax.set_ylabel("PHATE2")

plt.savefig(
    "/hpc/projects/comp.micro/infected_cell_imaging/Single_cell_phenotyping/ContrastiveLearning/Figure_panels/arXiv_rev2/infection/Phate_Mantis_infection.png",
    format="png",
    dpi=300,
)

