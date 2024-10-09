# %% Importing Necessary Libraries
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from viscy.representation.embedding_writer import read_embedding_dataset


# %% Function to Load Annotations from GMM CSV
def load_gmm_annotation(gmm_csv_path):
    gmm_df = pd.read_csv(gmm_csv_path)
    return gmm_df

# %% Function to Count and Calculate Percentage of Infected Cells Over Time Based on GMM Labels
def count_infected_cell_states_over_time(embedding_dataset, gmm_df):
    # Convert the embedding dataset to a DataFrame
    df = pd.DataFrame({
        "fov_name": embedding_dataset["fov_name"].values,
        "track_id": embedding_dataset["track_id"].values,
        "t": embedding_dataset["t"].values,
        "id": embedding_dataset["id"].values
    })
    
    # Merge with GMM data to add GMM labels
    df = pd.merge(df, gmm_df[['id', 'fov_name', 'Predicted_Label']], on=['fov_name', 'id'], how='left')

    # Filter by time range (3 HPI to 30 HPI)
    df = df[(df['t'] >= 3) & (df['t'] <= 27)]
    
    # Determine the well type (Mock, Zika, Dengue) based on fov_name
    df['well_type'] = df['fov_name'].apply(lambda x: 'Mock' if '/A/3' in x or '/B/3' in x else 
                                                     ('Zika' if '/A/4' in x else 'Dengue'))
    
    # Group by time, well type, and GMM label to count the number of infected cells
    state_counts = df.groupby(['t', 'well_type', 'Predicted_Label']).size().unstack(fill_value=0)
    
    # Ensure that 'infected' column exists
    if 'infected' not in state_counts.columns:
        state_counts['infected'] = 0
    
    # Calculate the percentage of infected cells
    state_counts['total'] = state_counts.sum(axis=1)
    state_counts['infected'] = (state_counts['infected'] / state_counts['total']) * 100
    
    return state_counts

# %% Function to Plot Percentage of Infected Cells Over Time
def plot_infected_cell_states(state_counts):
    plt.figure(figsize=(12, 8))

    # Loop through each well type
    for well_type in ['Mock', 'Zika', 'Dengue']:
        # Select the data for the current well type
        if well_type in state_counts.index.get_level_values('well_type'):
            well_data = state_counts.xs(well_type, level='well_type')
            
            # Plot only the percentage of infected cells
            if 'infected' in well_data.columns:
                plt.plot(well_data.index, well_data['infected'], label=f'{well_type} - Infected')

    plt.title("Percentage of Infected Cells Over Time - February")
    plt.xlabel("Hours Post Perturbation")
    plt.ylabel("Percentage of Infected Cells")
    plt.legend(title="Well Type")
    plt.grid(True)
    plt.show()

# %% Load and process Feb Dataset
feb_features_path = Path("/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/code_testing_soorya/output/June_140Patch_2chan/phaseRFP_140patch_99ckpt_Feb.zarr")
feb_embedding_dataset = read_embedding_dataset(feb_features_path)

# Load the GMM annotation CSV
gmm_csv_path = "june_logistic_regression_predicted_labels_feb_pca.csv"  # Path to CSV file
gmm_df = load_gmm_annotation(gmm_csv_path)

# %% Count Infected Cell States Over Time as Percentage using GMM labels
state_counts = count_infected_cell_states_over_time(feb_embedding_dataset, gmm_df)
print(state_counts.head())
state_counts.info()

# %% Plot Infected Cell States Over Time as Percentage
plot_infected_cell_states(state_counts)

# %%


