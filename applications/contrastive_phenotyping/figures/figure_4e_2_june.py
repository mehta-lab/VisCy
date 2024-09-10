# %% Importing Necessary Libraries
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from viscy.representation.embedding_writer import read_embedding_dataset


# %% Function to Load Annotations from CSV
def load_annotation(csv_path):
    return pd.read_csv(csv_path)

# %% Function to Count and Calculate Percentage of Infected Cells Over Time Based on Predicted Labels
def count_infected_cell_states_over_time(embedding_dataset, prediction_df):
    # Convert the embedding dataset to a DataFrame
    df = pd.DataFrame({
        "fov_name": embedding_dataset["fov_name"].values,
        "track_id": embedding_dataset["track_id"].values,
        "t": embedding_dataset["t"].values,
        "id": embedding_dataset["id"].values
    })
    
    # Merge with the prediction data to add Predicted Labels
    df = pd.merge(df, prediction_df[['id', 'fov_name', 'Infection_Class']], on=['fov_name', 'id'], how='left')

    # Filter by time range (2 HPI to 50 HPI)
    df = df[(df['t'] >= 2) & (df['t'] <= 50)]
    
    # Determine the well type (Mock, Dengue, Zika) based on fov_name
    df['well_type'] = df['fov_name'].apply(
        lambda x: 'Mock' if '/0/1' in x or '/0/2' in x or '/0/3' in x or '/0/4' in x else 
                  ('Dengue' if '/0/5' in x or '/0/6' in x else 'Zika'))
    
    # Group by time, well type, and Predicted_Label to count the number of infected cells
    state_counts = df.groupby(['t', 'well_type', 'Infection_Class']).size().unstack(fill_value=0)
    
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
    for well_type in ['Mock', 'Dengue', 'Zika']:
        # Select the data for the current well type
        if well_type in state_counts.index.get_level_values('well_type'):
            well_data = state_counts.xs(well_type, level='well_type')
            
            # Plot only the percentage of infected cells
            if 'infected' in well_data.columns:
                plt.plot(well_data.index, well_data['infected'], label=f'{well_type} - Infected')

    plt.title("Percentage of Infected Cells Over Time - June")
    plt.xlabel("Hours Post Perturbation")
    plt.ylabel("Percentage of Infected Cells")
    plt.legend(title="Well Type")
    plt.grid(True)
    plt.show()

# %% Load and process June Dataset
june_features_path = Path("/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/code_testing_soorya/output/Phase_RFP_smallPatch_June/phaseRFP_36patch_June.zarr")
june_embedding_dataset = read_embedding_dataset(june_features_path)

# Load the predicted labels from CSV
prediction_csv_path = "3up_gmm_clustering_results_june_pca_6components.csv"  # Path to predicted labels CSV file
prediction_df = load_annotation(prediction_csv_path)

# %% Count Infected Cell States Over Time as Percentage using Predicted labels
state_counts = count_infected_cell_states_over_time(june_embedding_dataset, prediction_df)
print(state_counts.head())
state_counts.info()

# %% Plot Infected Cell States Over Time as Percentage
plot_infected_cell_states(state_counts)

# %%
