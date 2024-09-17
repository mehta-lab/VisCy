# %% Importing Necessary Libraries
from pathlib import Path


import matplotlib.pyplot as plt
import pandas as pd


from viscy.representation.embedding_writer import read_embedding_dataset


# %%
def load_classification_annotation(classification_csv_path):
   classification_df = pd.read_csv(classification_csv_path)
   return classification_df


# %%
def count_infected_cell_states_over_time(embedding_dataset, classification_df):
   # Convert the embedding dataset to a DataFrame
   df = pd.DataFrame({
       "fov_name": embedding_dataset["fov_name"].values,
       "track_id": embedding_dataset["track_id"].values,
       "t": embedding_dataset["t"].values,
       "id": embedding_dataset["id"].values
   })
  
   df = pd.merge(df, classification_df[['id', 'fov_name', 'Predicted_Label']], on=['fov_name', 'id'], how='left')


   # Filter by time range (3 HPI to 30 HPI)
   df = df[(df['t'] >= 3) & (df['t'] <= 27)]
  
   df['well_type'] = df['fov_name'].apply(lambda x: 'Mock' if '/A/3' in x or '/B/3' in x else 'Dengue')
  
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
   for well_type in ['Mock', 'Dengue']:
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
# Prediction
feb_features_path = Path("/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/time_sampling_strategies/time_interval/predict/feb_test_time_interval_1_epoch_178.zarr")
feb_embedding_dataset = read_embedding_dataset(feb_features_path)


classification_csv_path = "/hpc/mydata/alishba.imran/VisCy/applications/contrastive_phenotyping/evaluation/predicted_labels.csv"  # Path to CSV file
classification_df = load_classification_annotation(classification_csv_path)


state_counts = count_infected_cell_states_over_time(feb_embedding_dataset, classification_df)
print(state_counts.head())
state_counts.info()


# %% Plot Infected Cell States Over Time as Percentage
plot_infected_cell_states(state_counts)


# %%
