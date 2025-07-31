
#%%
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_segmentation_metrics(csv_path, metrics=['dice', 'jaccard', 'mAP_50'], 
                              cell_type=None, infection_condition=None, organelle=None):
    """
    Plot segmentation metrics over time with filtering options.
    
    Parameters
    ----------
    csv_path : str or Path
        Path to the CSV file containing segmentation metrics data
    metrics : list of str, default ['dice', 'jaccard', 'mAP_50']
        List of metric column names to plot from the CSV
    cell_type : str, optional
        Filter by cell type
    infection_condition : str, optional
        Filter by infection condition
    organelle : str, optional
        Filter by organelle
    """
    # Load data
    df = pd.read_csv(csv_path)
    df_clean = df.dropna()
    
    # Apply filters
    filtered_data = df_clean.copy()
    if cell_type:
        filtered_data = filtered_data[filtered_data['cell_type'] == cell_type]
    if infection_condition:
        filtered_data = filtered_data[filtered_data['infection_condition'] == infection_condition]
    if organelle:
        filtered_data = filtered_data[filtered_data['organelle'] == organelle]
    
    if filtered_data.empty:
        print("No data found with the specified filters.")
        return
    
    # Set up colorblind-friendly colors (blue and orange)
    plt.style.use('seaborn-v0_8')
    colors = ['#1f77b4', '#ff7f0e']  # Blue and orange
    sns.set_palette(colors)
    
    # Validate metrics exist in data
    available_metrics = [m for m in metrics if m in df_clean.columns]
    if not available_metrics:
        print(f"None of the requested metrics {metrics} found in data.")
        print(f"Available columns: {list(df_clean.columns)}")
        return
    
    # Get unique conditions for plotting
    conditions = filtered_data['infection_condition'].unique()
    
    # Create individual plots for each metric
    for metric in available_metrics:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot each condition
        for i, condition in enumerate(conditions):
            condition_data = filtered_data[filtered_data['infection_condition'] == condition]
            
            # Plot individual trajectories with transparency
            for pos in condition_data['position'].unique():
                pos_data = condition_data[condition_data['position'] == pos]
                ax.plot(pos_data['time'], pos_data[metric], 
                       alpha=0.3, linewidth=1, color=colors[i % len(colors)])
            
            # Plot mean trend line
            sns.lineplot(data=condition_data, x='time', y=metric, 
                        ax=ax, label=condition, linewidth=2, marker='o', 
                        markersize=4, color=colors[i % len(colors)])
        
        # Customize plot
        filter_info = []
        if cell_type:
            filter_info.append(f"Cell: {cell_type}")
        if infection_condition:
            filter_info.append(f"Condition: {infection_condition}")
        if organelle:
            filter_info.append(f"Organelle: {organelle}")
        
        title = f'{metric} Over Time'
        if filter_info:
            title += f" ({', '.join(filter_info)})"
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.grid(True, alpha=0.3)
        
        if len(conditions) > 1:
            ax.legend()
        
        # Set appropriate y-axis limits
        y_min = filtered_data[metric].min()
        y_max = filtered_data[metric].max()
        padding = (y_max - y_min) * 0.05 if y_max > y_min else 0.05
        ax.set_ylim(max(0, y_min - padding), min(1, y_max + padding))
        
        plt.tight_layout()
        
        # Save plot
        filter_suffix = ""
        if cell_type or infection_condition or organelle:
            filter_parts = []
            if cell_type:
                filter_parts.append(cell_type)
            if infection_condition:
                filter_parts.append(infection_condition)
            if organelle:
                filter_parts.append(organelle)
            filter_suffix = f"_{'_'.join(filter_parts)}"
        
        output_path = Path(csv_path).parent / f'{metric}_over_time{filter_suffix}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved {metric} plot to: {output_path}")
        
        plt.show()

# Usage
if __name__ == "__main__":
    
    # Example for segmentation metrics
    segmentation_csv = "/home/eduardo.hirata/repos/viscy/applications/DynaCell/metrics/segmentation_2025_04_17_A549_H2B_CAAX_DENV_membrane_only/20250731_131927/metrics.csv"
    
    # Plot default segmentation metrics (dice, jaccard, mAP_50)
    plot_segmentation_metrics(segmentation_csv)
    
    # Plot custom metrics with filters
    plot_segmentation_metrics(segmentation_csv, 
                             metrics=['dice', 'jaccard', 'mAP_50', 'accuracy'],
                             cell_type='A549',
                             infection_condition='DENV',
                             organelle='HIST2H2BE')
# %%
