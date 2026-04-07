# %% code for organelle and nuclear segmentation and feature extraction

from iohub import open_ome_zarr
import numpy as np
# from iohub.ngff.utils import create_empty_plate
from skimage.filters import gabor
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, hog, canny
import pywt
from skimage.transform import resize
from skimage.util import img_as_float
from scipy.ndimage import convolve
import pandas as pd
from matplotlib import pyplot as plt
from skimage.measure import regionprops
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os
import seaborn as sns
from scipy.ndimage import label

# %%

input_path = '/hpc/projects/intracellular_dashboard/organelle_dynamics/2025_07_22_A549_SEC61_TOMM20_G3BP1_ZIKV/4-phenotyping/train-test/2025_07_22_A549_SEC61_TOMM20_G3BP1_ZIKV.zarr'
input_zarr = open_ome_zarr(input_path, mode='r+', layout='hcs')
in_chans = input_zarr.channel_names
Organelle_chan = 'raw GFP EX488 EM525-45'

org_seg_path = '/hpc/projects/intracellular_dashboard/organelle_dynamics/2025_07_22_A549_SEC61_TOMM20_G3BP1_ZIKV/4-phenotyping/predictions/quantify_remodeling/G3BP1/segmentation_G3BP1_puncta.zarr'
seg_zarr = open_ome_zarr(org_seg_path, mode='r+', layout='hcs')
seg_chans = seg_zarr.channel_names
Organelle_seg = 'Organelle_mask'

nucl_seg_path = '/hpc/projects/intracellular_dashboard/organelle_dynamics/2025_07_22_A549_SEC61_TOMM20_G3BP1_ZIKV/1-preprocess/label-free/3-track/2025_07_22_A549_SEC61_TOMM20_G3BP1_ZIKV_cropped.zarr'
nucl_seg_zarr = open_ome_zarr(nucl_seg_path, mode='r+', layout='hcs')
nucl_seg_chans = nucl_seg_zarr.channel_names
Nucl_seg = 'nuclei_prediction_labels_labels'

# %% library of feature extraction methods

def get_patch(data, cell_centroid, patch_size):
    
    x_centroid, y_centroid = cell_centroid
    # ensure patch boundaries stay within data dimensions
    x_start = max(0, x_centroid - patch_size//2)
    x_end = min(data.shape[1], x_centroid + patch_size//2)
    y_start = max(0, y_centroid - patch_size//2)
    y_end = min(data.shape[0], y_centroid + patch_size//2)
    
    # get patch of patch_size centered on centroid
    patch = data[int(y_start):int(y_end),int(x_start):int(x_end)]
    return patch

def compute_glycomatrix(data):

    # convert to uint8 for GLCM computation
    norm_data = ((data-np.min(data))/(np.max(data)-np.min(data)) * 255).astype(np.uint8)
    glcm = graycomatrix(norm_data, distances=[1, 2, 3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
    contrast = np.mean(graycoprops(glcm, 'contrast'))
    homogeneity = np.mean(graycoprops(glcm, 'homogeneity'))
    energy = np.mean(graycoprops(glcm, 'energy'))
    correlation = np.mean(graycoprops(glcm, 'correlation'))
    return contrast, homogeneity, energy, correlation

def compute_edge_density(data):
    
    # Convert to float for skimage.canny
    norm_data = ((data-np.min(data))/(np.max(data)-np.min(data)) * 255).astype(np.uint8)
    norm_data_float = img_as_float(norm_data)

    # Apply Canny edge detection
    edges = canny(norm_data_float, sigma=1.0)  # sigma controls the scale of detection

    # Edge density: ratio of edge pixels to total pixels
    edge_density = np.sum(edges) / edges.size

    return edge_density


def compute_organelle_frac_int(org_data, org_mask):
    org_cell_mask = org_mask
    org_masked = org_data * org_mask
    organelle_volume = np.sum(org_cell_mask)
    organelle_intensity = np.mean(org_masked)
    # change the mask to instance level segmentation mask using connected components
    org_mask_labeled, num_organelles = label(org_cell_mask)
    # average of size of each organelle
    size_organelle = []
    for i in range(1, num_organelles + 1):
        size_organelle.append(np.sum(org_mask_labeled == i))
    size_organelle = np.mean(size_organelle)
    return organelle_volume, organelle_intensity, num_organelles, size_organelle

# %%

feature_list = pd.DataFrame()
patch_size = 192
    
for well_id, well_data in seg_zarr.wells():
# well_id, well_data = next(seg_zarr.wells())
    well_name, well_no = well_id.split('/')
    if well_name == 'C':
        for pos_id, pos_data in well_data.positions():
            # pos_id, pos_data = next(well_data.positions())
            T, C, Z, Y, X = pos_data.data.shape
            seg_data = pos_data.data.numpy()
            org_seg_mask = seg_data[:,seg_chans.index(Organelle_seg)]
            
            # read the csv stored in each nucl seg zarr folder
            file_name = 'tracks_' + well_name + '_' + well_no + '_' + pos_id + '.csv'
            nucl_seg_csv = os.path.join(nucl_seg_path, well_id+'/'+pos_id+'/'+file_name)
            nucl_seg_df = pd.read_csv(nucl_seg_csv)

            in_data = input_zarr[well_id+'/'+pos_id].data.numpy()
            organelle_data = in_data[:,in_chans.index(Organelle_chan)]
            
            # Initialize an empty list to store values from each row of the csv
            position_features = []
            for idx,row in nucl_seg_df.iterrows():
                row = nucl_seg_df.iloc[idx]
                    
                if row['x'] > patch_size//2 and row['y'] > patch_size//2 and row['x'] < X - patch_size//2 and row['y'] < Y - patch_size//2:
                    cell_centroid = row['x'], row['y']

                    timepoint = row['t']
                    organelle_patch = get_patch(organelle_data[int(timepoint),0], cell_centroid, patch_size)
                    org_seg_patch = get_patch(org_seg_mask[int(timepoint),0], cell_centroid, patch_size)
                
                    # compute features from image patches around nuclear centroid
                    contrast, homogeneity, energy, correlation = compute_glycomatrix(organelle_patch)
                    edge_density = compute_edge_density(organelle_patch)

                    # compute organelle morphology features using organelle segmentation
                    organelle_volume, organelle_intensity, no_organelles, size_organelle = compute_organelle_frac_int(organelle_patch, org_seg_patch)

                    # Create a dictionary of features for this cell
                    cell_features = {
                        'fov_name': '/'+well_id+'/'+pos_id,
                        'track_id': row['track_id'],
                        'time_point': timepoint,
                        'x': row['x'],
                        'y': row['y'],
                        'contrast': contrast,
                        'homogeneity': homogeneity,
                        'energy': energy,
                        'correlation': correlation,
                        'edge_density': edge_density,
                        'organelle_volume': organelle_volume,
                        'organelle_intensity': organelle_intensity,
                        'no_organelles': no_organelles,
                        'size_organelle': size_organelle,
                    }
                    position_features.append(cell_features)

            # After processing all cells in this position, write to CSV
            if position_features:
                # Convert the list of dictionaries to a DataFrame
                position_df = pd.DataFrame(position_features)
                
                # Define the output file path
                output_file = '/hpc/projects/intracellular_dashboard/organelle_dynamics/2025_07_22_A549_SEC61_TOMM20_G3BP1_ZIKV/4-phenotyping/predictions/quantify_remodeling/G3BP1/feature_list_G3BP1_2025_07_22_192patch.csv'
                
                # Write to CSV - append if file exists, create new if it doesn't
                position_df.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)
                
                # Clear the list for the next position
                position_features = []

                print(f"Processed position {pos_id} in well {well_id}")

# %% 
