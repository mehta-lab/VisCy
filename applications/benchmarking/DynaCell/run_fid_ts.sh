# Generate nucleus embeddings separately
python fid_ts.py embed \
    -s /hpc/projects/virtual_staining/datasets/huang-lab/crops/mantis_figure_4.zarr/0/HIST2H2BE/0000010 \
    -t /hpc/projects/virtual_staining/datasets/huang-lab/crops/mantis_figure_4.zarr/0/HIST2H2BE/0000010 \
    -sc Nuclei-prediction \
    -tc Organelle \
    -c /hpc/projects/virtual_staining/models/huang-lab/fid/nucleus_vae_ts.pt \
    -so nuclei_prediction_embeddings.zarr \
    -to organelle_embeddings.zarr \
    -b 4 \
    -d cuda

# Generate membrane embeddings separately  
python fid_ts.py embed \
    -s /hpc/projects/virtual_staining/datasets/huang-lab/crops/mantis_figure_4.zarr/0/HIST2H2BE/0000010 \
    -t /hpc/projects/virtual_staining/datasets/huang-lab/crops/mantis_figure_4.zarr/0/HIST2H2BE/0000010 \
    -sc Membrane-prediction \
    -tc Membrane \
    -c /hpc/projects/virtual_staining/models/huang-lab/fid/membrane_vae_ts.pt \
    -so membrane_prediction_embeddings.zarr \
    -to membrane_embeddings.zarr \
    -b 4 \
    -d cuda

# Compute FID from separate embedding files
python fid_ts.py compute-fid \
    -sp nuclei_prediction_embeddings.zarr \
    -tp organelle_embeddings.zarr

python fid_ts.py compute-fid \
    -sp membrane_prediction_embeddings.zarr \
    -tp membrane_embeddings.zarr