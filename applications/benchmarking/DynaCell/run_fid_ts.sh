# Generate nucleus embeddings separately
python fid_ts.py embed \
    -s /hpc/projects/virtual_staining/datasets/huang-lab/crops/mantis_figure_4.zarr/0/HIST2H2BE/0000010 \
    -t /hpc/projects/virtual_staining/datasets/huang-lab/crops/mantis_figure_4.zarr/0/HIST2H2BE/0000010 \
    -sc Nuclei-prediction \
    -tc Organelle \
    -c /hpc/projects/virtual_staining/models/huang-lab/fid/nucleus_vae_ts.pt \
    -o . \
    -b 4 \
    -d cuda

# Generate membrane embeddings separately  
python fid_ts.py embed \
    -s /hpc/projects/virtual_staining/datasets/huang-lab/crops/mantis_figure_4.zarr/0/HIST2H2BE/0000010 \
    -t /hpc/projects/virtual_staining/datasets/huang-lab/crops/mantis_figure_4.zarr/0/HIST2H2BE/0000010 \
    -sc Membrane-prediction \
    -tc Membrane \
    -c /hpc/projects/virtual_staining/models/huang-lab/fid/membrane_vae_ts.pt \
    -o . \
    -b 4 \
    -d cuda

# Compute FID from separate embedding files
python fid_ts.py compute-fid \
    -s _Nuclei-prediction.zarr \
    -t _Organelle.zarr

python fid_ts.py compute-fid \
    -s _Membrane-prediction.zarr \
    -t _Membrane.zarr