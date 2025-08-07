python fid_ts.py \
    --data_path1 /hpc/projects/virtual_staining/datasets/huang-lab/prediction/a549/output.zarr \
    --data_path2 /hpc/projects/virtual_staining/datasets/huang-lab/prediction/a549/output.zarr \
    --channel_name1 Nuclei-prediction \
    --channel_name2 Nuclei-prediction \
    --loadcheck_path /hpc/projects/virtual_staining/models/huang-lab/fid/nucleus_vae_ts.pt \
    --batch_size 4 \
    --device cuda

python fid_ts.py \
    --data_path1 /hpc/projects/virtual_staining/datasets/huang-lab/prediction/a549/output.zarr \
    --data_path2 /hpc/projects/virtual_staining/datasets/huang-lab/prediction/a549/output.zarr \
    --channel_name1 Membrane-prediction \
    --channel_name2 Membrane-prediction \
    --loadcheck_path /hpc/projects/virtual_staining/models/huang-lab/fid/membrane_vae_ts.pt \
    --batch_size 4 \
    --device cuda