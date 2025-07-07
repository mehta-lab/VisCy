python applications/dynacell/fid.py \
    --data_path1 /hpc/projects/group.huang/dihan.zheng/CELL-Diff-VS/prediction/a549/output.zarr \
    --data_path2 /hpc/projects/group.huang/dihan.zheng/CELL-Diff-VS/prediction/a549/output.zarr \
    --channel_name1 Nuclei-prediction \
    --channel_name2 Nuclei-prediction \
    --loadcheck_path /hpc/projects/group.huang/dihan.zheng/CELL-Diff-VS/pretrain_cyto3d/PT_VAE-3D_nucleus_poisson_KL1e-3_LC2_ch-32-64-128-256/checkpoint-50000/pytorch_model.bin \
    --batch_size 4 \
    --device cuda