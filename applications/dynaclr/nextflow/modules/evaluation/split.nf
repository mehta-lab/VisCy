// Split combined embeddings.zarr into per-experiment zarrs.
// Also generates configs/viewer.yaml using the cell index parquet.
// Emits per-experiment zarr paths as a list.

process SPLIT {
    executor 'local'

    input:
    val predict_done   // dependency signal from PREDICT
    val embeddings_dir
    val cell_index_path
    val output_dir
    val workspace_dir

    output:
    path 'zarr_paths.txt', emit: zarr_paths_file

    script:
    def combined_zarr = "${embeddings_dir}/embeddings.zarr"
    def viewer_yaml   = "${output_dir}/configs/viewer.yaml"
    """
    uv run --project=${workspace_dir} --package=dynaclr \
        dynaclr split-embeddings \
        --input ${combined_zarr} \
        --output-dir ${embeddings_dir}

    # Generate viewer YAML from cell index parquet
    uv run --project=${workspace_dir} --package=dynaclr python3 -c "
import pandas as pd, yaml, pathlib
embeddings_dir = pathlib.Path('${embeddings_dir}')
df = pd.read_parquet('${cell_index_path}', columns=['experiment', 'store_path'])
exp_to_plate = df.drop_duplicates('experiment').set_index('experiment')['store_path'].to_dict()
datasets = {}
for zarr_path in sorted(embeddings_dir.glob('*.zarr')):
    exp_name = zarr_path.stem
    datasets[exp_name] = {'hcs_plate': exp_to_plate[exp_name], 'anndata': str(zarr_path)}
with open('${viewer_yaml}', 'w') as f:
    yaml.dump({'datasets': datasets}, f, default_flow_style=False, sort_keys=False)
print('Viewer YAML written to ${viewer_yaml}')
"

    # Write per-experiment zarr paths to a file for Nextflow to read
    ls -d ${embeddings_dir}/*.zarr > zarr_paths.txt
    """
}
