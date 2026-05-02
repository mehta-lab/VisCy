// Per-experiment dimensionality reduction (PCA).
// Patches the __ZARR_PATH__ placeholder in the template YAML, then runs.
// Emits zarr_path for downstream processes.

process REDUCE {
    executor 'local'

    input:
    val zarr_path
    val reduce_yaml
    val workspace_dir

    output:
    val zarr_path, emit: zarr_path

    script:
    """
    python3 -c "
import yaml
with open('${reduce_yaml}') as f:
    cfg = yaml.safe_load(f)
cfg['input_path'] = '${zarr_path}'
with open('reduce_patched.yaml', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
"
    uv run --project=${workspace_dir} --package=dynaclr \
        dynaclr reduce-dimensionality -c reduce_patched.yaml
    """
}
