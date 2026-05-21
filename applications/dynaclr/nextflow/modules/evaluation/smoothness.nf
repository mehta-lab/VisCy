// Per-experiment temporal smoothness evaluation.
// Patches the __ZARR_PATH__ placeholder in the template YAML.

process SMOOTHNESS {
    label 'cpu'

    input:
    val zarr_path
    val smoothness_yaml
    val workspace_dir

    output:
    val zarr_path, emit: zarr_path

    script:
    """
    python3 -c "
import yaml
with open('${smoothness_yaml}') as f:
    cfg = yaml.safe_load(f)
cfg['models'][0]['path'] = '${zarr_path}'
with open('smoothness_patched.yaml', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
"
    uv run --project=${workspace_dir} --package=dynaclr \
        dynaclr evaluate-smoothness -c smoothness_patched.yaml
    """
}
