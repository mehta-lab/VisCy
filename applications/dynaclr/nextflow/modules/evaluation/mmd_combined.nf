// Cross-experiment MMD for one block (with per-experiment batch centering).
// Collects all zarr paths and patches the template YAML's input_paths list.

process MMD_COMBINED {
    label 'cpu'

    input:
    tuple val(zarr_paths), val(block_name), val(mmd_combined_yaml)
    val workspace_dir

    output:
    val block_name, emit: block_name

    script:
    def paths_repr = zarr_paths.split('\n').collect { "'${it}'" }.join(', ')
    """
    python3 -c "
import yaml
with open('${mmd_combined_yaml}') as f:
    cfg = yaml.safe_load(f)
cfg['input_paths'] = [${paths_repr}]
with open('mmd_${block_name}_combined_patched.yaml', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
"
    uv run --project=${workspace_dir} --package=dynaclr \
        dynaclr compute-mmd --combined -c mmd_${block_name}_combined_patched.yaml
    """
}
