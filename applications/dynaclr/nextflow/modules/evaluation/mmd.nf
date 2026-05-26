// Per-experiment MMD for one (zarr, mmd_block) pair.
// Patches __ZARR_PATH__ in the per-block template YAML.

process MMD {
    label 'cpu'

    input:
    tuple val(zarr_path), val(block_name), val(mmd_yaml)
    val workspace_dir

    output:
    val zarr_path, emit: zarr_path

    script:
    def exp_name = new File(zarr_path).name.replaceAll(/\.zarr$/, '')
    """
    python3 -c "
import yaml
with open('${mmd_yaml}') as f:
    cfg = yaml.safe_load(f)
cfg['input_path'] = '${zarr_path}'
with open('mmd_${block_name}_${exp_name}_patched.yaml', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
"
    uv run --project=${workspace_dir} --package=dynaclr \
        dynaclr compute-mmd -c mmd_${block_name}_${exp_name}_patched.yaml
    """
}
