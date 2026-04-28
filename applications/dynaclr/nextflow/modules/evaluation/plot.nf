// Per-experiment embedding scatter plots (X_pca).
// Patches __ZARR_PATH__ and __PLOT_DIR__ placeholders in the template YAML.

process PLOT {
    label 'cpu'

    input:
    val zarr_path
    val plot_yaml
    val plots_dir
    val workspace_dir

    output:
    val zarr_path, emit: zarr_path

    script:
    def exp_name  = new File(zarr_path).name.replaceAll(/\.zarr$/, '')
    def plot_subdir = "${plots_dir}/${exp_name}"
    """
    python3 -c "
import yaml
with open('${plot_yaml}') as f:
    cfg = yaml.safe_load(f)
cfg['input_path'] = '${zarr_path}'
cfg['output_dir'] = '${plot_subdir}'
with open('plot_patched.yaml', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
"
    mkdir -p ${plot_subdir}
    uv run --project=${workspace_dir} --package=dynaclr \
        dynaclr plot-embeddings -c plot_patched.yaml
    """
}
