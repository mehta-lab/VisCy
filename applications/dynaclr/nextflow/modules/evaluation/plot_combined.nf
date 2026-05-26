// Combined embedding plots across all experiments (X_pca_combined, X_phate_combined).
// Collects all zarr paths and patches the template YAML's input_paths list.

process PLOT_COMBINED {
    label 'cpu'

    input:
    val zarr_paths   // list of all per-experiment zarr paths
    val plot_combined_yaml
    val workspace_dir

    output:
    val 'done', emit: done

    script:
    def paths_repr = zarr_paths.collect { "'${it}'" }.join(', ')
    """
    python3 -c "
import yaml
with open('${plot_combined_yaml}') as f:
    cfg = yaml.safe_load(f)
cfg['input_paths'] = [${paths_repr}]
with open('plot_combined_patched.yaml', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
"
    uv run --project=${workspace_dir} --package=dynaclr \
        dynaclr plot-embeddings -c plot_combined_patched.yaml
    """
}
