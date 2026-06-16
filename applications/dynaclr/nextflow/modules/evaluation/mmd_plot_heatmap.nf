// Gather step: plot one combined MMD heatmap (all markers) from all per-experiment CSVs.
// Runs once per block after all MMD scatter jobs complete.

process MMD_PLOT_HEATMAP {
    executor 'local'

    input:
    val mmd_dir

    script:
    """
    uv run --project=${params.workspace_dir} --package=dynaclr \
        dynaclr plot-mmd-heatmap ${mmd_dir}
    """
}
