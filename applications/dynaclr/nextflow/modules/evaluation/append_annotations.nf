// Append annotation columns to per-experiment zarr obs.
// Reads per-experiment annotation CSVs and writes task columns (e.g. infection_state)
// directly into each zarr so plots can color by ground truth labels.
// Runs after SPLIT; depends on split_done signal.

process APPEND_ANNOTATIONS {
    executor 'local'

    input:
    val split_done       // dependency signal from SPLIT (all zarrs exist)
    val aa_yaml
    val workspace_dir

    output:
    val 'done', emit: done

    script:
    """
    uv run --project=${workspace_dir} --package=dynaclr \
        dynaclr append-annotations -c ${aa_yaml}
    """
}
