// Build the flat cell index parquet from a collection YAML.
// Reads tracking CSVs + zarr shape metadata; writes one row per (cell, timepoint, channel).

process BUILD_CELL_INDEX {
    label 'cpu'

    input:
    val collection_yaml
    val parquet_out
    val num_workers
    val workspace_dir

    output:
    val parquet_out, emit: parquet

    script:
    """
    uv run --project=${workspace_dir} --package=dynaclr \
        dynaclr build-cell-index \
        ${collection_yaml} \
        ${parquet_out} \
        --num-workers ${num_workers}
    """
}
