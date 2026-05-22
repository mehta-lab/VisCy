// Enrich the cell index parquet with norm stats + per-timepoint focus slice z.
// Opens each unique FOV once from zarr zattrs; overwrites parquet in place.

process PREPROCESS_CELL_INDEX {
    label 'cpu'

    input:
    val parquet_in
    val focus_channel
    val workspace_dir

    output:
    val parquet_in, emit: parquet

    script:
    """
    uv run --project=${workspace_dir} --package=dynaclr \
        dynaclr preprocess-cell-index \
        ${parquet_in} \
        --focus-channel ${focus_channel}
    """
}
