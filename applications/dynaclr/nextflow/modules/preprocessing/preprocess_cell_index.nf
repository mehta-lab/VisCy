// Enrich the cell index parquet with norm stats + per-timepoint focus slice z.
// Opens each unique FOV once from zarr zattrs (or CSV sidecars if csv_dir is
// set); overwrites parquet in place.

process PREPROCESS_CELL_INDEX {
    label 'cpu'

    input:
    val parquet_in
    val focus_channel
    val csv_dir
    val workspace_dir

    output:
    val parquet_in, emit: parquet

    script:
    def csvDirFlag = csv_dir ? "--csv-dir ${csv_dir}" : ""
    """
    uv run --project=${workspace_dir} --package=dynaclr \
        dynaclr preprocess-cell-index \
        ${parquet_in} \
        --focus-channel ${focus_channel} \
        ${csvDirFlag}
    """
}
