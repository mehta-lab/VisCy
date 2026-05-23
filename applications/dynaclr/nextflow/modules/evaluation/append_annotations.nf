// Append annotation columns to per-experiment zarr obs.
// Reads per-experiment annotation CSVs and writes task columns (e.g. infection_state)
// directly into each zarr so plots can color by ground truth labels.
// Runs after SPLIT; depends on split_done signal.
//
// The CSV files are declared as a `path` input so Nextflow content-hashes
// them and invalidates the resume cache when an annotation file changes.
// They are staged into a side-directory so the basename collisions across
// datasets don't pollute the task CWD; the script still reads them by
// absolute path via the YAML.

process APPEND_ANNOTATIONS {
    executor 'local'

    input:
    val split_done       // dependency signal from SPLIT (all zarrs exist)
    val aa_yaml
    path csv_files, stageAs: 'csv_files_staged/*'
    val workspace_dir

    output:
    val 'done', emit: done

    script:
    """
    uv run --project=${workspace_dir} --package=dynaclr \
        dynaclr append-annotations -c ${aa_yaml}
    """
}
