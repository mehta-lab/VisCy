// Linear classifiers on per-experiment embeddings.
// Reads directly from the embeddings directory (all zarrs).
//
// The annotation CSVs are declared as a `path` input so Nextflow
// content-hashes them and invalidates the resume cache when an annotation
// file changes. They are staged into a side-directory; the script still
// reads them by absolute path via the YAML.

process LINEAR_CLASSIFIERS {
    executor 'local'

    input:
    val split_done    // dependency signal from SPLIT (embeddings dir is populated)
    val lc_yaml
    path csv_files, stageAs: 'csv_files_staged/*'
    val workspace_dir

    output:
    val 'done', emit: done

    script:
    """
    uv run --project=${workspace_dir} --package=dynaclr \
        dynaclr run-linear-classifiers -c ${lc_yaml}
    """
}
