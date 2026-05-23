// Apply saved linear classifiers to per-experiment zarrs and write predictions.
// Loads pipelines saved by LINEAR_CLASSIFIERS, predicts on all cells per marker,
// and writes predicted_{task} columns to obs alongside probabilities in obsm.
// Depends on LINEAR_CLASSIFIERS completing (pipelines must exist).
//
// The LC pipeline files are declared as a `path` input so Nextflow
// content-hashes them and invalidates the resume cache when an upstream LC
// re-trains and produces new pipelines. They are staged into a side-directory;
// the script still loads them by absolute path via the YAML's pipelines_dir.

process APPEND_PREDICTIONS {
    executor 'local'

    input:
    val lc_done          // dependency signal from LINEAR_CLASSIFIERS
    val ap_yaml
    path pipeline_files, stageAs: 'pipelines_staged/*'
    val workspace_dir

    output:
    val 'done', emit: done

    script:
    """
    uv run --project=${workspace_dir} --package=dynaclr \
        dynaclr append-predictions -c ${ap_yaml}
    """
}
