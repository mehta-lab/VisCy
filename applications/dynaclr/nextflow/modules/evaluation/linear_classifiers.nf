// Linear classifiers on per-experiment embeddings.
// Reads directly from the embeddings directory (all zarrs).

process LINEAR_CLASSIFIERS {
    executor 'local'

    input:
    val split_done    // dependency signal from SPLIT (embeddings dir is populated)
    val lc_yaml
    val workspace_dir

    output:
    val 'done', emit: done

    script:
    """
    uv run --project=${workspace_dir} --package=dynaclr \
        dynaclr run-linear-classifiers -c ${lc_yaml}
    """
}
