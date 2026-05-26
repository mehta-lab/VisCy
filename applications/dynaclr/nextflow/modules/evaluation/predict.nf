// Run viscy predict to extract embeddings from a checkpoint.
// Writes embeddings/embeddings.zarr in output_dir.

process PREDICT {
    label "${params.gpu_label}"

    input:
    val predict_yaml
    val workspace_dir

    output:
    val 'done', emit: done

    script:
    """
    uv run --project=${workspace_dir} --package=viscy-utils \
        viscy predict -c ${predict_yaml}
    """
}
