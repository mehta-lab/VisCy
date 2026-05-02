// Run dynaclr prepare-eval-configs to generate per-step YAML configs.
// Prints a JSON manifest to stdout; we capture it as the process output.

process PREPARE_CONFIGS {
    executor 'local'

    input:
    path eval_config
    val workspace_dir

    output:
    path 'manifest.json', emit: manifest

    script:
    """
    uv run --project=${workspace_dir} --package=dynaclr \
        dynaclr prepare-eval-configs -c ${eval_config} > manifest.json
    """
}
