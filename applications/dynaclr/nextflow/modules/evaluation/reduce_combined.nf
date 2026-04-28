// Joint PCA + PHATE across all experiments.
// Collects all reduced per-experiment zarr paths, patches the template YAML,
// then runs combined-dim-reduction which writes X_pca_combined / X_phate_combined
// back into each per-experiment zarr.

process REDUCE_COMBINED {
    label 'cpu_heavy'

    input:
    val zarr_paths   // list of all per-experiment zarr paths (after REDUCE.collect())
    val reduce_combined_yaml
    val workspace_dir

    output:
    val zarr_paths, emit: zarr_paths

    script:
    def paths_repr = zarr_paths.collect { "'${it}'" }.join(', ')
    """
    # Cap BLAS thread pools. PHATE -> graphtools -> sklearn PCA -> scipy LU
    # deadlocks when BLAS tries to use all 48 cpus on cpu_heavy nodes.
    # 16 threads is plenty for our matrix sizes (~350k x 768).
    export OPENBLAS_NUM_THREADS=16
    export MKL_NUM_THREADS=16
    export OMP_NUM_THREADS=16
    export NUMEXPR_NUM_THREADS=16
    export VECLIB_MAXIMUM_THREADS=16

    python3 -c "
import yaml
with open('${reduce_combined_yaml}') as f:
    cfg = yaml.safe_load(f)
cfg['input_paths'] = [${paths_repr}]
with open('reduce_combined_patched.yaml', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
"
    uv run --project=${workspace_dir} --package=dynaclr \
        dynaclr combined-dim-reduction -c reduce_combined_patched.yaml
    """
}
