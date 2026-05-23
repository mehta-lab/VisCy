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
    # Pin BLAS to 1 thread per process: PHATE's n_jobs spawns one joblib
    # worker per allocated CPU (SLURM_CPUS_PER_TASK), each running KNN
    # search single-threaded. If BLAS were unbounded, every worker would
    # also try to spawn ~cores threads on its own internal matmuls,
    # producing ~cores^2 threads and thrashing the node. Standard sklearn
    # parallelism pattern: one axis at a time. KNN search dominates wall
    # time on PHATE-with-PCA-input; the BLAS-heavy phases (joint PCA,
    # diffusion matrix powers) are bounded fast even at 1 thread.
    # Also avoids the scipy.lu deadlock that hit when BLAS tried to use
    # all 48 cores on cpu_heavy nodes.
    export OPENBLAS_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export OMP_NUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1
    export VECLIB_MAXIMUM_THREADS=1

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
