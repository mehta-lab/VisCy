// DynaCLR Eval-From-Embeddings Workflow
//
// Named sub-workflow invoked via `-entry eval_from_embeddings` from main.nf.
// Decouples inference from evaluation: instead of running PREDICT / SPLIT, it
// sources pre-computed per-experiment embedding zarrs from a glob and runs the
// same shared DOWNSTREAM DAG as the evaluation workflow.

include { PREPARE_CONFIGS     } from '../modules/evaluation/prepare_configs'
include { DOWNSTREAM          } from './_downstream'


workflow EVAL_FROM_EMBEDDINGS {
    take:
        eval_config
        embeddings_glob
        workspace_dir

    main:
    // -----------------------------------------------------------------------
    // Step 1: Generate per-step YAML configs → JSON manifest.
    // Only the per-step configs are consumed here; the manifest's
    // embeddings_dir is not used to source zarrs — the glob does that.
    // -----------------------------------------------------------------------
    PREPARE_CONFIGS(eval_config, workspace_dir)

    manifest_ch = PREPARE_CONFIGS.out.manifest
        .map { f -> new groovy.json.JsonSlurper().parse(f) }

    // -----------------------------------------------------------------------
    // Step 2: Source per-experiment embedding zarrs from the glob (directories)
    // instead of running PREDICT / SPLIT.
    // -----------------------------------------------------------------------
    per_exp_zarrs_ch = Channel.fromPath(embeddings_glob, type: 'dir')
        .map { it.toString() }
        .filter { it.endsWith('.zarr') }

    // -----------------------------------------------------------------------
    // Steps 4-10: Shared downstream DAG (reduce / smoothness / mmd /
    // classifiers / plots) with all per-experiment-zarr write-order barriers.
    // -----------------------------------------------------------------------
    DOWNSTREAM(per_exp_zarrs_ch, manifest_ch, workspace_dir)
}
