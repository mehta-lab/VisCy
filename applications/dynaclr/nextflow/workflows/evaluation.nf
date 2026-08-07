// DynaCLR Evaluation Workflow
//
// Named sub-workflow invoked via `-entry EVALUATION` from main.nf.
// Takes an eval_config path + workspace_dir, runs the full embedding DAG:
// prepare-configs → predict → split → (reduce / smoothness / mmd / classifiers / plots).

include { PREPARE_CONFIGS     } from '../modules/evaluation/prepare_configs'
include { PREDICT             } from '../modules/evaluation/predict'
include { SPLIT               } from '../modules/evaluation/split'
include { DOWNSTREAM          } from './_downstream'


workflow EVALUATION {
    take:
        eval_config
        workspace_dir

    main:
    // -----------------------------------------------------------------------
    // Step 1: Generate per-step YAML configs → JSON manifest
    // -----------------------------------------------------------------------
    PREPARE_CONFIGS(eval_config, workspace_dir)

    manifest_ch = PREPARE_CONFIGS.out.manifest
        .map { f -> new groovy.json.JsonSlurper().parse(f) }

    // -----------------------------------------------------------------------
    // Step 2: Predict (GPU) — only if "predict" key is in manifest
    // -----------------------------------------------------------------------
    predict_yaml_ch = manifest_ch
        .flatMap { manifest ->
            manifest.containsKey('predict') ? [manifest.predict] : []
        }

    PREDICT(predict_yaml_ch, workspace_dir)

    predict_signal_ch = PREDICT.out.done
        .ifEmpty('skip')
        .first()

    // -----------------------------------------------------------------------
    // Step 3: Split — runs after predict (or immediately if predict skipped)
    // -----------------------------------------------------------------------
    SPLIT(
        predict_signal_ch,
        manifest_ch.map { it.embeddings_dir },
        manifest_ch.map { it.cell_index_path },
        manifest_ch.map { it.output_dir },
        workspace_dir
    )

    per_exp_zarrs_ch = SPLIT.out.zarr_paths_file
        .splitText()
        .map { it.trim() }
        .filter { it.endsWith('.zarr') }

    // -----------------------------------------------------------------------
    // Steps 4-10: Shared downstream DAG (reduce / smoothness / mmd /
    // classifiers / plots) with all per-experiment-zarr write-order barriers.
    // -----------------------------------------------------------------------
    DOWNSTREAM(per_exp_zarrs_ch, manifest_ch, workspace_dir)
}
