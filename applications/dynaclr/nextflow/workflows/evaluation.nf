// DynaCLR Evaluation Workflow
//
// Named sub-workflow invoked via `-entry EVALUATION` from main.nf.
// Takes an eval_config path + workspace_dir, runs the full embedding DAG:
// prepare-configs → predict → split → (reduce / smoothness / mmd / classifiers / plots).

include { PREPARE_CONFIGS     } from '../modules/evaluation/prepare_configs'
include { PREDICT             } from '../modules/evaluation/predict'
include { SPLIT               } from '../modules/evaluation/split'
include { REDUCE              } from '../modules/evaluation/reduce'
include { REDUCE_COMBINED     } from '../modules/evaluation/reduce_combined'
include { PLOT                } from '../modules/evaluation/plot'
include { PLOT_COMBINED       } from '../modules/evaluation/plot_combined'
include { SMOOTHNESS          } from '../modules/evaluation/smoothness'
include { MMD                 } from '../modules/evaluation/mmd'
include { MMD_COMBINED        } from '../modules/evaluation/mmd_combined'
include { MMD_PLOT_HEATMAP    } from '../modules/evaluation/mmd_plot_heatmap'
include { SMOOTHNESS_GATHER   } from '../modules/evaluation/smoothness_gather'
include { LINEAR_CLASSIFIERS  } from '../modules/evaluation/linear_classifiers'
include { APPEND_ANNOTATIONS  } from '../modules/evaluation/append_annotations'
include { APPEND_PREDICTIONS  } from '../modules/evaluation/append_predictions'


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

    split_done_ch = per_exp_zarrs_ch.collect().map { 'done' }

    // -----------------------------------------------------------------------
    // Step 4a: Per-experiment dim reduction (scatter) — after split
    // -----------------------------------------------------------------------
    reduce_yaml_ch = manifest_ch
        .filter { it.containsKey('reduce') }
        .map { it.reduce }

    reduce_inputs_ch = per_exp_zarrs_ch.combine(reduce_yaml_ch)

    REDUCE(
        reduce_inputs_ch.map { zarr, yaml -> zarr },
        reduce_inputs_ch.map { zarr, yaml -> yaml },
        workspace_dir
    )

    // -----------------------------------------------------------------------
    // Step 4b: Combined dim reduction (gather) — after all REDUCE finish
    // -----------------------------------------------------------------------
    reduce_combined_yaml_ch = manifest_ch
        .filter { it.containsKey('reduce_combined') }
        .map { it.reduce_combined }

    REDUCE_COMBINED(
        REDUCE.out.zarr_path.collect(),
        reduce_combined_yaml_ch,
        workspace_dir
    )

    // Barrier: per-experiment zarr writes (X_pca_combined / X_phate_combined)
    // must finish before APPEND_ANNOTATIONS / APPEND_PREDICTIONS start writing
    // to the same zarrs. ifEmpty('skip') keeps the chain alive when
    // reduce_combined isn't in steps.
    reduce_combined_done_ch = REDUCE_COMBINED.out.zarr_paths
        .ifEmpty('skip')
        .first()

    // -----------------------------------------------------------------------
    // Step 5: Smoothness (scatter, depends only on split)
    // -----------------------------------------------------------------------
    smoothness_yaml_ch = manifest_ch
        .filter { it.containsKey('smoothness') }
        .map { it.smoothness }

    smoothness_inputs_ch = per_exp_zarrs_ch.combine(smoothness_yaml_ch)

    SMOOTHNESS(
        smoothness_inputs_ch.map { zarr, yaml -> zarr },
        smoothness_inputs_ch.map { zarr, yaml -> yaml },
        workspace_dir
    )

    smoothness_dir_ch = manifest_ch.map { "${it.output_dir}/smoothness" }
    smoothness_done_ch = SMOOTHNESS.out.zarr_path.collect().map { 'done' }
    SMOOTHNESS_GATHER(smoothness_dir_ch, smoothness_done_ch)

    // -----------------------------------------------------------------------
    // Step 6: MMD per-experiment (scatter, depends only on split)
    // -----------------------------------------------------------------------
    mmd_block_inputs_ch = manifest_ch
        .filter { it.containsKey('mmd_blocks') && it.mmd_blocks.size() > 0 }
        .flatMap { manifest ->
            manifest.mmd_blocks.collect { block_name ->
                [block_name, manifest["mmd_${block_name}"]]
            }
        }

    mmd_per_exp_ch = per_exp_zarrs_ch
        .combine(mmd_block_inputs_ch)
        .map { zarr, block_name, mmd_yaml -> tuple(zarr, block_name, mmd_yaml) }

    MMD(mmd_per_exp_ch, workspace_dir)

    mmd_heatmap_dirs_ch = manifest_ch
        .filter { it.containsKey('mmd_blocks') && it.mmd_blocks.size() > 0 }
        .flatMap { manifest ->
            manifest.mmd_blocks.collect { block_name -> manifest["mmd_${block_name}_dir"] }
        }

    MMD.out.zarr_path.collect()
        .combine(mmd_heatmap_dirs_ch)
        .map { items -> items[-1] }
        | MMD_PLOT_HEATMAP

    // -----------------------------------------------------------------------
    // Step 6b: MMD combined (gather per block, depends only on split)
    // -----------------------------------------------------------------------
    mmd_combined_inputs_ch = manifest_ch
        .filter { it.containsKey('mmd_combined_blocks') && it.mmd_combined_blocks.size() > 0 }
        .flatMap { manifest ->
            manifest.mmd_combined_blocks.collect { block_name ->
                [block_name, manifest["mmd_${block_name}_cross_exp"]]
            }
        }

    mmd_combined_zarrs_str_ch = per_exp_zarrs_ch.collect().map { zarrs -> zarrs.join('\n') }

    mmd_combined_ch = mmd_combined_zarrs_str_ch
        .combine(mmd_combined_inputs_ch)
        .map { zarrs_str, block_name, mmd_yaml -> tuple(zarrs_str, block_name, mmd_yaml) }

    MMD_COMBINED(mmd_combined_ch, workspace_dir)

    // -----------------------------------------------------------------------
    // Step 7: Append annotations — must run AFTER reduce_combined (both
    // mutate per-experiment zarrs), and after split (zarrs must exist).
    // -----------------------------------------------------------------------
    // Concurrency invariant: per-experiment zarrs have one writer at a time.
    // Pre-write order: SPLIT -> REDUCE -> REDUCE_COMBINED -> APPEND_ANNOTATIONS
    // -> LINEAR_CLASSIFIERS (reads only) -> APPEND_PREDICTIONS -> PLOT (reads).
    aa_yaml_ch = manifest_ch
        .filter { it.containsKey('append_annotations') }
        .map { it.append_annotations }

    aa_ready_ch = split_done_ch.mix(reduce_combined_done_ch).collect().map { 'ready' }

    APPEND_ANNOTATIONS(aa_ready_ch, aa_yaml_ch, workspace_dir)

    aa_done_ch = APPEND_ANNOTATIONS.out.done
        .ifEmpty('skip')
        .first()

    // -----------------------------------------------------------------------
    // Step 8: Linear classifiers — after append_annotations
    // -----------------------------------------------------------------------
    lc_yaml_ch = manifest_ch
        .filter { it.containsKey('linear_classifiers') }
        .map { it.linear_classifiers }

    LINEAR_CLASSIFIERS(aa_done_ch, lc_yaml_ch, workspace_dir)

    // -----------------------------------------------------------------------
    // Step 9: Append predictions — after linear classifiers AND split
    // -----------------------------------------------------------------------
    // append_predictions reads per-experiment zarrs (produced by SPLIT) and
    // writes predicted_* columns to obs. It must wait on BOTH:
    //   - LINEAR_CLASSIFIERS (when present in steps): pipelines must exist
    //   - SPLIT: the zarrs to predict on must exist
    // For Wave-2 evaluations that fetch pipelines from an external registry
    // (no LINEAR_CLASSIFIERS in steps), lc_done is 'skip' immediately and
    // only the split dependency keeps APPEND_PREDICTIONS gated.
    ap_yaml_ch = manifest_ch
        .filter { it.containsKey('append_predictions') }
        .map { it.append_predictions }

    lc_done_ch = LINEAR_CLASSIFIERS.out.done
        .ifEmpty('skip')
        .first()

    // Combine the two upstream signals into one barrier value.
    ap_ready_ch = lc_done_ch.mix(split_done_ch).collect().map { 'ready' }

    APPEND_PREDICTIONS(ap_ready_ch, ap_yaml_ch, workspace_dir)

    ap_done_ch = APPEND_PREDICTIONS.out.done
        .ifEmpty('skip')
        .first()

    // -----------------------------------------------------------------------
    // Step 10a: Per-experiment plots — after reduce_combined + enrichment
    // -----------------------------------------------------------------------
    plot_yaml_ch = manifest_ch
        .filter { it.containsKey('plot') }
        .map { it.plot }

    plots_dir_ch = manifest_ch.map { "${it.output_dir}/plots" }

    enrichment_done_ch = aa_done_ch.mix(ap_done_ch).collect().map { 'ready' }

    reduce_zarrs_str_ch = REDUCE_COMBINED.out.zarr_paths
        .map { zarrs -> zarrs.join('\n') }

    post_reduce_zarrs_ch = reduce_zarrs_str_ch
        .combine(enrichment_done_ch)
        .map { zarrs_str, _ready -> zarrs_str.split('\n').toList() }
        .flatten()

    plot_inputs_ch = post_reduce_zarrs_ch.combine(plot_yaml_ch).combine(plots_dir_ch)

    PLOT(
        plot_inputs_ch.map { zarr, yaml, dir -> zarr },
        plot_inputs_ch.map { zarr, yaml, dir -> yaml },
        plot_inputs_ch.map { zarr, yaml, dir -> dir },
        workspace_dir
    )

    // -----------------------------------------------------------------------
    // Step 10b: Combined plots (gather) — after reduce_combined + enrichment
    // -----------------------------------------------------------------------
    plot_combined_yaml_ch = manifest_ch
        .filter { it.containsKey('plot_combined') }
        .map { it.plot_combined }

    plot_combined_input_ch = reduce_zarrs_str_ch
        .combine(enrichment_done_ch)
        .map { zarrs_str, _ready -> zarrs_str.split('\n').toList() }

    PLOT_COMBINED(
        plot_combined_input_ch,
        plot_combined_yaml_ch,
        workspace_dir
    )
}
