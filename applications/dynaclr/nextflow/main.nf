#!/usr/bin/env nextflow
// DynaCLR Nextflow Router
//
// Thin entry-point that dispatches to a named sub-workflow via `-entry`.
// Each sub-workflow lives in workflows/<name>.nf and owns its own DAG.
//
// Usage:
//   module load nextflow/24.10.5
//
//   # Evaluation
//   nextflow run applications/dynaclr/nextflow/main.nf -entry evaluation \
//       --eval_config /path/to/eval_config.yaml \
//       --workspace_dir /hpc/mydata/eduardo.hirata/repos/viscy \
//       -resume
//
//   # Evaluation from pre-computed embeddings (skips predict/split)
//   nextflow run applications/dynaclr/nextflow/main.nf -entry eval_from_embeddings \
//       --eval_config /path/to/eval_config.yaml \
//       --embeddings_glob '/hpc/projects/intracellular_dashboard/organelle_dynamics/*/2-phenotyping/predictions/MODEL/RUN/CKPT/*.zarr' \
//       --workspace_dir /hpc/mydata/eduardo.hirata/repos/viscy \
//       -resume
//
//   # Training preprocessing (collection → parquet)
//   nextflow run applications/dynaclr/nextflow/main.nf -entry training_preprocessing \
//       --collection_yaml /path/to/collection.yml \
//       --parquet_out /hpc/.../collections/<name>.parquet \
//       --focus_channel Phase3D \
//       --workspace_dir /hpc/mydata/eduardo.hirata/repos/viscy \
//       -resume
//
// Zarr/parquet files are read/written in place on VAST (no staging).

nextflow.enable.dsl = 2

include { EVALUATION              } from './workflows/evaluation'
include { EVAL_FROM_EMBEDDINGS    } from './workflows/eval_from_embeddings'
include { TRAINING_PREPROCESSING  } from './workflows/training_preprocessing'


// Default (unnamed) workflow — fail loudly if invoked without -entry.
workflow {
    error """
    No entry workflow selected.

    Use one of:
      -entry evaluation              (requires --eval_config)
      -entry eval_from_embeddings    (requires --eval_config, --embeddings_glob)
      -entry training_preprocessing  (requires --collection_yaml, --parquet_out)
    """.stripIndent()
}


// Entry workflow names are lowercase to avoid clashing with the imported
// named workflows above (Nextflow treats sub-workflows with `take:` as
// non-directly-invocable, so we wrap each one here).

workflow evaluation {
    if (!params.eval_config) {
        error "ERROR: --eval_config is required for -entry evaluation"
    }
    EVALUATION(file(params.eval_config), params.workspace_dir)
}


workflow eval_from_embeddings {
    if (!params.eval_config) {
        error "ERROR: --eval_config is required for -entry eval_from_embeddings"
    }
    if (!params.embeddings_glob) {
        error "ERROR: --embeddings_glob is required for -entry eval_from_embeddings"
    }
    EVAL_FROM_EMBEDDINGS(
        file(params.eval_config),
        params.embeddings_glob,
        params.workspace_dir
    )
}


workflow training_preprocessing {
    if (!params.collection_yaml) {
        error "ERROR: --collection_yaml is required for -entry training_preprocessing"
    }
    if (!params.parquet_out) {
        error "ERROR: --parquet_out is required for -entry training_preprocessing"
    }
    TRAINING_PREPROCESSING(
        params.collection_yaml,
        params.parquet_out,
        params.focus_channel,
        params.num_workers,
        params.workspace_dir
    )
}
