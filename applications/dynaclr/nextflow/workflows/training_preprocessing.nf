// DynaCLR Training Preprocessing Workflow
//
// Named sub-workflow invoked via `-entry TRAINING_PREPROCESSING` from main.nf.
// Takes a collection YAML and produces a training-ready parquet:
//   build-cell-index → preprocess-cell-index (norm stats + focus slice z).
//
// Required params:
//   --collection_yaml   path to configs/collections/<name>.yml
//   --parquet_out       output parquet path
//   --focus_channel     channel used for per-timepoint z (default: Phase3D)
//   --num_workers       build-cell-index parallelism (default: 8)

include { BUILD_CELL_INDEX       } from '../modules/preprocessing/build_cell_index'
include { PREPROCESS_CELL_INDEX  } from '../modules/preprocessing/preprocess_cell_index'


workflow TRAINING_PREPROCESSING {
    take:
        collection_yaml
        parquet_out
        focus_channel
        num_workers
        workspace_dir

    main:
    BUILD_CELL_INDEX(collection_yaml, parquet_out, num_workers, workspace_dir)
    PREPROCESS_CELL_INDEX(BUILD_CELL_INDEX.out.parquet, focus_channel, workspace_dir)
}
