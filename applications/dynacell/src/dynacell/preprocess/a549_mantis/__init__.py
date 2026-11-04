"""A549 mantis dataset assembly utilities.

Produces per-target train/test zarrs from per-plate dynacell zarrs by
applying a 2-h odd-hpi grid, ±1.5-h tail snap, per-well channel rename,
and train/test routing authored in the packaged
``dynacell/_configs/datasets/a549-mantis/authoring/`` YAMLs.
"""

from dynacell.preprocess.a549_mantis.assemble import (
    GENE_TO_FILENAME,
    VALID_CONDITIONS,
    assemble_pool,
)
from dynacell.preprocess.a549_mantis.authoring import (
    Platemap,
    Splits,
    TargetSplit,
    WellMetadata,
    load_platemap,
    load_splits,
)
from dynacell.preprocess.a549_mantis.channels import (
    CANONICAL_TARGET_CHANNELS,
    COMBINED_TARGETS,
    DECONVOLVED_SUFFIX,
    PASSTHROUGH_CHANNELS,
    RAW_PREFIX,
    ChannelSelection,
    resolve_channels,
    resolve_target_genes,
)
from dynacell.preprocess.a549_mantis.grid import (
    GridFrame,
    build_grid,
)

__all__ = [
    "CANONICAL_TARGET_CHANNELS",
    "COMBINED_TARGETS",
    "DECONVOLVED_SUFFIX",
    "ChannelSelection",
    "GENE_TO_FILENAME",
    "GridFrame",
    "PASSTHROUGH_CHANNELS",
    "Platemap",
    "RAW_PREFIX",
    "Splits",
    "TargetSplit",
    "VALID_CONDITIONS",
    "WellMetadata",
    "assemble_pool",
    "build_grid",
    "load_platemap",
    "load_splits",
    "resolve_channels",
    "resolve_target_genes",
]
