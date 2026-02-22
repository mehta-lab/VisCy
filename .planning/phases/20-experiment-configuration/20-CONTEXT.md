# Phase 20: Experiment Configuration - Context

**Gathered:** 2026-02-21
**Status:** Ready for planning

<domain>
## Phase Boundary

Define multi-experiment training setups via `ExperimentConfig` dataclass and `ExperimentRegistry`, with automatic channel resolution and YAML config parsing for Lightning CLI. New files in `applications/dynaclr/src/dynaclr/`. No modification to triplet.py or existing data modules.

</domain>

<decisions>
## Implementation Decisions

### Condition modeling
- Conditions are arbitrary string labels mapped to wells (not hard-coded infected/uninfected)
- `condition_wells` is `dict[str, list[str]]` — multiple wells per condition supported (replicate wells)
- One condition per well (no mixed populations within a well)
- Default condition balance is 50/50, configurable via `condition_ratio` dict in FlexibleBatchSampler (Phase 22 concern, but captured here)
- `hours_post_infection` computed identically for all cells: `start_hpi + (frame * interval_minutes / 60)`. Same clock for uninfected and infected — different semantic meaning but same computation
- `start_hpi` is a per-experiment field in ExperimentConfig (e.g., 3.0 for experiments starting at 3 HPI)
- For uninfected wells, `hours_post_infection` is just "hours since experiment start" on the same clock

### Channel resolution
- **Explicit list only** — no "shared" or "all" modes. User specifies `source_channel: list[str]` per experiment
- **Positional alignment** across experiments: position 0 = first source channel, position 1 = second, etc. Names can differ between experiments (GFP in exp A = RFP in exp B) as long as position count matches
- ExperimentRegistry validates that all experiments have the same **number** of source channels
- If any experiment's `source_channel` references a name not in its `channel_names`, raise ValueError at registry creation
- `channel_names` is the full list of channels in the zarr store; `source_channel` selects which to use for training

### YAML config structure
- Separate experiments file: `experiments_file: "experiments.yml"` in DataModule config
- DataModule loads the file and builds ExperimentRegistry internally
- ExperimentRegistry also has `from_yaml(path)` classmethod for standalone use in notebooks/scripts
- `tau_range` is in **hours**, not frames — converted to frames per experiment using `interval_minutes`
  - Example: `tau_range_hours: [0.5, 2.0]` with 30-min interval → frames [1, 4]; with 15-min interval → frames [2, 8]
  - Warn if tau range yields fewer than 2 valid frames for any experiment

### Validation
- **Fail fast at `__init__`** — validate everything at registry creation
- **Path validation**: Check `data_path` exists AND open zarr briefly to read channel names from metadata
- **Channel validation**: If `channel_names` in ExperimentConfig doesn't match zarr metadata, raise ValueError with diff showing expected vs actual
- **Source channel validation**: If any `source_channel` entry not found in `channel_names`, raise ValueError
- **Channel count**: All experiments must have same number of `source_channel` entries

### Claude's Discretion
- Additional validations: duplicate experiment names, empty condition_wells, negative interval_minutes
- ExperimentConfig field ordering and defaults
- ExperimentRegistry internal data structures (how channel_maps are stored)
- Whether to use pydantic or plain dataclass (project uses dataclass pattern)

</decisions>

<specifics>
## Specific Ideas

- Channel metadata in zarr `.zattrs` follows a rich schema with protein_tag, organelle, fluorophore, modality fields. Future helper functions can read this metadata and auto-populate ExperimentConfig. For v2.2, user specifies channels manually.
- Example channel_metadata schema from user:
  ```json
  {
    "channel_metadata": {
      "channels": {
        "raw GFP EX488 EM525-45": {
          "protein_tag": "SEC61B",
          "organelle": "endoplasmic_reticulum",
          "fluorophore": "eGFP",
          "modality": "fluorescence"
        },
        "Phase": {
          "modality": "phase"
        }
      },
      "perturbation": "ZIKV",
      "time_sampling_minutes": 30,
      "hours_post_perturbation": 24
    }
  }
  ```
- Experiment time intervals vary significantly: 15 min, 30 min, 1 hr, 2 hrs across different experiments
- Infected wells start at ~3 HPI, infection becomes visible around ~9 HPI. Early timepoints look similar to uninfected — this is the core challenge the temporal enrichment addresses (Phase 22)

</specifics>

<deferred>
## Deferred Ideas

- **Channel metadata auto-resolution**: Read `channel_metadata` from zarr `.zattrs` and auto-populate ExperimentConfig by modality/organelle — future helper function
- **"shared" and "all" training_channels modes**: Automatic channel intersection/union resolution — v2.3+
- **Zero-padding for missing channels**: When training_channels="all" and experiment lacks a channel → pad with zeros — v2.3+
- **Per-cell condition assignment**: Within-well condition heterogeneity (some cells infected, some resistant) — requires fluorescence-based classification, not well-level assignment

</deferred>

---

*Phase: 20-experiment-configuration*
*Context gathered: 2026-02-21*
