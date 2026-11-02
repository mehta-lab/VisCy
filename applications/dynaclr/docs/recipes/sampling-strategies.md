# Choose sampling settings

Sampling has three separate decisions: which rows share a batch, which channels
are loaded, and how an anchor's positive is selected. Configure them under
`data.init_args` in a training YAML.

## Batch composition

| Setting | Effect |
| --- | --- |
| `batch_group_by` | Restricts a batch to one value or value combination. |
| `stratify_by` | Balances rows across one or more cell-index columns. |
| `leaky` | Replaces part of a grouped batch with rows outside its group. |
| `temporal_enrichment` | Concentrates a batch around a sampled time window. |

A safe multi-experiment starting point is:

```yaml
data:
  init_args:
    batch_group_by: [experiment]
    stratify_by: [perturbation]
    leaky: 0.0
    temporal_enrichment: false
```

Keep `batch_group_by: [experiment]` when channel semantics or acquisition
conditions differ between experiments. Set it to `null` only when rows can be
mixed meaningfully. Use `leaky` only when intentional cross-experiment mixing
has been validated.

For time-matched batches:

```yaml
data:
  init_args:
    temporal_enrichment: true
    temporal_window_hours: 2.0
    temporal_global_fraction: 0.3
```

All grouping and stratification columns must exist in the preprocessed cell
index.

## Channels per sample

```yaml
data:
  init_args:
    channels_per_sample: 1
```

`channels_per_sample` accepts:

- `1`: randomly select one indexed channel per sample (bag of channels);
- a list such as `[Phase3D, raw GFP EX488 EM525-45]`: load those channels;
- `null`: load every source channel.

The model's `in_channels` must match the resulting tensor. Integer values above
one are unsupported; use a channel list. `channel_dropout_prob` is for selected
channels in multi-channel inputs, not a replacement for bag-of-channels
sampling.

## Positive pairs

Temporal positives from the same lineage:

```yaml
data:
  init_args:
    positive_cell_source: lookup
    positive_match_columns: [lineage_id]
    positive_channel_source: same
    tau_range: [0.5, 2.0]
    tau_decay_rate: 2.0
```

`tau_range` is in hours and is converted to frames per experiment. Use
`positive_cell_source: self` for two augmented views of the same observation.
With `channels_per_sample: 1`, `positive_channel_source: same` preserves the
anchor channel and `any` samples the positive channel independently.

For non-temporal lookup, use stable cell-index columns that define a valid
positive group, for example:

```yaml
positive_cell_source: lookup
positive_match_columns: [gene_name, reporter]
```

## Check before a full run

Run one training and validation batch:

```sh
uv run dynaclr fit -c /path/to/training.yml \
  --trainer.fast_dev_run=true
```

Inspect the index group counts and verify that every anchor group contains a
valid positive. A small valid-anchor set usually means short or broken tracks,
missing lineage information, or a `tau_range` that does not match the imaging
interval.
