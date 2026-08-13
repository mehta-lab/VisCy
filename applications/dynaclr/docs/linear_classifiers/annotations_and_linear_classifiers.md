# Annotation and classifier contract

This page defines how annotation CSVs join to embedding rows and how classifier
outputs are named. Run commands and configs are in the
[linear-classifier runbook](README.md).

## Annotation CSV

Each CSV needs one task-label column and one supported join key.

| Requirement | Columns |
| --- | --- |
| Always | `fov_name`, `<task>` |
| Preferred join | `id` |
| Fallback join | `t`, `track_id` |
| Duplicate-key disambiguation | `y`, `x` |

Preferred form:

```csv
fov_name,id,infection_state
A/1/000001,101,uninfected
A/1/000001,102,infected
```

Fallback form:

```csv
fov_name,t,track_id,y,x,infection_state
A/1/000001,4,17,128.5,256.3,infected
A/1/000001,5,17,130.2,255.8,infected
```

The loader joins on `(fov_name, id)` when `id` exists in both inputs.
Otherwise, it joins on `(fov_name, t, track_id)`. Leading and trailing slashes
are stripped from `fov_name` before matching.

If a join key is duplicated, both annotations and embedding `obs` must contain
`y` and `x`. The nearest row within the spatial tolerance is used. Avoid
duplicate `(fov_name, id)` values when possible.

`dataset_name`, lineage fields, and coordinates are not required for a unique
preferred join. Preserve them when they are useful for auditing or group-aware
splits.

## Labels

- The task name is the CSV column containing class labels.
- Empty, `NaN`, and `unknown` labels are excluded from training.
- Labels must use one spelling and capitalization across all training CSVs.
- At least two classes must remain after filtering.

The batch workflow accepts any task column name. The standalone W&B config
currently restricts `task` to:

```text
infection_state
organelle_state
cell_division_state
cell_death_state
```

Its `input_channel` values are `phase`, `sensor`, and `marker`. These values
identify the embedding input; they do not replace the physical marker stored in
embedding metadata.

## Embedding requirements

The feature matrix is read from AnnData `.X`. Embedding `obs` must contain the
same join columns used by the CSV.

The batch workflow additionally requires:

- `experiment`, matching the `annotations[].experiment` config value;
- `marker`, used to select task-marker training rows;
- every column listed in `split_groups_by`.

For tracked time series, use
`split_groups_by: [experiment, fov_name, track_id]` so complete tracks stay in
one split.

`include_wells` matches the `{row}/{column}/` prefix of `fov_name` and is useful
when a physical channel has different marker semantics across wells.

## Model and output names

Batch bundles contain:

```text
manifest.json
<task>_<marker>.joblib
```

Published bundles live under `<publish_dir>/vN/`. The manifest records the task,
marker, and pipeline file consumed by `append-predictions`.

W&B artifact names follow:

```text
linear-classifier-<task>-<input_channel>[-<marker>]
```

Applying either workflow writes:

```python
adata.obs[f"predicted_{task}"]
adata.obsm[f"predicted_{task}_proba"]
adata.uns[f"predicted_{task}_classes"]
```

The probability columns follow the class order stored in `.uns`. Application
must use embeddings produced by the same model, checkpoint, representation key,
and preprocessing contract as the training embeddings.
