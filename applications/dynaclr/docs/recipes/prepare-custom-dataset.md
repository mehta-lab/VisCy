# Prepare a custom time-lapse dataset

DynaCLR needs aligned image data, tracking tables, and a collection YAML. This
page defines that input contract; it does not prescribe an image-conversion or
tracking algorithm.

## Image store

Store images as HCS OME-Zarr with `TCZYX` arrays at:

```text
<dataset>.zarr/<row>/<column>/<fov>/0
```

All positions must expose consistent channel names. Before training or
inference, add focus and normalization metadata using the
[AI-ready dataset workflow](../DAGs/ai_ready_datasets.md).

## Tracking store

Mirror the image FOV layout under the tracking root, commonly named
`tracking.zarr`, and place exactly one CSV in every included FOV directory:

```text
tracking.zarr/<row>/<column>/<fov>/tracks.csv
```

Required columns are:

| Column | Meaning |
| --- | --- |
| `track_id` | Track identifier within the FOV. |
| `t` | Zero-based image timepoint. |
| `y`, `x` | Cell centroid in image pixels. |

Use `z` for 3D centroids. Preserve `id`, `parent_track_id`, and other lineage
columns when available; temporal positive sampling depends on consistent tracks
and reconstructed lineages.

## Collection YAML

Add the dataset under
[`applications/dynaclr/configs/collections/`](../../configs/collections/):

```yaml
name: my-collection
experiments:
  - name: my-experiment
    data_path: /path/to/my-experiment.zarr
    tracks_path: /path/to/tracking.zarr
    channels:
      - name: Phase3D
        marker: Phase3D
      - name: raw GFP EX488 EM525-45
        marker: SEC61B
        wells: [A/2]
    perturbation_wells:
      control: [A/1]
      perturbed: [A/2]
    interval_minutes: 30.0
    start_hpi: 0.0
    pixel_size_xy_um: 0.1494
    pixel_size_z_um: 0.174
```

`channels[].name` must match zarr metadata exactly. `marker` is the stable label
used in sampling and output paths. If one physical channel maps to several
markers, each mapping must have explicit, non-overlapping `wells`.

## Validate the contract

Build and preprocess a small index before writing a training config:

```sh
uv run dynaclr build-cell-index collection.yml /tmp/my-collection.parquet \
  --num-workers 1

uv run dynaclr preprocess-cell-index /tmp/my-collection.parquet \
  --focus-channel Phase3D
```

This catches missing paths, channel mismatches, missing or duplicate tracking
CSVs, invalid wells, and malformed tracking columns. Then follow
[Build a cell index](build-cell-index.md) and
[Train across experiments](train-multi-experiment.md).
