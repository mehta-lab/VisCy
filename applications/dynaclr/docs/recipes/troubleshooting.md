# Recipe: Troubleshooting DynaCLR

Common issues and how to fix them.

## Startup and configuration

### "Duplicate experiment name"

```
ValueError: Duplicate experiment name 'my_exp'. Each experiment must have a unique name.
```

Each experiment in `experiments.yml` needs a unique `name` field.

### "channel_names mismatch"

```
ValueError: Experiment 'my_exp': channel_names mismatch.
Expected (from config): ['Phase3D', 'GFP'], got (from zarr): ['Phase', 'GFP']
```

The `channel_names` in your YAML must exactly match the zarr metadata. Check:

```python
from iohub.ngff import open_ome_zarr
plate = open_ome_zarr("my_experiment.zarr", mode="r")
pos = next(iter(plate.positions()))[1]
print(pos.channel_names)
```

### "source_channel entries not found in channel_names"

Your `source_channel` list references channels not in `channel_names`.
Every entry in `source_channel` must be a member of `channel_names`.

### "All experiments must have the same number of source_channel entries"

Multi-experiment training requires positional channel alignment. If one
experiment has `source_channel: ["Phase3D", "GFP"]` (2 channels), all
experiments must also have exactly 2 source channels.

### "No training experiments remaining after splitting"

All your experiments ended up in `val_experiments`. Make sure at least one
experiment name in `experiments.yml` is **not** listed in `val_experiments`.

## Data loading

### "No tracking CSV in ..., skipping"

The expected CSV file is missing. Check that your tracking CSVs follow the
directory structure:

```
{tracks_path}/{row}/{col}/{fov_idx}/something.csv
```

The loader globs for `*.csv` in each FOV directory.

### Slow startup

If `MultiExperimentIndex` takes minutes to initialize, use a pre-built
cell index parquet:

```bash
dynaclr build-cell-index experiments.yml cell_index.parquet
```

Then add to your training config:

```yaml
data:
  init_args:
    cell_index_path: /path/to/cell_index.parquet
```

See `build-cell-index.md`.

### "valid_anchors" is very small

Valid anchors require that for each cell observation, at least one other
observation from the **same lineage** exists within `tau_range` frames.

Common causes:
- `tau_range` is too narrow for the imaging interval
- Tracks are very short (few timepoints)
- No lineage links (`parent_track_id` column missing or all NaN)

Check your tau conversion:

```python
from dynaclr.data.experiment import ExperimentRegistry
registry = ExperimentRegistry.from_yaml("experiments.yml")
for exp in registry.experiments:
    min_f, max_f = registry.tau_range_frames(exp.name, (0.5, 2.0))
    print(f"{exp.name}: tau_range_frames = ({min_f}, {max_f})")
```

## Training

### Out of memory (OOM)

Reduce memory usage in order of impact:

1. **Reduce `yx_patch_size`** — e.g., `[256, 256]` instead of `[384, 384]`
2. **Reduce `batch_size`** — halving batch size roughly halves GPU memory
3. **Reduce `z_range`** — fewer Z-slices = smaller input volume
4. **Reduce `in_stack_depth`** — must match `z_range[1] - z_range[0]`
5. **Use `precision: 16-mixed`** — mixed precision halves activation memory

### Loss is NaN

- Check that normalizations produce finite values (no division by zero)
- Ensure `temperature` in `NTXentHCL` is not too small (typical: 0.05-0.1)
- Verify your image data doesn't contain NaN or Inf values

### Loss plateaus early

- Try lower `temperature` (sharper contrastive objective)
- Increase `beta` in `NTXentHCL` (harder negative mining)
- Ensure `channel_dropout_prob` isn't too high — the model needs to see
  fluorescence often enough to learn from it
- Check that `condition_balanced: true` is set — imbalanced conditions can
  cause the model to collapse to trivial solutions

### DDP hangs

- Set `export NCCL_DEBUG=INFO` to see communication logs
- Ensure all GPUs can see each other (`nvidia-smi` on compute node)
- Check that `use_distributed_sampler: false` is set (FlexibleBatchSampler
  handles DDP internally)

## Prediction and evaluation

### Embeddings look random / poor quality

- **Match normalizations exactly** between training and inference configs
- **Match `final_yx_patch_size`** — using a different crop size changes the
  effective receptive field
- Ensure you're loading the correct checkpoint (`ckpt_path`)
- Check that `source_channel` order matches training (positional alignment)

### Linear classifier accuracy is low

- Verify annotation quality — check for label noise or ambiguous categories
- Try `use_pca: true` with `n_pca_components: 32` to reduce noise
- Ensure `class_weight: balanced` is set for imbalanced label distributions
- Increase `max_iter` if the solver doesn't converge

### "KeyError: fov_name" when applying classifier

Annotations CSV must have a `fov_name` column that matches the FOV naming
convention in the embeddings zarr (e.g., `A/1/0`).
