---
license: bsd-3-clause
pretty_name: DynaCell Virtual Staining Demo Data
task_categories:
  - image-to-image
tags:
  - microscopy
  - virtual-staining
  - fluorescence
  - phase-contrast
  - ome-zarr
  - cell-biology
---

# DynaCell Virtual Staining Demo Data

Paired phase-contrast + fluorescence OME-Zarr datasets used by the
[`biohub/dynacell`](https://huggingface.co/spaces/biohub/dynacell) virtual-staining
demo. Each file is one live-cell **A549** acquisition with a different fluorescent
organelle marker, zipped as a single OME-Zarr HCS store.

Companion checkpoints: [`biohub/dynacell-checkpoints`](https://huggingface.co/biohub/dynacell-checkpoints).

## Files

| File | Marker | Target | Size |
| --- | --- | --- | --- |
| `CAAX_mock.zarr.zip` | CAAX | Membrane | ~188 MB |
| `H2B_mock.zarr.zip` | H2B | Nuclei / chromatin | ~184 MB |
| `SEC61B_mock.zarr.zip` | SEC61B | ER | ~156 MB |
| `TOMM20_mock.zarr.zip` | TOMM20 | Mitochondria | ~156 MB |

`mock` denotes the uninfected (control) imaging condition.

## Layout

Each `.zip` unpacks to an OME-Zarr HCS store:

```
{marker}_mock.zarr/
  0/0/fov0000/0      # array (T, C, Z, Y, X)
                     # C[0] = Phase3D (input), C[2] = experimental fluorescence (target)
                     # Z = 16, YX = 512×512
```

## Usage

```python
from huggingface_hub import hf_hub_download

zip_path = hf_hub_download(
    "biohub/dynacell-demo-data", "CAAX_mock.zarr.zip", repo_type="dataset"
)
# unzip → open with iohub.ngff.open_ome_zarr
```

Read the stores with [iohub](https://github.com/czbiohub-sf/iohub). The demo Space
loads these automatically via its "Load Demo Data" button.

## License

BSD 3-Clause — © CZ Biohub SF.

## Citation

Please cite the DynaCell study and VisCy. <!-- TODO: add final paper reference / DOI -->
