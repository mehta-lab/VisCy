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
demo. Each file is one held-out live-cell **A549** field of view with a different
fluorescent organelle marker, zipped as a single OME-Zarr HCS store.

Companion checkpoints: [`biohub/dynacell-checkpoints`](https://huggingface.co/biohub/dynacell-checkpoints).

## Files

| File | Marker | Target | Size |
| --- | --- | --- | --- |
| `CAAX_mock.zarr.zip` | CAAX | Membrane | ~381 MB |
| `H2B_mock.zarr.zip` | H2B | Nuclei / chromatin | ~372 MB |
| `SEC61B_mock.zarr.zip` | SEC61B | ER | ~314 MB |
| `TOMM20_mock.zarr.zip` | TOMM20 | Mitochondria | ~315 MB |

`mock` denotes the uninfected (control) imaging condition.

## Provenance

Each store is a crop of `fov0006` from the DynaCell v1 **test** split,
`s3://dynacell/v1/data/biohub-a549/test/{marker}_mock.ozx`: timepoints 0, 2, 4, 6, 8
(5–21 hpi), the central 32 of 48 Z slices, and the central 512×512 of the 640×960 field
of view. The crop is recorded in each position's `dynacell_demo` attributes and is
rebuilt by `build_demo_data.py` in
[VisCy](https://github.com/mehta-lab/VisCy/tree/dynacell-models/applications/dynacell/examples/hf_demo).
ER and mitochondria targets are raw (not deconvolved) fluorescence, as used for
training and evaluation.

## Layout

Each `.zip` unpacks to an OME-Zarr HCS store:

```
{marker}_mock.zarr/
  0/0/fov0006/0      # array (T, C, Z, Y, X) = (5, 3, 32, 512, 512), float32
                     # C = Phase3D (input), Brightfield, experimental fluorescence (target)
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

```bibtex
@inproceedings{kalinin2026dynacell,
  title     = {{DynaCell}: An Evaluation Framework for Dynamic {3D} Virtual Staining of Live Cells},
  author    = {Kalinin, Alexandr A. and Zheng, Dihan and Theodoro, Taylla Milena and
               Ivanov, Ivan and Hirata-Miyasaki, Eduardo and Lee, See-Chi and Liu, Aofei and
               Varra, Sricharan Reddy and Chandler, Talon and Pradeep, Soorya and Liu, Chad and
               Leonetti, Manuel D. and Arias, Carolina and Huang, Bo and Mehta, Shalin B.},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS 2026), Evaluations and Datasets Track},
  year      = {2026}
}
```
