## Preprocessing output format

The preprocessing step performs the following steps.

* Segments the target images using selected segmentation algorithm in the configuration file(otsu, mem_detection, unimodal).

* Stores the mask output as an extra channel in the zarr store with the name of the segmented channel with added subscript '_mask'. For instance, if the user segments the channel named 'Deconvolved-Nuc', the mask channel added will be called 'Deconvolved-Nuc_mask'. The datatype is the same as the channel which is segmented (most possibly float32).

* Stores the information related to normalization of all input channels mentioned in the configuration. The dataset statistics are stored in the plate level .zattrs file, while the information specific to the position is added in the .zattrs file at each position. The details are explained below.

Here is the structure of a 0.4 NGFF version HCS format zarr store wriiten using [iohub](https://github.com/czbiohub/iohub) for a dataset with a single condition and multiple imaging FOVs.

```text
.                             # Root folder
│
└── my_zarr_name.zarr         # Zarr folder name
    ├── .zgroup
    ├── .zattrs               # Implements "plate" specification
    ├── FOVs                  # Named 'FOVs' to indicate different FOVs inside
    │   ├── .zgroup
    │   │
    │   ├── 000               # First FOV
    │   │   ├── .zgroup
    │   │   ├── .zattrs       # Implements "well" specification
    │   │   │
    │   │   ├── 0
    │   │   │   │
    │   │   │   ├── .zgroup
    │   │   │   ├── .zattrs   # Implements "multiscales", "omero"
    │   │   │   ├── 0         # (T, C, Z, Y, X) float32
    │   │   │   │   ...       # Resolution levels
    │   │   │   ├── n
    │   │   │   └── labels    # Labels (optional)
    |   |
    |   ├── 001               # Second FOV

 ```

Here the dataset statistics is stored inside the 'plate' folder and the position statistics is stored in '.zattrs' inside plate/A/1/0 folder.

If the dataset contains multiple conditions from different wells the structure can be as follows.

```text
.                             # Root folder
│
└── my_zarr_name.zarr         # Zarr folder level
    ├── .zgroup
    ├── .zattrs               # Implements "plate" specification
    ├── A                     # First row of the plate
    │   ├── .zgroup
    │   │
    │   ├── 1                 # First column (well A1 in plate)
    │   │   ├── .zgroup
    │   │   ├── .zattrs       # Implements "well" specification
    │   │   │
    │   │   ├── 0             # First field of view of well A1
    │   │   │   │
    │   │   │   ├── .zgroup
    │   │   │   ├── .zattrs   # Implements "multiscales", "omero"
    │   │   │   ├── 0         # (T, C, Z, Y, X) float32
    │   │   │   │   ...       # Resolution levels
    │   │   │   ├── n
    │   │   │   └── labels    # Labels (optional)

 ```

The statistics are added as dictionaries into the .zattrs file. An example of plate level metadata is here:

```json
    "normalization": {
        "Deconvolved-Nuc": {
            "dataset_statistics": {
                "iqr": 149.7620086669922,
                "mean": 262.2070617675781,
                "median": 65.5246353149414,
                "std": 890.0471801757812
            }
        },
        "Phase3D": {
            "dataset_statistics": {
                "iqr": 0.0011349652777425945,
                "mean": -1.9603044165705796e-06,
                "median": 3.388232289580628e-05,
                "std": 0.005480962339788675
            }
        }
    }
```

FOV level statistics added to every position as well as the dataset_statistics to read dataset statistics:

```json
    "normalization": {
        "Deconvolved-Nuc": {
            "dataset_statistics": {
                "iqr": 149.7620086669922,
                "mean": 262.2070617675781,
                "median": 65.5246353149414,
                "std": 890.0471801757812
            },
            "fov_statistics": {
                "iqr": 450.4745788574219,
                "mean": 486.3854064941406,
                "median": 83.43557739257812,
                "std": 976.02392578125
            }
        },
        "Phase3D": {           
            "dataset_statistics": {
                "iqr": 0.0011349652777425945,
                "mean": -1.9603044165705796e-06,
                "median": 3.388232289580628e-05,
                "std": 0.005480962339788675
            },
            "fov_statistics": {
                "iqr": 0.006403466919437051,
                "mean": 0.0010083537781611085,
                "median": 0.00022060875198803842,
                "std": 0.007864165119826794
            }
        }
