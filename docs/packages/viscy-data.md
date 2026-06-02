# viscy-data

:material-database-outline:{ .lg .middle } Data loading and Lightning
`DataModule`s for AI × imaging tasks.

=== ":material-language-python: pip"

    ```sh
    pip install viscy-data
    ```

=== ":material-package-variant-closed: uv"

    ```sh
    uv add viscy-data
    ```

!!! abstract "What's here"

    Lightning `DataModule`s for OME-Zarr microscopy — HCS plates and wells,
    triplet sampling for contrastive learning, and memory-mapped caching for
    terabyte-scale datasets.

!!! info "Optional extras"

    `viscy-data[triplet]` adds tensorstore-backed triplet sampling ·
    `viscy-data[livecell]` adds LiveCell dataset support ·
    `viscy-data[mmap]` adds memory-mapped caching · `viscy-data[all]` enables all.

## API reference

::: viscy_data
