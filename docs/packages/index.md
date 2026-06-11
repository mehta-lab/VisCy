# Packages

VisCy ships as focused, independently versioned packages. Each is published to
PyPI and versioned from its own `viscy-<name>-vX.Y.Z` git tag,[^tags] so one
package can be released without bumping the others.

[^tags]:
    Versions are derived at build time by `uv-dynamic-versioning` from each
    package's prefixed git tag — the table below reflects whatever the docs were
    built against.

## At a glance

The table below is regenerated at build time from the installed distributions
(`docs/_gen_versions.py`):

<!-- versions:start -->

| Package | Version | Install |
|---------|---------|---------|
| [`viscy-data`](viscy-data.md) | `0.0.0.post209.dev0+4edfaf8b` | `pip install viscy-data` |
| [`viscy-models`](viscy-models.md) | `0.0.0.post209.dev0+4edfaf8b` | `pip install viscy-models` |
| [`viscy-transforms`](viscy-transforms.md) | `0.0.0.post209.dev0+4edfaf8b` | `pip install viscy-transforms` |
| [`viscy-utils`](viscy-utils.md) | `0.0.0.post209.dev0+4edfaf8b` | `pip install viscy-utils` |

<!-- versions:end -->

## Dependency layering

Three base packages have no internal dependencies; `viscy-utils` builds on
`viscy-data`. Everything downstream (the umbrella and the applications) composes
all four.

## What each package provides

`viscy-data`
:   Data loading and Lightning `DataModule`s for OME-Zarr microscopy datasets —
    HCS plates, triplet sampling, memory-mapped caches.
    [:octicons-arrow-right-24: Reference](viscy-data.md)

`viscy-models`
:   Neural network architectures: UNet variants, contrastive encoders, and VAEs.
    [:octicons-arrow-right-24: Reference](viscy-models.md)

`viscy-transforms`
:   GPU-friendly image transforms tuned for virtual staining microscopy.
    [:octicons-arrow-right-24: Reference](viscy-transforms.md)

`viscy-utils`
:   Shared ML infrastructure — the glue used across the stack.
    [:octicons-arrow-right-24: Reference](viscy-utils.md)
