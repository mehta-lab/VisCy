"""Per-well channel selection and rename for A549 mantis assembly.

Each plate carries instrument-named channels (e.g. ``GFP EX488 EM525-45``)
whose biological meaning depends on the well's construct. The assembly
step renames those channels to the iPSC-shaped canonical names
(``Structure``, ``Nuclei``, ``Membrane``) expected by VisCy's
dataset_v4 convention.

Two target flavours are supported:

- **Single-gene targets** (``sec61b``, ``tomm20``, ``h2b``, ``caax``):
  one gene → one canonical output channel. ER/mito (``sec61b`` /
  ``tomm20``) emit BOTH the raw and the deconvolved GFP as separate
  output channels (``Structure`` = raw, ``Structure_deconvolved`` =
  deconvolved). Nuclei/Membrane (``h2b`` / ``caax``) are raw-only.
- **Combined multi-gene targets** (``dual_nucl_memb``): several genes are
  routed into one pool, each to its own canonical channel. Used to merge
  the co-imaged nucleus + membrane v2 plate into a single store
  ``[Phase3D, Brightfield, Nuclei, Membrane]``.
"""

from dataclasses import dataclass

CANONICAL_TARGET_CHANNELS: dict[str, str] = {
    "sec61b": "Structure",
    "tomm20": "Structure",
    "h2b": "Nuclei",
    "caax": "Membrane",
}
"""Gene → canonical output channel name. Matches ``aics-hipsc`` layout."""

COMBINED_TARGETS: dict[str, tuple[str, ...]] = {
    "dual_nucl_memb": ("h2b", "caax"),
}
"""Combined target key → ordered constituent genes.

Each combined target routes several genes into one pool. Output channel
order is ``[<passthrough...>, <gene_0 canonical>, <gene_1 canonical>, ...]``
following the constituent-gene order listed here. ``dual_nucl_memb`` →
``[Phase3D, Brightfield, Nuclei, Membrane]`` (both v2 fluor genes,
co-imaged on the same FOV lattice).
"""

PASSTHROUGH_CHANNELS: tuple[str, ...] = ("Phase3D", "BF", "Brightfield")
"""Source/auxiliary channels that pass through to the output unchanged."""

RAW_PREFIX: str = "raw "
"""Instrument prefix marking a raw (camera-offset, un-deconvolved) channel."""

DECONVOLVED_SUFFIX: str = "_deconvolved"
"""Canonical-name suffix for the deconvolved counterpart of a raw target."""

# Genes whose target is emitted as BOTH raw + deconvolved output channels.
# The platemap ``gene_channel_map`` maps the BARE (deconvolved) native
# name for these genes; the raw counterpart (``raw <bare>``) is required
# on the plate and emitted as ``<canonical>`` while the deconvolved one
# is emitted as ``<canonical>_deconvolved``.
_RAW_AND_DECONV_GENES: frozenset[str] = frozenset({"sec61b", "tomm20"})


@dataclass(frozen=True)
class ChannelSelection:
    """Resolved channel plan for one well + one target."""

    input_indices: list[int]
    """Native zarr channel indices to select, in output order."""

    output_names: list[str]
    """Output channel names aligned with ``input_indices``."""


def _passthrough_selection(
    native_channel_names: list[str],
) -> tuple[list[int], list[str]]:
    """Return indices + output names for passthrough channels, in order."""
    input_indices: list[int] = []
    output_names: list[str] = []
    for passthrough in PASSTHROUGH_CHANNELS:
        if passthrough in native_channel_names:
            input_indices.append(native_channel_names.index(passthrough))
            output_names.append("Brightfield" if passthrough == "BF" else passthrough)
    return input_indices, output_names


def _native_name_for_gene(gene_channel_map: dict[str, str], gene: str) -> str:
    """Return the native channel name mapped to ``gene`` in the platemap.

    Raises
    ------
    ValueError
        If ``gene`` is not present as a value in ``gene_channel_map``.
    """
    for native_name, mapped_gene in gene_channel_map.items():
        if mapped_gene == gene:
            return native_name
    raise ValueError(f"gene_channel_map does not map any native channel to gene={gene!r}. Map: {gene_channel_map}")


def _append_gene_channels(
    native_channel_names: list[str],
    native_for_gene: str,
    gene: str,
    input_indices: list[int],
    output_names: list[str],
) -> None:
    """Append the target channel(s) for one gene to the running selection.

    For ER/mito genes (``_RAW_AND_DECONV_GENES``) the platemap maps the
    BARE deconvolved name; both raw (required) and deconvolved are
    emitted, raw first as the canonical channel and deconvolved as
    ``<canonical>_deconvolved``. For all other genes the single mapped
    native channel is copied verbatim to the canonical name.

    Raises
    ------
    ValueError
        If a required native channel is absent on the plate.
    """
    canonical = CANONICAL_TARGET_CHANNELS[gene]

    if gene in _RAW_AND_DECONV_GENES:
        # Platemap maps the deconvolved (bare) name; derive the raw one.
        deconv_native = native_for_gene
        raw_native = f"{RAW_PREFIX}{deconv_native}"
        if raw_native not in native_channel_names:
            raise ValueError(
                f"raw counterpart {raw_native!r} for gene={gene!r} not found in "
                f"plate channels {native_channel_names}; raw is required for the "
                f"raw+deconvolved target"
            )
        if deconv_native not in native_channel_names:
            raise ValueError(
                f"deconvolved channel {deconv_native!r} for gene={gene!r} not "
                f"found in plate channels {native_channel_names}"
            )
        input_indices.append(native_channel_names.index(raw_native))
        output_names.append(canonical)
        input_indices.append(native_channel_names.index(deconv_native))
        output_names.append(f"{canonical}{DECONVOLVED_SUFFIX}")
    else:
        if native_for_gene not in native_channel_names:
            raise ValueError(f"Native channel {native_for_gene!r} not found in plate channels {native_channel_names}")
        input_indices.append(native_channel_names.index(native_for_gene))
        output_names.append(canonical)


def resolve_channels(
    native_channel_names: list[str],
    gene_channel_map: dict[str, str],
    target: str,
) -> ChannelSelection:
    """Resolve native channels for a target (single-gene or combined).

    Parameters
    ----------
    native_channel_names : list of str
        Channel names on the plate zarr (position's channel_names).
    gene_channel_map : dict of str to str
        Native channel name → gene key (as authored in the platemap YAML).
        For ER/mito the mapped native name is the BARE (deconvolved) GFP
        name; the ``raw <name>`` counterpart is derived. Example:
        ``{"GFP EX488 EM525-45": "sec61b"}``.
    target : str
        Single gene key in ``CANONICAL_TARGET_CHANNELS`` (``sec61b``,
        ``tomm20``, ``h2b``, ``caax``) or a combined key in
        ``COMBINED_TARGETS`` (``dual_nucl_memb``).

    Returns
    -------
    ChannelSelection
        Indices into the native channel list and aligned output names.
        Order: ``[Phase3D, Brightfield(if present), <target channels…>]``.
        For ER/mito the target block is ``[Structure, Structure_deconvolved]``
        (raw then deconvolved); for combined targets it is one canonical
        channel per constituent gene, in ``COMBINED_TARGETS`` order.

    Raises
    ------
    ValueError
        If ``target`` is unknown, a constituent gene isn't mapped for this
        well, or a required native channel doesn't exist on the plate.
    """
    genes = resolve_target_genes(target)

    input_indices, output_names = _passthrough_selection(native_channel_names)
    for gene in genes:
        native_for_gene = _native_name_for_gene(gene_channel_map, gene)
        _append_gene_channels(
            native_channel_names,
            native_for_gene,
            gene,
            input_indices,
            output_names,
        )

    return ChannelSelection(input_indices=input_indices, output_names=output_names)


def resolve_target_genes(target: str) -> tuple[str, ...]:
    """Return the ordered constituent gene(s) for a target key.

    Single-gene targets return a 1-tuple; combined targets return the
    tuple registered in ``COMBINED_TARGETS``.

    Raises
    ------
    ValueError
        If ``target`` is neither a canonical single gene nor a combined key.
    """
    if target in COMBINED_TARGETS:
        return COMBINED_TARGETS[target]
    if target in CANONICAL_TARGET_CHANNELS:
        return (target,)
    raise ValueError(
        f"target={target!r} is not a known single-gene target "
        f"({sorted(CANONICAL_TARGET_CHANNELS)}) or combined target "
        f"({sorted(COMBINED_TARGETS)})"
    )
