"""Per-plate, HPI-binned control-reference normalization for embeddings.

Removes the plate (batch) and time-of-experiment component of an embedding by
expressing every cell relative to the *control* (untreated) population of the
same plate at the same hours-post-perturbation (HPI).

The reference statistics are robust to outliers (median and scaled median
absolute deviation, MAD) and computed per ``experiment`` (plate), pooling all
control wells, in fixed-width HPI bins. Only the small statistics table is
stored; the robust z-score ``(x - median[bin]) / mad[bin]`` is applied on the fly.
"""

from dataclasses import dataclass

import anndata as ad
import numpy as np


@dataclass
class ControlReference:
    """Robust control-reference statistics for one plate at one bin width.

    Parameters
    ----------
    bin_hours : float
        Width of the HPI bins, anchored at 0 h.
    bin_indices : np.ndarray
        ``(n_bins,)`` sorted integer indices of the occupied bins
        (``bin = floor(hpi / bin_hours)``).
    median : np.ndarray
        ``(n_bins, D)`` per-dimension control median for each occupied bin.
    mad : np.ndarray
        ``(n_bins, D)`` per-dimension control scaled median absolute deviation
        (``1.4826 * median(|x - median|)``, so it matches the standard deviation
        for normal data) for each occupied bin. Zero entries are floored to 1.0
        so the robust z-score never divides by zero.
    """

    bin_hours: float
    bin_indices: np.ndarray
    median: np.ndarray
    mad: np.ndarray

    def _reference_row(self, bins: np.ndarray) -> np.ndarray:
        """Map each requested bin to a row in ``bin_indices``, nearest occupied bin as fallback."""
        occupied = self.bin_indices
        rows = np.searchsorted(occupied, bins)
        rows = np.clip(rows, 0, len(occupied) - 1)
        # searchsorted lands on the right neighbour; check the left one too and keep the closer.
        left = np.clip(rows - 1, 0, len(occupied) - 1)
        pick_left = np.abs(occupied[left] - bins) < np.abs(occupied[rows] - bins)
        rows[pick_left] = left[pick_left]
        return rows

    def apply(self, x: np.ndarray, hpi: np.ndarray) -> np.ndarray:
        """Robust-z-score ``x`` against the matched control bin.

        Parameters
        ----------
        x : np.ndarray
            ``(N, D)`` embeddings to normalize.
        hpi : np.ndarray
            ``(N,)`` hours-post-perturbation for each row of ``x``.

        Returns
        -------
        np.ndarray
            ``(N, D)`` robust z-scored embeddings. A bin with no control cells
            borrows statistics from the nearest occupied bin.
        """
        bins = np.floor(np.asarray(hpi) / self.bin_hours).astype(int)
        rows = self._reference_row(bins)
        mad = np.where(self.mad[rows] == 0, 1.0, self.mad[rows])
        return (x - self.median[rows]) / mad


def control_reference_stats(
    adata: ad.AnnData,
    *,
    bin_hours: float,
    experiment_key: str = "experiment",
    perturbation_key: str = "perturbation",
    control_value: str = "uninfected",
    hpi_key: str = "hours_post_perturbation",
    min_cells: int = 20,
) -> dict[str, ControlReference]:
    """Compute per-plate control-reference statistics at one bin width.

    For each plate (``experiment``), keeps only control cells
    (``perturbation == control_value``), bins them by HPI into fixed-width bins
    anchored at 0 h, and records the per-dimension median and scaled median
    absolute deviation (MAD) of the embedding ``X`` in each bin. Bins with fewer
    than ``min_cells`` control cells are dropped; :meth:`ControlReference.apply`
    borrows the nearest occupied bin for them.

    Parameters
    ----------
    adata : ad.AnnData
        Embeddings with ``X`` of shape ``(N, D)`` and ``obs`` carrying
        ``experiment_key``, ``perturbation_key`` and ``hpi_key``.
    bin_hours : float
        Width of the HPI bins.
    experiment_key : str, optional
        obs column identifying the plate, by default ``"experiment"``.
    perturbation_key : str, optional
        obs column identifying the perturbation, by default ``"perturbation"``.
    control_value : str, optional
        Value of ``perturbation_key`` marking control cells, by default
        ``"uninfected"``.
    hpi_key : str, optional
        obs column with hours-post-perturbation, by default
        ``"hours_post_perturbation"``.
    min_cells : int, optional
        Minimum control cells for a bin to be kept, by default 20.

    Returns
    -------
    dict[str, ControlReference]
        Mapping from plate name to its :class:`ControlReference`.
    """
    obs = adata.obs
    x = np.asarray(adata.X)
    is_control = (obs[perturbation_key] == control_value).to_numpy()
    if not is_control.any():
        raise ValueError(f"No control cells found ({perturbation_key} == {control_value!r}).")

    # Every plate present in the data must have controls, or its perturbed cells
    # would have no matched reference and be silently dropped.
    all_plates = set(obs[experiment_key].unique())
    control_plates = set(obs[experiment_key][is_control].unique())
    plates_without_control = all_plates - control_plates
    if plates_without_control:
        raise ValueError(
            f"Plates with no control ({perturbation_key} == {control_value!r}) cells: "
            f"{sorted(map(str, plates_without_control))}. Every plate needs a control reference."
        )

    references: dict[str, ControlReference] = {}

    for plate in sorted(control_plates, key=str):
        plate_mask = is_control & (obs[experiment_key] == plate).to_numpy()
        plate_x = x[plate_mask]
        plate_hpi = obs[hpi_key].to_numpy()[plate_mask]
        plate_bins = np.floor(plate_hpi / bin_hours).astype(int)

        bin_indices = []
        medians = []
        mads = []
        for b in np.unique(plate_bins):
            bin_x = plate_x[plate_bins == b]
            if len(bin_x) < min_cells:
                continue
            med = np.median(bin_x, axis=0)
            # Scaled MAD: 1.4826 * median(|x - median|) matches sigma for normal data.
            mad = 1.4826 * np.median(np.abs(bin_x - med), axis=0)
            mad[mad == 0] = 1.0
            bin_indices.append(int(b))
            medians.append(med)
            mads.append(mad)

        if not bin_indices:
            raise ValueError(f"Plate {plate!r}: no HPI bin has >= {min_cells} control cells at bin_hours={bin_hours}.")

        references[str(plate)] = ControlReference(
            bin_hours=bin_hours,
            bin_indices=np.array(bin_indices),
            median=np.stack(medians),
            mad=np.stack(mads),
        )

    return references


def control_reference_stats_multi(
    adata: ad.AnnData,
    *,
    bin_hours: tuple[float, ...] = (1.0, 2.0),
    **kwargs,
) -> dict[float, dict[str, ControlReference]]:
    """Compute control-reference statistics at several bin widths.

    Convenience wrapper that runs :func:`control_reference_stats` for each bin
    width and keys the result by bin width, so a caller can pick 1 h or 2 h
    bins from one call.

    Parameters
    ----------
    adata : ad.AnnData
        Embeddings, as for :func:`control_reference_stats`.
    bin_hours : tuple[float, ...], optional
        Bin widths to compute, by default ``(1.0, 2.0)``.
    **kwargs
        Forwarded to :func:`control_reference_stats`.

    Returns
    -------
    dict[float, dict[str, ControlReference]]
        ``{bin_hours: {plate: ControlReference}}``.
    """
    return {bh: control_reference_stats(adata, bin_hours=bh, **kwargs) for bh in bin_hours}
