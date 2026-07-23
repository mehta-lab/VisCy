"""Stage-A witness-GMM diagnostic plots — the label-decision evidence.

Stage A (:mod:`dynaclr.evaluation.linear_classifiers.witness_gmm_labels`) turns
the MMD witness + per-condition GMM into an annotation file. These plots are the
*evidence* behind each labeling decision, written alongside the CSV so the call
(label / skip) is auditable:

- :func:`plot_witness_gmm` — per (marker, condition): the perturbed cells'
  witness-score histogram with the fitted two-component GMM overlaid (the
  bimodality the gate keys on) and the positive-posterior threshold marked.
- :func:`plot_mmd_null` — per (marker, condition): the MMD² permutation-null
  histogram with the observed MMD² and p-value (the significance gate).
- :func:`plot_remodeling_vs_time` — per marker: the fraction of cells in the
  perturbed (remodeled/infected) class vs time, per condition — the biological
  kinetics the labels imply.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from viscy_utils.evaluation.witness_gmm import GmmLabelResult


def plot_witness_gmm(
    scores: NDArray,
    result: GmmLabelResult,
    pos_threshold: float,
    marker: str,
    condition: str,
    output_path: Path,
) -> None:
    """Plot perturbed-cell witness scores with the fitted GMM overlaid.

    Shows the histogram of the condition's witness scores, the two Gaussian
    component densities, and the posterior-threshold decision boundary (cells
    whose remodeled-component posterior clears ``pos_threshold`` are the
    confident positives). This is the bimodality the GMM gate reads.

    Parameters
    ----------
    scores : NDArray
        Witness scores for the condition's perturbed cells, shape ``(n,)``.
    result : GmmLabelResult
        The fitted GMM result for this condition.
    pos_threshold : float
        Posterior threshold used to call confident positives.
    marker : str
        Marker name (title).
    condition : str
        Condition name (title).
    output_path : Path
        Output file path.
    """
    scores = np.asarray(scores).ravel()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(scores, bins=60, density=True, color="0.7", edgecolor="white", label="witness scores")

    grid = np.linspace(scores.min(), scores.max(), 400)
    means = result.gmm.means_.ravel()
    stds = np.sqrt(result.gmm.covariances_.ravel())
    weights = result.gmm.weights_.ravel()
    for k in range(len(means)):
        density = weights[k] / (stds[k] * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((grid - means[k]) / stds[k]) ** 2)
        is_remod = k == result.remod_component
        ax.plot(
            grid,
            density,
            lw=2,
            color="tab:red" if is_remod else "tab:blue",
            label=f"{'remodeled' if is_remod else 'unaffected'} (w={weights[k]:.2f})",
        )

    # Score at which the remodeled-component posterior equals pos_threshold — the
    # decision boundary, found on the score grid (posterior is monotone in score
    # for a two-component 1-D GMM).
    grid_post = result.gmm.predict_proba(grid.reshape(-1, 1))[:, result.remod_component]
    crossing = grid[grid_post >= pos_threshold]
    if crossing.size:
        boundary = crossing.max() if means[result.remod_component] < means.mean() else crossing.min()
        ax.axvline(boundary, color="k", ls="--", lw=1, label=f"posterior ≥ {pos_threshold:g}")

    n_pos = int((result.hard_label == 1).sum())
    ax.set_title(
        f"Witness-GMM gate — {marker} / {condition}\n"
        f"{'bimodal' if result.separated else 'UNIMODAL (skipped)'} · "
        f"{n_pos}/{len(scores)} confident positives"
    )
    ax.set_xlabel("witness score  (negative = perturbed-leaning)")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_mmd_null(
    observed_mmd2: float,
    null: NDArray,
    p_value: float,
    p_adjusted: float,
    significant: bool,
    marker: str,
    condition: str,
    output_path: Path,
) -> None:
    """Plot the MMD² permutation-null with the observed statistic and p-values.

    The significance gate: is this condition's cloud distinct from the control
    reference? The observed MMD² is marked against its permutation null; the raw
    and Benjamini-Yekutieli-adjusted p-values and the gate verdict are annotated.

    Parameters
    ----------
    observed_mmd2 : float
        Observed unbiased MMD² (control vs condition).
    null : NDArray
        Permutation-null MMD² values, shape ``(n_permutations,)``.
    p_value : float
        Raw permutation p-value.
    p_adjusted : float
        Benjamini-Yekutieli-adjusted p-value (run-wide FDR).
    significant : bool
        Whether the condition cleared the FDR gate.
    marker : str
        Marker name (title).
    condition : str
        Condition name (title).
    output_path : Path
        Output file path.
    """
    null = np.asarray(null).ravel()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(null, bins=50, color="0.7", edgecolor="white", label="permutation null")
    ax.axvline(observed_mmd2, color="tab:red", lw=2, label=f"observed MMD² = {observed_mmd2:.3g}")
    ax.set_title(
        f"MMD significance — {marker} / {condition}\n"
        f"raw p={p_value:.3g} · BY-adj p={p_adjusted:.3g} · "
        f"{'SIGNIFICANT' if significant else 'not significant (skipped)'}"
    )
    ax.set_xlabel("MMD²")
    ax.set_ylabel("count")
    ax.legend(fontsize=8)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _pct_positive_by_time(times: NDArray, is_pos: NDArray) -> pd.DataFrame:
    """Per-timepoint % positive ± Bernoulli SEM over the given cell population.

    Groups by timepoint and returns, per timepoint, ``100·k/n`` and its Bernoulli
    standard error ``100·sqrt(p(1-p)/n)`` — the standard binomial error bar used
    for a per-timepoint positive rate (matches the paper figure convention).
    """
    df = pd.DataFrame({"t": np.asarray(times), "is_pos": np.asarray(is_pos, dtype=float)})
    rows = []
    for t, grp in df.groupby("t"):
        n = len(grp)
        p = float(grp["is_pos"].mean())
        rows.append({"t": t, "pct": 100.0 * p, "sem": 100.0 * np.sqrt(p * (1 - p) / n), "n": n})
    return pd.DataFrame(rows).sort_values("t")


def plot_remodeling_vs_time(
    cond_time: dict,
    cond_gmm: dict,
    control_time: NDArray | None,
    positive_class: str,
    marker: str,
    output_path: Path,
    time_is_hpp: bool = True,
) -> None:
    """Plot % of the *whole well population* in the positive class vs time ± SEM.

    For each perturbed condition, the denominator at each timepoint is **every**
    perturbed-well cell (labeled positive plus the GMM-dropped ambiguous ones),
    not only the confident positives — plotting positives / all-perturbed-cells is
    what reveals the remodeling/infection *rise* over time (the fraction over
    labeled cells alone is flat ~100% by construction, since the gate keeps only
    positives). The numerator is the GMM confident-positive count
    (``hard_label == 1``). The control reference is drawn dashed as the ~0%
    baseline. Error bars are Bernoulli SEM per timepoint. Style follows the
    paper's ``plot_infection_state_vs_time`` figure.

    Parameters
    ----------
    cond_time : dict
        ``condition -> np.ndarray`` of the time value per perturbed cell (same
        order as the condition's GMM ``hard_label``). Empty → nothing plotted.
    cond_gmm : dict
        ``condition -> GmmLabelResult``; ``hard_label == 1`` marks confident
        positives among that condition's perturbed cells.
    control_time : NDArray or None
        Time value per control-reference cell (the 0% baseline, drawn dashed).
        None → no baseline line.
    positive_class : str
        The positive class value (e.g. ``"infected"``), for labeling.
    marker : str
        Marker name (title).
    output_path : Path
        Output file path.
    time_is_hpp : bool
        Whether the time axis is ``hours_post_perturbation`` (else raw timepoint
        ``t``), for the axis label. By default True.
    """
    if not cond_time:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, (condition, times) in enumerate(cond_time.items()):
        res = cond_gmm.get(condition)
        if res is None:
            continue
        stats = _pct_positive_by_time(times, res.hard_label == 1)
        ax.errorbar(
            stats["t"],
            stats["pct"],
            yerr=stats["sem"],
            marker="o",
            markersize=4,
            capsize=2,
            linewidth=1.8,
            color=colors[i % len(colors)],
            label=f"{condition} (n={int(stats['n'].sum())})",
        )

    if control_time is not None and len(control_time):
        stats = _pct_positive_by_time(control_time, np.zeros(len(control_time)))
        ax.errorbar(
            stats["t"],
            stats["pct"],
            yerr=stats["sem"],
            linestyle="--",
            linewidth=1.0,
            alpha=0.6,
            color="0.4",
            label=f"control (n={int(stats['n'].sum())})",
        )

    ax.set_ylim(-5, 105)
    ax.set_title(f"% cells in '{positive_class}' vs time — {marker}", fontsize=11, fontweight="bold")
    ax.set_xlabel("hours post perturbation" if time_is_hpp else "timepoint", fontsize=11)
    ax.set_ylabel(f"% cells '{positive_class}'", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True, fontsize=9)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
