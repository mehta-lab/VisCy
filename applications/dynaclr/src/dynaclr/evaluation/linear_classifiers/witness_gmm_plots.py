"""Stage-A witness-GMM diagnostic plots — the label-decision evidence.

Stage A (:mod:`dynaclr.evaluation.linear_classifiers.witness_gmm_labels`) turns
the MMD witness + per-condition GMM into an annotation file. These plots are the
*evidence* behind each labeling decision, written alongside the CSV so the call
(label / skip) is auditable:

- :func:`plot_witness_gmm` — per (marker, condition): the perturbed cells'
  witness-score histogram with the fitted two-component GMM overlaid (the
  bimodality the gate keys on), both posterior thresholds marked, and the
  ambiguous posterior interval shaded.
- :func:`plot_mmd_null` — per (marker, condition): the MMD² permutation-null
  histogram with the observed MMD² and p-value (the significance gate).
- :func:`plot_remodeling_vs_time` — per marker: the fraction of cells in the
  perturbed (remodeled/infected) class vs time, per condition — the biological
  kinetics the labels imply.
- :func:`plot_mmd_vs_hpi` — per marker: MMD²(control, condition) per HPI bin —
  the population-divergence kinetics (label-free), with a control-vs-control null
  band. Answers "when, and how much, does the population diverge from control?"
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
    control_scores: NDArray,
    result: GmmLabelResult,
    pos_threshold: float,
    marker: str,
    condition: str,
    output_path: Path,
    pos_label: str = "remodeled",
    neg_label: str = "unaffected",
    fit_population: str = "perturbed",
) -> None:
    """Plot perturbed vs control witness scores with the fitted GMM overlaid.

    Two histograms are overlaid so the reader can see *which population each GMM
    mode corresponds to*:

    - **perturbed** (grey): the condition's cells.
    - **control** (blue): the clean negative-reference cells (control wells).

    Under ``fit_population="perturbed"`` the blue distribution is an overlay only.
    Under ``"joint_control_perturbed"`` every score from both distributions fits
    one mixture; under ``"balanced_control_perturbed"`` equal-size samples from
    both distributions fit one mixture. The two fitted GMM component densities are drawn on
    top: the ``pos_label`` mode (red, lower mean, perturbed-leaning) and the
    ``neg_label`` mode (blue). A sound gate has the ``neg_label`` component sitting
    on top of the control histogram (both are control-like) and the ``pos_label``
    component pulled away toward negative scores. Dashed lines mark the
    confident-negative and confident-positive posterior boundaries; the
    interval between them is the abstention region.

    Parameters
    ----------
    scores : NDArray
        Witness scores for the condition's perturbed cells, shape ``(n,)``.
    control_scores : NDArray
        Witness scores of the control-reference cells, shape ``(m,)``. They are
        an overlay in perturbation-only mode and part of either joint fit mode.
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
    pos_label : str
        Class name for the positive (perturbed-leaning) mode, for the legend.
    neg_label : str
        Class name for the negative (control-like) mode, for the legend.
    fit_population : str
        Provenance label describing which populations fit the GMM.
    """
    scores = np.asarray(scores).ravel()
    control_scores = np.asarray(control_scores).ravel()
    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Shared bins across both populations so the overlaid histograms are comparable.
    lo = float(min(scores.min(), control_scores.min())) if control_scores.size else float(scores.min())
    hi = float(max(scores.max(), control_scores.max())) if control_scores.size else float(scores.max())
    bins = np.linspace(lo, hi, 61)
    ax.hist(scores, bins=bins, density=True, color="0.7", alpha=0.7, label=f"perturbed ({condition}, n={len(scores)})")
    if control_scores.size:
        ax.hist(
            control_scores,
            bins=bins,
            density=True,
            histtype="step",
            color="tab:blue",
            lw=1.5,
            label=f"control reference (n={len(control_scores)})",
        )

    grid = np.linspace(lo, hi, 400)
    means = result.gmm.means_.ravel()
    stds = result.component_stds
    weights = result.gmm.weights_.ravel()
    for k in range(len(means)):
        density = weights[k] / (stds[k] * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((grid - means[k]) / stds[k]) ** 2)
        is_pos = k == result.remod_component
        ax.plot(
            grid,
            density,
            lw=2,
            color="tab:red" if is_pos else "tab:cyan",
            label=f"GMM {pos_label if is_pos else neg_label} mode (w={weights[k]:.2f})",
        )

    # Scores at which the positive-component posterior clears the confident
    # positive and negative gates. The common two-mode case is monotone between
    # component means; shading is driven directly by posterior values so it also
    # remains correct if unequal variances produce a non-linear transition.
    grid_post = result.gmm.predict_proba(grid.reshape(-1, 1))[:, result.remod_component]
    neg_threshold = 1.0 - pos_threshold

    def _crossings(threshold: float) -> np.ndarray:
        delta = grid_post - threshold
        indices = np.flatnonzero(delta[:-1] * delta[1:] <= 0)
        values = []
        for index in indices:
            x0, x1 = grid[index : index + 2]
            y0, y1 = delta[index : index + 2]
            if y1 == y0:
                values.append(float((x0 + x1) / 2))
            else:
                values.append(float(x0 - y0 * (x1 - x0) / (y1 - y0)))
        return np.asarray(values, dtype=float)

    positive_boundaries = _crossings(pos_threshold)
    negative_boundaries = _crossings(neg_threshold)
    posterior_diff = np.diff(grid_post)
    nonmonotonic = bool(np.any(posterior_diff > 1e-6) and np.any(posterior_diff < -1e-6))

    y_max = ax.get_ylim()[1]
    ambiguous = (grid_post > neg_threshold) & (grid_post < pos_threshold)
    ax.fill_between(
        grid,
        0,
        y_max,
        where=ambiguous,
        color="#F2C94C",
        alpha=0.28,
        zorder=0,
        label=(f"ambiguous: {neg_threshold:g} < P({pos_label}) < {pos_threshold:g}"),
    )
    for index, boundary in enumerate(negative_boundaries):
        ax.axvline(
            boundary,
            color="tab:blue",
            ls="--",
            lw=1.3,
            label=(f"P({pos_label}) = {neg_threshold:g}" if index == 0 else None),
        )
    for index, boundary in enumerate(positive_boundaries):
        ax.axvline(
            boundary,
            color="tab:red",
            ls="--",
            lw=1.3,
            label=(f"P({pos_label}) = {pos_threshold:g}" if index == 0 else None),
        )
    ax.set_ylim(0, y_max)

    n_pos = int((result.hard_label == 1).sum())
    ax.set_title(
        f"Witness-GMM gate — {marker} / {condition}\n"
        f"{'bimodal' if result.separated else 'UNIMODAL (skipped)'} · "
        f"{n_pos}/{len(scores)} confident {pos_label} · yellow = abstain\n"
        f"fit population: {fit_population.replace('_', ' ')} · "
        f"posterior: {'NON-MONOTONIC' if nonmonotonic else 'monotonic'}"
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
    null_max = float(null.max())
    null_std = float(null.std())
    # Effect size: how many null-SD the observed value sits above the null mean.
    z = (observed_mmd2 - float(null.mean())) / null_std if null_std > 0 else np.inf

    # The null clusters near ~0 while the observed MMD² is far to the right; on one
    # shared axis the null collapses to an invisible sliver. Use a broken x-axis:
    # left panel zooms the null distribution, right panel marks the observed value.
    fig, (axl, axr) = plt.subplots(
        1, 2, figsize=(8, 4.5), sharey=True, gridspec_kw={"width_ratios": [3, 1], "wspace": 0.05}
    )
    axl.hist(null, bins=50, color="0.7", edgecolor="0.5", label="permutation null")
    axl.set_xlim(min(0.0, float(null.min())), null_max * 1.15 + 1e-9)
    axl.set_ylabel("count")
    axl.set_xlabel("MMD²  (null region)")
    axl.legend(fontsize=8, loc="upper right")

    axr.axvline(observed_mmd2, color="tab:red", lw=2, label=f"observed = {observed_mmd2:.3g}")
    pad = max(observed_mmd2 * 0.02, null_max)
    axr.set_xlim(observed_mmd2 - pad, observed_mmd2 + pad)
    axr.set_xlabel("observed")
    axr.legend(fontsize=8, loc="upper right")

    # Broken-axis diagonal marks between the two panels.
    d = 0.015
    for ax_, xs in ((axl, (1 - d, 1 + d)), (axr, (-d, d))):
        ax_.plot(xs, (-d, d), transform=ax_.transAxes, color="k", clip_on=False, lw=1)
        ax_.plot(xs, (1 - d, 1 + d), transform=ax_.transAxes, color="k", clip_on=False, lw=1)
    axl.spines["right"].set_visible(False)
    axr.spines["left"].set_visible(False)
    axr.tick_params(left=False)

    fig.suptitle(
        f"MMD significance — {marker} / {condition}\n"
        f"raw p={p_value:.3g} · BY-adj p={p_adjusted:.3g} · observed {z:.0f}σ above null · "
        f"{'SIGNIFICANT' if significant else 'not significant (skipped)'}",
        fontsize=11,
    )
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
    control_pos: NDArray | None,
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
    (``hard_label == 1``).

    The control reference is drawn dashed and is the **empirical false-positive
    rate**: the fraction of control-well cells whose witness score would clear the
    same GMM posterior threshold (``control_pos``), NOT a hardcoded zero. It
    should hug 0 if the gate is clean; a rise flags leakage into the negatives.
    Error bars are Bernoulli SEM per timepoint. Style follows the paper's
    ``plot_infection_state_vs_time`` figure.

    Parameters
    ----------
    cond_time : dict
        ``condition -> np.ndarray`` of the time value per perturbed cell (same
        order as the condition's GMM ``hard_label``). Empty → nothing plotted.
    cond_gmm : dict
        ``condition -> GmmLabelResult``; ``hard_label == 1`` marks confident
        positives among that condition's perturbed cells.
    control_time : NDArray or None
        Time value per control-reference cell. None → no baseline line.
    control_pos : NDArray or None
        Boolean per control cell: clears the GMM posterior threshold (would be
        a false positive). Aligned with ``control_time``. Drives the dashed line.
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

    if control_time is not None and len(control_time) and control_pos is not None:
        stats = _pct_positive_by_time(control_time, control_pos)
        fp_rate = 100.0 * float(np.mean(control_pos))
        ax.errorbar(
            stats["t"],
            stats["pct"],
            yerr=stats["sem"],
            linestyle="--",
            linewidth=1.2,
            alpha=0.7,
            color="0.4",
            label=f"control false-positive rate (n={int(stats['n'].sum())}, {fp_rate:.1f}%)",
        )

    ax.set_ylim(-5, 105)
    ax.set_title(f"% cells in '{positive_class}' vs time — {marker}", fontsize=11, fontweight="bold")
    ax.set_xlabel("hours post perturbation" if time_is_hpp else "timepoint", fontsize=11)
    ax.set_ylabel(f"% cells '{positive_class}'", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True, fontsize=9)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_mmd_vs_hpi(
    hpi_mmd: dict,
    pvalue_threshold: float,
    marker: str,
    output_path: Path,
) -> None:
    """Plot MMD²(control, condition) per HPI bin — population-divergence kinetics.

    Each condition curve traces how far its cell cloud sits from the control
    reference at each time window (label-free — no GMM, no per-cell labels). The
    ``__control_null__`` series (control split against itself) is drawn as a grey
    band: the no-difference floor the condition curves must exceed to be real.
    A condition curve that **grows** with HPI is the signature of a genuine,
    progressive perturbation; one that sits flat near the null (or as a constant
    offset that never grows) is weak/ambiguous — possibly a fixed well effect
    rather than a time-developing response. Points failing the per-bin
    significance test (raw p > ``pvalue_threshold``) are drawn hollow.

    Parameters
    ----------
    hpi_mmd : dict
        ``{condition: [(hpi_center, mmd2, p), ...]}`` from ``_compute_hpi_mmd``;
        key ``"__control_null__"`` is the control-vs-control baseline.
    pvalue_threshold : float
        Per-bin raw-p threshold for the hollow/filled marker distinction.
    marker : str
        Marker name (title).
    output_path : Path
        Output file path.
    """
    if not hpi_mmd:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    null = hpi_mmd.get("__control_null__")
    if null:
        arr = np.array(sorted(null))
        ax.plot(arr[:, 0], arr[:, 1], color="0.5", ls="--", lw=1.2, label="control vs control (null)")
        # Shade up to the max null MMD² as the no-difference band.
        ax.axhspan(0, float(arr[:, 1].max()), color="0.85", alpha=0.5, zorder=0)

    ci = 0
    for cond, series in hpi_mmd.items():
        if cond == "__control_null__":
            continue
        arr = np.array(sorted(series))
        centers, mmd2, pvals = arr[:, 0], arr[:, 1], arr[:, 2]
        color = colors[ci % len(colors)]
        ci += 1
        ax.plot(centers, mmd2, color=color, lw=1.8, label=f"control vs {cond}")
        sig = pvals <= pvalue_threshold
        ax.scatter(centers[sig], mmd2[sig], color=color, s=30, zorder=3)
        ax.scatter(centers[~sig], mmd2[~sig], facecolors="none", edgecolors=color, s=30, zorder=3, label="_nolegend_")

    ax.set_ylim(bottom=0)
    ax.set_title(f"MMD² vs time — {marker}\n(population divergence from control; hollow = n.s.)", fontsize=11)
    ax.set_xlabel("hours post perturbation", fontsize=11)
    ax.set_ylabel("MMD²  (control vs condition)", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True, fontsize=9)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_percentile_gate(
    perturbed_scores: NDArray,
    control_scores: NDArray,
    result,
    marker: str,
    condition: str,
    output_path: Path,
    pos_label: str = "remodel",
    neg_label: str = "noremodel",
) -> None:
    """Plot the control-calibrated percentile gate.

    Nothing is fit here, so unlike the witness-GMM plot there are no fitted
    component curves to draw. The plot shows the two score distributions and the
    two decision boundaries — both quantiles of the **control** scores — so the
    reader can judge the gate the way it is actually defined: by where the
    perturbed mass sits relative to cuts placed on the control distribution.

    The unshaded strip between the cuts is what gets dropped: cells resembling
    neither reference confidently.

    Read it by asking whether the perturbed histogram is *shifted* relative to
    control, not whether either is bimodal. That is the regime this gate is for;
    when a genuine second mode exists, :func:`plot_witness_gmm` is the better view.

    Parameters
    ----------
    perturbed_scores, control_scores : NDArray
        Witness scores of the condition's perturbed cells and the (time-matched)
        control-reference cells.
    result : PercentileLabelResult
        The fitted gate (``gate``, ``control_fp``, ``hard_label``).
    marker, condition : str
        For the title.
    output_path : Path
        Output file path.
    pos_label, neg_label : str
        Class names for the legend.
    """
    p = np.asarray(perturbed_scores).ravel()
    c = np.asarray(control_scores).ravel()
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    lo = float(min(p.min(), c.min()))
    hi = float(max(p.max(), c.max()))
    bins = np.linspace(lo, hi, 61)
    ax.hist(p, bins=bins, density=True, color="0.7", alpha=0.7, label=f"perturbed ({condition}, n={len(p)})")
    ax.hist(c, bins=bins, density=True, histtype="step", color="tab:blue", lw=1.5, label=f"control (n={len(c)})")

    ax.axvline(
        result.gate,
        color="k",
        ls="--",
        lw=1.5,
        label=f"{pos_label} cut = {result.gate:.4f}  (control FP {result.control_fp:.1%})",
    )
    ax.axvline(
        result.neg_gate,
        color="tab:blue",
        ls="--",
        lw=1.5,
        label=f"{neg_label} cut = {result.neg_gate:.4f}",
    )
    # Shade both labeled regions; the unshaded middle is what gets dropped.
    ax.axvspan(lo, result.gate, color="tab:red", alpha=0.08, label=f"{pos_label} region")
    ax.axvspan(result.neg_gate, hi, color="tab:blue", alpha=0.08, label=f"{neg_label} region")

    n_pos = int((result.hard_label == 1).sum())
    n_neg = int((result.hard_label == 0).sum())
    n_amb = int((result.hard_label == -1).sum())
    # Report where each cut actually falls on the control distribution, measured
    # rather than assumed — the negative cut is NOT the mirror of the positive one
    # (see fit_percentile_labels), so it cannot be derived from control_fp.
    pos_pctl = 100.0 * float((c < result.gate).mean())
    neg_pctl = 100.0 * float((c < result.neg_gate).mean())
    ax.set_title(
        f"Percentile gate — {marker} / {condition}\n"
        f"{n_pos} {pos_label} · {n_neg} {neg_label} · {n_amb} ambiguous (dropped) "
        f"of {len(p)} · cuts = control p{pos_pctl:.0f}/p{neg_pctl:.0f}"
    )
    ax.set_xlabel("witness score  (negative = perturbed-leaning)")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
