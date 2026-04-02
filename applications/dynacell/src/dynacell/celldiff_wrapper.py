"""Flow-matching virtual staining wrapper for CELLDiffNet.

Wraps the :class:`~viscy_models.celldiff.CELLDiffNet` backbone with
flow-matching transport to provide training loss computation and
ODE-based generation (single-patch, non-overlapping tiles, sliding window).

This module belongs in the application layer because it owns training
semantics (transport sampling, path planning, loss aggregation).
The reusable backbone and transport numerics live in ``viscy-models``.
"""

import itertools

import torch
from torch import Tensor, nn

from viscy_models.celldiff import CELLDiffNet
from viscy_models.celldiff.modules.transport import Sampler, create_transport


class CELLDiff3DVS(nn.Module):
    """Flow-matching virtual staining model.

    Wraps a :class:`CELLDiffNet` backbone with a flow-matching transport to
    provide training loss computation and inference (generation) methods.

    Parameters
    ----------
    net : CELLDiffNet
        Backbone network for velocity prediction.
    path_type : str
        Flow path type, e.g. ``"Linear"``.
    prediction : str
        Prediction target, e.g. ``"velocity"``.
    loss_weight : str or None
        Optional loss weighting scheme (``"velocity"`` or ``"likelihood"``).
    train_eps : float or None
        Training epsilon for transport stability.
    sample_eps : float or None
        Sampling epsilon for transport stability.
    """

    def __init__(
        self,
        net: CELLDiffNet,
        path_type: str = "Linear",
        prediction: str = "velocity",
        loss_weight: str | None = None,
        train_eps: float | None = None,
        sample_eps: float | None = None,
    ) -> None:
        super().__init__()
        self.net = net
        self.transport = create_transport(path_type, prediction, loss_weight, train_eps, sample_eps)
        self.transport_sampler = Sampler(self.transport)

    def forward(self, phase: Tensor, target: Tensor) -> Tensor:
        """Compute flow-matching training loss.

        Parameters
        ----------
        phase : Tensor
            Phase contrast input of shape ``(B, 1, D, H, W)``.
        target : Tensor
            Fluorescence target of shape ``(B, C, D, H, W)``.

        Returns
        -------
        Tensor
            Scalar training loss.
        """
        t, x0, x1 = self.transport.sample(target)
        t, xt, ut = self.transport.path_sampler.plan(t, x0, x1)
        pred = self.net(xt, phase, t)
        loss_dict = self.transport.training_losses(pred, x0, x1, xt, ut, t)
        return loss_dict["loss"].mean()

    def generate(self, phase: Tensor, num_steps: int = 100) -> Tensor:
        """Generate virtual staining via ODE sampling.

        Parameters
        ----------
        phase : Tensor
            Phase contrast input of shape ``(B, 1, D, H, W)``.
        num_steps : int
            Number of ODE integration steps.

        Returns
        -------
        Tensor
            Predicted fluorescence of shape ``(B, 1, D, H, W)``.
        """
        target = torch.randn_like(phase)
        sample_fn = self.transport_sampler.sample_ode(num_steps=num_steps)

        def fn(xt: Tensor, t: Tensor) -> Tensor:
            return self.net(xt, phase, t)

        with torch.no_grad():
            target = sample_fn(target, fn)[-1]

        return target

    def generate_non_overlapping(self, phase: Tensor, num_steps: int = 100) -> Tensor:
        """Generate virtual staining via non-overlapping tiling.

        Tiles the full input into non-overlapping patches matching
        ``net.input_spatial_size``, generates each patch independently,
        and assembles the results.

        Parameters
        ----------
        phase : Tensor
            Phase contrast input of shape ``(..., D, H, W)``.
        num_steps : int
            Number of ODE integration steps per patch.

        Returns
        -------
        Tensor
            Predicted fluorescence of shape ``(..., D, H, W)``.
        """
        spatial = tuple(phase.shape[-3:])
        patch_spatial = tuple(self.net.input_spatial_size)
        n_spatial = 3

        for i in range(n_spatial):
            if spatial[i] < patch_spatial[i]:
                raise ValueError(f"spatial dim {i} ({spatial[i]}) must be >= patch dim ({patch_spatial[i]})")

        out = torch.empty_like(phase)
        sample_fn = self.transport_sampler.sample_ode(num_steps=num_steps)

        start_lists: list[list[int]] = []
        for i in range(n_spatial):
            S, P = spatial[i], patch_spatial[i]
            starts = list(range(0, S - P + 1, P))
            if starts[-1] != S - P:
                starts.append(S - P)
            start_lists.append(starts)

        with torch.no_grad():
            for starts in itertools.product(*start_lists):
                slicer = [slice(None)] * phase.dim()
                for i, st in enumerate(starts):
                    slicer[-(n_spatial - i)] = slice(st, st + patch_spatial[i])
                phase_patch = phase[tuple(slicer)]
                xt = torch.randn_like(phase_patch)

                def fn(
                    xt_: Tensor,
                    t_: Tensor,
                    _p: Tensor = phase_patch,
                ) -> Tensor:
                    return self.net(xt_, _p, t_)

                out[tuple(slicer)] = sample_fn(xt, fn)[-1]

        return out

    def generate_sliding_window(
        self,
        phase: Tensor,
        num_steps: int = 100,
        overlap_size: int | tuple[int, ...] = 256,
    ) -> Tensor:
        """Generate virtual staining via overlapping sliding window.

        Uses overlapping patches for generation, anchoring already-computed
        values in the overlap region to guide subsequent patches.

        Parameters
        ----------
        phase : Tensor
            Phase contrast input of shape ``(..., D, H, W)``.
        num_steps : int
            Number of ODE integration steps per patch.
        overlap_size : int or tuple of int
            Overlap in each spatial dimension ``(od, oh, ow)``.
            A single int applies the same overlap to all three dimensions.

        Returns
        -------
        Tensor
            Predicted fluorescence of shape ``(..., D, H, W)``.
        """
        spatial = tuple(phase.shape[-3:])
        patch_spatial = tuple(self.net.input_spatial_size)
        n_spatial = 3

        if isinstance(overlap_size, int):
            overlap = (overlap_size,) * n_spatial
        else:
            overlap = tuple(overlap_size)
            if len(overlap) != n_spatial:
                raise ValueError("overlap_size must be int or a 3-tuple")

        for i in range(n_spatial):
            s_i, p_i, ov = spatial[i], patch_spatial[i], overlap[i]
            if s_i < p_i:
                raise ValueError(f"spatial dim {i} ({s_i}) must be >= patch dim ({p_i})")
            if not (0 <= ov < p_i):
                raise ValueError(f"overlap at dim {i} must satisfy 0 <= overlap < patch (got {ov} vs patch {p_i})")

        out = torch.full_like(phase, float("nan"))
        sample_fn = self.transport_sampler.sample_ode(num_steps=num_steps)

        start_lists: list[list[int]] = []
        for i in range(n_spatial):
            s_i, p_i, ov = spatial[i], patch_spatial[i], overlap[i]
            stride = p_i - ov
            last = s_i - p_i
            starts = [0]
            while True:
                nxt = starts[-1] + stride
                if nxt >= last:
                    break
                starts.append(nxt)
            if starts[-1] != last:
                starts.append(last)
            start_lists.append(starts)

        with torch.no_grad():
            for starts in itertools.product(*start_lists):
                slicer = [slice(None)] * phase.dim()
                for i, st in enumerate(starts):
                    slicer[-(n_spatial - i)] = slice(st, st + patch_spatial[i])

                phase_patch = phase[tuple(slicer)]
                out_patch = out[tuple(slicer)].clone()
                xt = torch.randn_like(phase_patch)
                known_mask = ~torch.isnan(out_patch)

                def fn(
                    xt_: Tensor,
                    t_: Tensor,
                    _p: Tensor = phase_patch,
                    _out: Tensor = out_patch,
                    _mask: Tensor = known_mask,
                ) -> Tensor:
                    v = self.net(xt_, _p, t_)
                    # Reshape t from (B,) to (B, 1, 1, 1, 1) for broadcasting.
                    t_exp = t_.reshape(t_.shape[0], *([1] * (xt_.dim() - 1)))
                    x0_ = xt_ - t_exp * v
                    v_out = _out - x0_
                    return torch.where(_mask, v_out, v)

                patch_out = sample_fn(xt, fn)[-1]
                # Preserve already-computed values in the overlap region.
                patch_out = torch.where(known_mask, out_patch, patch_out)
                out[tuple(slicer)] = patch_out

        return out
