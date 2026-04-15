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
        self.path_type = path_type
        self.prediction = prediction
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

    def _noise_like_target(self, phase: Tensor) -> Tensor:
        """Create Gaussian noise with the network's output channel count.

        Parameters
        ----------
        phase : Tensor
            Phase conditioning tensor whose batch and spatial dims are reused.

        Returns
        -------
        Tensor
            Noise of shape ``(B, in_channels, D, H, W)``.
        """
        b, _c, *spatial = phase.shape
        in_ch = self.net.inconv.in_channels
        return torch.randn(b, in_ch, *spatial, device=phase.device, dtype=phase.dtype)

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
            Predicted fluorescence of shape ``(B, in_channels, D, H, W)``.
        """
        target = self._noise_like_target(phase)
        sample_fn = self.transport_sampler.sample_ode(num_steps=num_steps)

        def fn(xt: Tensor, t: Tensor) -> Tensor:
            return self.net(xt, phase, t)

        with torch.no_grad():
            target = sample_fn(target, fn)[-1]

        return target

    def generate_sliding_window(self, phase: Tensor, num_steps: int = 100) -> Tensor:
        """Generate virtual staining via non-overlapping tiling.

        Partitions the input into non-overlapping patches of size
        ``net.input_spatial_size``.  Each patch is generated independently
        with fresh Gaussian noise and the results are written back into the
        corresponding region of the output tensor.  The last tile along each
        axis is snapped to the image edge, so it may overlap its predecessor
        when the image size is not an exact multiple of the patch size.

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

        in_ch = self.net.inconv.in_channels
        out_shape = (*phase.shape[:-4], in_ch, *phase.shape[-3:])
        out = torch.empty(out_shape, device=phase.device, dtype=phase.dtype)
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
                xt = self._noise_like_target(phase_patch)

                def fn(
                    xt_: Tensor,
                    t_: Tensor,
                    _p: Tensor = phase_patch,
                ) -> Tensor:
                    return self.net(xt_, _p, t_)

                out[tuple(slicer)] = sample_fn(xt, fn)[-1]

        return out

    def generate_iterative(
        self,
        phase: Tensor,
        num_steps: int = 100,
        overlap_size: int | tuple[int, ...] = 256,
    ) -> Tensor:
        """Generate virtual staining via overlapping sliding window with velocity anchoring.

        Slides overlapping patches across the input.  For each patch the
        overlap region (already generated by an earlier patch) is used to
        steer the ODE trajectory toward the previously computed output values
        rather than letting the solver integrate freely.

        **Anchoring mechanism** (requires Linear path + velocity prediction):
        At every ODE step the network predicts a velocity ``v``.  Under the
        Linear flow the starting point is ``x0 = xt - t * v``.  For pixels in
        the overlap region we override the velocity with
        ``v_anchored = out_known - x0``, which is the exact velocity that
        would integrate ``x0`` to the already-computed target ``out_known``.
        Outside the overlap the free velocity ``v`` is used unchanged.

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

        Raises
        ------
        NotImplementedError
            If ``path_type`` is not ``"Linear"`` or ``prediction`` is not
            ``"velocity"``, since the anchoring formula is path-specific.
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

        if self.path_type != "Linear" or self.prediction != "velocity":
            raise NotImplementedError(
                "generate_sliding_window only supports Linear path with velocity prediction, "
                f"got path_type={self.path_type!r}, prediction={self.prediction!r}"
            )

        in_ch = self.net.inconv.in_channels
        out_shape = (*phase.shape[:-4], in_ch, *phase.shape[-3:])
        out = torch.full(out_shape, float("nan"), device=phase.device, dtype=phase.dtype)
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
                xt = self._noise_like_target(phase_patch)
                known_mask = ~torch.isnan(out_patch)

                def fn(
                    xt_: Tensor,
                    t_: Tensor,
                    _p: Tensor = phase_patch,
                    _out: Tensor = out_patch,
                    _mask: Tensor = known_mask,
                ) -> Tensor:
                    v = self.net(xt_, _p, t_)
                    # Infer x0 from the Linear-path formula: x0 = xt - t*v.
                    t_exp = t_.reshape(t_.shape[0], *([1] * (xt_.dim() - 1)))
                    x0_ = xt_ - t_exp * v
                    # Velocity that integrates x0 exactly to the known target: v = x1 - x0.
                    v_out = _out - x0_
                    # Use the anchored velocity in the overlap region, free velocity elsewhere.
                    return torch.where(_mask, v_out, v)

                patch_out = sample_fn(xt, fn)[-1]
                out[tuple(slicer)] = patch_out

        return out
