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

from dynacell.tiling import window_starts
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

    def generate_trajectory(self, phase: Tensor, num_steps: int = 100) -> Tensor:
        """Generate virtual staining and return the full ODE trajectory.

        Parameters
        ----------
        phase : Tensor
            Phase contrast input of shape ``(B, 1, D, H, W)``.
        num_steps : int
            Number of ODE integration steps.

        Returns
        -------
        Tensor
            All intermediate ODE states of shape ``(num_steps, B, in_channels, D, H, W)``.
            Index 0 is pure Gaussian noise; index ``-1`` is the final prediction.
        """
        target = self._noise_like_target(phase)
        sample_fn = self.transport_sampler.sample_ode(num_steps=num_steps)

        def fn(xt: Tensor, t: Tensor) -> Tensor:
            return self.net(xt, phase, t)

        with torch.no_grad():
            return sample_fn(target, fn)  # (num_steps, B, C, D, H, W)

    def generate_sliding_window(self, phase: Tensor, num_steps: int = 100) -> Tensor:
        """Generate virtual staining via tiled sliding window (stride == patch size).

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
        start_lists = window_starts(spatial, patch_spatial, (0, 0, 0))

        in_ch = self.net.inconv.in_channels
        out_shape = (*phase.shape[:-4], in_ch, *phase.shape[-3:])
        out = torch.empty(out_shape, device=phase.device, dtype=phase.dtype)
        sample_fn = self.transport_sampler.sample_ode(num_steps=num_steps)

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

    def generate_sliding_window_trajectory(self, phase: Tensor, num_steps: int = 100) -> Tensor:
        """Generate the full ODE trajectory via tiled sliding window (stride == patch size).

        Like :meth:`generate_sliding_window`, but retains **every** ODE
        integration step instead of only the final one. Each non-overlapping
        patch is integrated independently with fresh Gaussian noise; step ``i``
        is at the same ODE time across all patches, so assembling by step index
        yields a spatially-complete volume per step.

        Parameters
        ----------
        phase : Tensor
            Phase contrast input of shape ``(B, 1, D, H, W)``.
        num_steps : int
            Number of ODE integration steps per patch.

        Returns
        -------
        Tensor
            All intermediate ODE states of shape
            ``(num_steps, B, in_channels, D, H, W)``. Index 0 is pure Gaussian
            noise; index ``-1`` is the final prediction.
        """
        spatial = tuple(phase.shape[-3:])
        patch_spatial = tuple(self.net.input_spatial_size)
        n_spatial = 3

        for i in range(n_spatial):
            if spatial[i] < patch_spatial[i]:
                raise ValueError(f"spatial dim {i} ({spatial[i]}) must be >= patch dim ({patch_spatial[i]})")

        in_ch = self.net.inconv.in_channels
        out_shape = (num_steps, *phase.shape[:-4], in_ch, *phase.shape[-3:])
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

                # (num_steps, B, C, pd, ph, pw); prepend step axis to the slicer.
                out[(slice(None), *slicer)] = sample_fn(xt, fn)

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
        if self.path_type != "Linear" or self.prediction != "velocity":
            raise NotImplementedError(
                "generate_iterative only supports Linear path with velocity prediction, "
                f"got path_type={self.path_type!r}, prediction={self.prediction!r}"
            )

        spatial = tuple(phase.shape[-3:])
        patch_spatial = tuple(self.net.input_spatial_size)
        n_spatial = 3
        start_lists = window_starts(spatial, patch_spatial, overlap_size)

        in_ch = self.net.inconv.in_channels
        out_shape = (*phase.shape[:-4], in_ch, *phase.shape[-3:])
        out = torch.full(out_shape, float("nan"), device=phase.device, dtype=phase.dtype)
        sample_fn = self.transport_sampler.sample_ode(num_steps=num_steps)

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

    def generate_iterative_trajectory(
        self,
        phase: Tensor,
        num_steps: int = 100,
        overlap_size: int | tuple[int, ...] = 256,
    ) -> Tensor:
        """Generate the full ODE trajectory via overlapping sliding window with velocity anchoring.

        Like :meth:`generate_iterative`, but retains **every** ODE integration
        step instead of only the final one. Patches are still processed
        sequentially; each patch's overlap region is anchored toward the
        **final** output of previously-completed patches (identical anchoring to
        :meth:`generate_iterative`), and the whole per-step trajectory of each
        patch is written into the output. Overlapping voxels are last-write-wins
        across patches, matching :meth:`generate_iterative`. Step ``i`` is at the
        same ODE time across all patches, so assembling by step index yields a
        spatially-complete volume per step.

        Parameters
        ----------
        phase : Tensor
            Phase contrast input of shape ``(B, 1, D, H, W)``.
        num_steps : int
            Number of ODE integration steps per patch.
        overlap_size : int or tuple of int
            Overlap in each spatial dimension ``(od, oh, ow)``.
            A single int applies the same overlap to all three dimensions.

        Returns
        -------
        Tensor
            All intermediate ODE states of shape
            ``(num_steps, B, in_channels, D, H, W)``. Index 0 is pure Gaussian
            noise; index ``-1`` is the final prediction (equal to
            :meth:`generate_iterative`).

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
                "generate_iterative_trajectory only supports Linear path with velocity prediction, "
                f"got path_type={self.path_type!r}, prediction={self.prediction!r}"
            )

        in_ch = self.net.inconv.in_channels
        out_shape = (*phase.shape[:-4], in_ch, *phase.shape[-3:])
        # `out` holds the FINAL values used for anchoring (as in generate_iterative);
        # `out_traj` accumulates every ODE step across the whole volume.
        out = torch.full(out_shape, float("nan"), device=phase.device, dtype=phase.dtype)
        traj_shape = (num_steps, *out_shape)
        out_traj = torch.empty(traj_shape, device=phase.device, dtype=phase.dtype)
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

                traj = sample_fn(xt, fn)  # (num_steps, B, C, pd, ph, pw)
                out_traj[(slice(None), *slicer)] = traj
                out[tuple(slicer)] = traj[-1]

        return out_traj

    def generate_iterative_denoise_trajectory(
        self,
        phase: Tensor,
        num_steps: int = 100,
        overlap_size: int | tuple[int, ...] = 256,
    ) -> Tensor:
        """Return the per-step denoised clean-target estimate ``x1`` of the iterative ODE.

        Runs the identical overlapping-window, velocity-anchored ODE integration
        as :meth:`generate_iterative` (the underlying prediction is unchanged),
        but at every returned ODE node ``i`` records the network's **denoised
        estimate of the clean target** rather than the raw ODE state ``xt``.

        For the Linear path with velocity prediction the flow satisfies
        ``xt = (1 - t) * x0 + t * x1`` with velocity ``v = x1 - x0``. Solving for
        the clean target gives

            ``x1 = xt + (1 - t) * v``

        which is the exact complement of the noise estimate ``x0 = xt - t * v``
        used for anchoring in :meth:`generate_iterative`. ``v`` here is the
        **raw** network velocity ``net(xt, phase, t)`` (not the anchored velocity
        used to drive the solver). At ``t = 0`` (step 1) this is the one-step
        denoise-from-noise estimate; at ``t = 1`` (last step) ``(1 - t) = 0`` so
        the estimate equals the final ODE state, i.e. the final prediction.

        Parameters
        ----------
        phase : Tensor
            Phase contrast input of shape ``(B, 1, D, H, W)``.
        num_steps : int
            Number of ODE integration steps per patch.
        overlap_size : int or tuple of int
            Overlap in each spatial dimension ``(od, oh, ow)``.
            A single int applies the same overlap to all three dimensions.

        Returns
        -------
        Tensor
            Per-step clean-target estimates of shape
            ``(num_steps, B, in_channels, D, H, W)``. Index 0 is the one-step
            denoise-from-noise estimate; index ``-1`` equals the final prediction.

        Raises
        ------
        NotImplementedError
            If ``path_type`` is not ``"Linear"`` or ``prediction`` is not
            ``"velocity"``, since the ``x1`` formula is path-specific.
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
                "generate_iterative_denoise_trajectory only supports Linear path with velocity "
                f"prediction, got path_type={self.path_type!r}, prediction={self.prediction!r}"
            )

        in_ch = self.net.inconv.in_channels
        out_shape = (*phase.shape[:-4], in_ch, *phase.shape[-3:])
        # `out` holds the FINAL values used for anchoring (as in generate_iterative);
        # `out_traj` accumulates the per-step x1 (clean-target) estimate.
        out = torch.full(out_shape, float("nan"), device=phase.device, dtype=phase.dtype)
        traj_shape = (num_steps, *out_shape)
        out_traj = torch.empty(traj_shape, device=phase.device, dtype=phase.dtype)
        sample_fn = self.transport_sampler.sample_ode(num_steps=num_steps)
        # ODE node times for the Linear+velocity config: linspace(0, 1, num_steps).
        t_grid = torch.linspace(0.0, 1.0, num_steps, device=phase.device, dtype=phase.dtype)

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
                batch_size = phase_patch.shape[0]

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

                # Raw ODE state trajectory (drives the prediction, kept for anchoring).
                traj = sample_fn(xt, fn)  # (num_steps, B, C, pd, ph, pw)

                # Per node, the clean-target estimate x1 = xt + (1 - t) * v with the RAW
                # network velocity at that node (re-evaluated; dopri5 nodes are dense-output
                # interpolants at t_grid, so this is the true velocity field at each node).
                for i in range(num_steps):
                    t_i = t_grid[i]
                    v_i = self.net(traj[i], phase_patch, t_i.expand(batch_size))
                    out_traj[(i, *slicer)] = traj[i] + (1.0 - t_i) * v_i

                out[tuple(slicer)] = traj[-1]

        return out_traj

    def denoise_sliding_window(
        self,
        phase: Tensor,
        overlap_size: int | tuple[int, ...] = 0,
    ) -> Tensor:
        """Estimate the conditional mean via overlapping tiled single-step Euler updates.

        Slides overlapping patches across the input.  Each patch is denoised
        independently with fresh Gaussian noise and the results are accumulated
        with a count tensor; overlapping regions are averaged, which reduces
        variance and approximates the conditional mean.

        Parameters
        ----------
        phase : Tensor
            Phase contrast input of shape ``(..., D, H, W)``.
        overlap_size : int or tuple of int
            Overlap in each spatial dimension ``(od, oh, ow)``.
            A single int applies the same overlap to all three dimensions.

        Returns
        -------
        Tensor
            Predicted fluorescence of shape ``(..., D, H, W)``.
        """
        if self.path_type != "Linear" or self.prediction != "velocity":
            raise NotImplementedError(
                "denoise_sliding_window only supports Linear path with velocity prediction, "
                f"got path_type={self.path_type!r}, prediction={self.prediction!r}"
            )

        spatial = tuple(phase.shape[-3:])
        patch_spatial = tuple(self.net.input_spatial_size)
        n_spatial = 3
        start_lists = window_starts(spatial, patch_spatial, overlap_size)

        in_ch = self.net.inconv.in_channels
        out_shape = (*phase.shape[:-4], in_ch, *phase.shape[-3:])
        prediction_sum = torch.zeros(out_shape, device=phase.device, dtype=phase.dtype)
        prediction_count = torch.zeros(out_shape, device=phase.device, dtype=phase.dtype)

        with torch.no_grad():
            for starts in itertools.product(*start_lists):
                slicer = [slice(None)] * phase.dim()
                for i, st in enumerate(starts):
                    slicer[-(n_spatial - i)] = slice(st, st + patch_spatial[i])
                phase_patch = phase[tuple(slicer)]
                xt = self._noise_like_target(phase_patch)
                t = torch.zeros(xt.shape[0], device=xt.device, dtype=xt.dtype)
                pred = self.net(xt, phase_patch, t)
                patch_out = pred + xt
                prediction_sum[tuple(slicer)] += patch_out
                prediction_count[tuple(slicer)] += 1

        if not torch.all(prediction_count > 0):
            raise RuntimeError("sliding window left uncovered voxels")
        return prediction_sum / prediction_count
