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

from dynacell.mask_conditioning import binarize_mask, encode_mask
from dynacell.tiling import window_starts
from viscy_models.celldiff import CELLDiffNet
from viscy_models.celldiff.modules.transport import Sampler, create_transport
from viscy_models.celldiff.modules.transport.utils import expand_t_like_x
from viscy_utils.losses import SegAuxDice


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
    seg_aux : SegAuxDice or None
        Auxiliary segmentation loss on the one-step data estimate
        ``x1_hat = x_t + (1 - t) * v_hat`` (linear path, t=1 is data). Only
        consulted when ``forward`` receives ``fg_mask``; requires
        ``path_type="Linear"`` and ``prediction="velocity"``, the only case in
        which that estimate is exact.
    seg_aux_t0 : float
        Gate for ``seg_aux``: a sample contributes only when
        ``t >= 1 - seg_aux_t0``, i.e. the least-noisy ``seg_aux_t0`` fraction
        of the time range. ``1.0`` is ungated. Also gates the mask Dice of
        :meth:`joint_mask_losses`.
    joint_mask : bool
        C-joint: the flow generates ``[image, mask]``, the mask channels
        being ``fg_mask`` encoded to ``{-1, 1}``; train with
        :meth:`joint_mask_losses`. ``net.in_channels`` must then be twice the
        target channel count, and every ``generate*`` method returns both
        halves. Requires ``path_type="Linear"`` and ``prediction="velocity"``
        (the mask Dice scores ``x1_hat``).
    """

    def __init__(
        self,
        net: CELLDiffNet,
        path_type: str = "Linear",
        prediction: str = "velocity",
        loss_weight: str | None = None,
        train_eps: float | None = None,
        sample_eps: float | None = None,
        seg_aux: SegAuxDice | None = None,
        seg_aux_t0: float = 0.7,
        joint_mask: bool = False,
    ) -> None:
        super().__init__()
        if (seg_aux is not None or joint_mask) and (path_type != "Linear" or prediction != "velocity"):
            raise ValueError(
                "seg_aux and joint_mask use x1_hat = x_t + (1 - t) * v_hat, which holds only for "
                f"path_type='Linear' and prediction='velocity'; got {path_type!r}, {prediction!r}."
            )
        if not 0.0 < seg_aux_t0 <= 1.0:
            raise ValueError(f"seg_aux_t0 must be in (0, 1], got {seg_aux_t0}")
        self.net = net
        self.path_type = path_type
        self.prediction = prediction
        self.transport = create_transport(path_type, prediction, loss_weight, train_eps, sample_eps)
        self.transport_sampler = Sampler(self.transport)
        self.seg_aux = seg_aux
        self.seg_aux_t0 = seg_aux_t0
        self.joint_mask = joint_mask
        if joint_mask and net.inconv.in_channels % 2:
            raise ValueError(f"joint_mask needs net in_channels = 2 x target channels, got {net.inconv.in_channels}")

    def forward(
        self, phase: Tensor, target: Tensor, fg_mask: Tensor | None = None
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """Compute flow-matching training loss.

        Parameters
        ----------
        phase : Tensor
            Phase contrast input of shape ``(B, 1, D, H, W)``.
        target : Tensor
            Fluorescence target of shape ``(B, C, D, H, W)``.
        fg_mask : Tensor or None
            Foreground mask shaped like ``target``. When given, ``seg_aux``
            must be set and its gated Dice is returned alongside the loss.

        Returns
        -------
        Tensor or tuple of (Tensor, dict of str to Tensor)
            Scalar velocity loss; with ``fg_mask``, also
            ``{"dice", "n_valid", "n_gated"}``. ``dice`` is the batch mean of
            ``1[t >= 1 - t0] * Dice(x1_hat)`` over patches with a usable mask
            (unweighted; the caller applies its weight), ``n_valid`` the
            patches that contributed, ``n_gated`` the samples inside the gate.
            The velocity loss is the same quantity as without ``fg_mask``.
        """
        t, x0, x1 = self.transport.sample(target)
        t, xt, ut = self.transport.path_sampler.plan(t, x0, x1)
        pred = self.net(xt, phase, t)
        loss_dict = self.transport.training_losses(pred, x0, x1, xt, ut, t)
        loss = loss_dict["loss"].mean()
        if fg_mask is None:
            return loss
        if self.seg_aux is None:
            raise ValueError("fg_mask was passed but seg_aux is not configured.")
        x1_hat = xt + expand_t_like_x(1 - t, xt) * pred
        dice, valid = self.seg_aux.per_channel(x1_hat, target, fg_mask)
        gate = t >= 1 - self.seg_aux_t0
        contrib = valid & gate.unsqueeze(1)
        # Mean over valid patches of g(t) * Dice: gated-out samples add 0 but
        # still count, so the term is an unbiased estimate of E[g(t) * Dice].
        gated_dice = (dice * contrib).sum() / valid.sum().clamp(min=1)
        return loss, {"dice": gated_dice, "n_valid": contrib.sum().float(), "n_gated": gate.sum().float()}

    def joint_mask_losses(self, phase: Tensor, target: Tensor, fg_mask: Tensor) -> dict[str, Tensor]:
        """Compute the C-joint losses on the flow over ``[target, encoded fg_mask]``.

        One noise draw and one ``t`` per sample cover both halves. The mask
        Dice is a squared-denominator soft Dice of
        ``p = clamp((x1_hat_mask + 1) / 2, 0, 1)`` against the binary mask,
        gated to ``t >= 1 - seg_aux_t0``; patches with an empty mask are
        excluded from the mean (gated-out samples count as 0).

        Parameters
        ----------
        phase : Tensor
            Conditioning of shape ``(B, cond_channels, D, H, W)``.
        target : Tensor
            Fluorescence target of shape ``(B, C, D, H, W)``.
        fg_mask : Tensor
            Foreground mask shaped like ``target``; binarized at 0.5.

        Returns
        -------
        dict of str to Tensor
            ``velocity_image`` (the baseline's velocity loss, on the image
            half), ``velocity_mask``, ``dice`` (unweighted), ``n_valid`` and
            ``n_gated``.
        """
        if not self.joint_mask:
            raise ValueError("joint_mask_losses needs joint_mask=True.")
        if fg_mask.shape != target.shape:
            raise ValueError(f"fg_mask {tuple(fg_mask.shape)} must match target {tuple(target.shape)}")
        n = target.shape[1]
        if 2 * n != self.net.inconv.in_channels:
            raise ValueError(f"target has {n} channel(s) but the net generates {self.net.inconv.in_channels}")
        mask = binarize_mask(fg_mask).to(target.dtype)
        t, x0, x1 = self.transport.sample(torch.cat([target, encode_mask(mask)], dim=1))
        t, xt, ut = self.transport.path_sampler.plan(t, x0, x1)
        pred = self.net(xt, phase, t)
        image, masked = slice(None, n), slice(n, None)
        velocity = {
            name: self.transport.training_losses(pred[:, sl], x0[:, sl], x1[:, sl], xt[:, sl], ut[:, sl], t)[
                "loss"
            ].mean()
            for name, sl in (("velocity_image", image), ("velocity_mask", masked))
        }
        x1_hat = xt[:, masked] + expand_t_like_x(1 - t, xt) * pred[:, masked]
        with torch.autocast(device_type=x1_hat.device.type, enabled=False):
            p = ((x1_hat.float() + 1.0) / 2.0).clamp(0.0, 1.0).flatten(2)
            m = mask.float().flatten(2)
            dice = 1.0 - 2.0 * (p * m).sum(-1) / ((p * p).sum(-1) + m.sum(-1) + 1e-6)
        valid = m.sum(-1) > 0
        gate = t >= 1 - self.seg_aux_t0
        contrib = valid & gate.unsqueeze(1)
        gated_dice = (dice * contrib).sum() / valid.sum().clamp(min=1)
        return {
            **velocity,
            "dice": gated_dice,
            "n_valid": contrib.sum().float(),
            "n_gated": gate.sum().float(),
        }

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
