"""Flow-matching 3D U-Net with Vision Transformer bottleneck for virtual staining.

Diffusion (flow-matching) model mapping label-free phase contrast to fluorescence
virtual staining.  Architecture: CNN encoder with skip connections,
timestep-conditioned transformer bottleneck, CNN decoder with skip connections.

Reference: cell_diff_3d_vs_fm in the cell_diff project.
"""

import itertools
import math

import torch
import torch.nn as nn

from viscy_models.cell_diff.modules import TimestepEmbedder
from viscy_models.cell_diff.modules.patch_embed_3d import PatchEmbed3D
from viscy_models.cell_diff.modules.positional_embedding import get_3d_sincos_pos_embed
from viscy_models.cell_diff.modules.simple_diffusion import ResnetBlock
from viscy_models.cell_diff.modules.transformer import FinalLayer, TransformerBlock, unpatchify
from viscy_models.cell_diff.modules.transport import Sampler, create_transport


class CellDiffNet(nn.Module):
    """3D U-Net with ViT bottleneck for flow-matching virtual staining.

    Takes a noisy target, a phase contrast conditioning image, and a diffusion
    timestep, and predicts the velocity field for flow-matching training.

    Parameters
    ----------
    input_spatial_size : list[int]
        Expected input spatial size ``[D, H, W]``.  Used for positional
        embedding computation and forward-pass assertions.
    in_channels : int
        Number of input/output channels.
    dims : list[int]
        Channel widths at each encoder level, length ``L``.
        Must satisfy ``len(dims) == len(num_res_block) + 1``.
    num_res_block : list[int]
        Number of residual blocks at each encoder/decoder level, length ``L-1``.
    hidden_size : int
        Transformer hidden dimension.
    num_heads : int
        Number of self-attention heads.
    dim_head : int
        Dimension per attention head.
    dropout : float
        Attention dropout rate.
    final_dropout : float
        Feed-forward output dropout rate.
    num_hidden_layers : int
        Number of transformer blocks in the bottleneck.
    patch_size : int
        Cubic patch size for the 3D patch embedding.
    """

    def __init__(
        self,
        input_spatial_size: list[int] = [8, 512, 512],
        in_channels: int = 1,
        dims: list[int] = [32, 64, 128],
        num_res_block: list[int] = [2, 2],
        hidden_size: int = 512,
        num_heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        final_dropout: float = 0.0,
        num_hidden_layers: int = 2,
        patch_size: int = 4,
    ):
        super().__init__()

        assert len(dims) == len(num_res_block) + 1, (
            f"len(dims)={len(dims)} must equal len(num_res_block)+1={len(num_res_block) + 1}"
        )

        self.input_spatial_size = input_spatial_size
        self.num_res_block = num_res_block
        self._dims = dims
        self._patch_size = patch_size

        # ── Input projections ───────────────────────────────────────────────
        self.inconv = nn.Conv3d(in_channels, dims[0], kernel_size=3, stride=1, padding=1)
        self.cond_inconv = nn.Conv3d(1, dims[0], kernel_size=3, stride=1, padding=1)

        # ── Timestep embedding ──────────────────────────────────────────────
        self.t_embedding = TimestepEmbedder(hidden_size=hidden_size)

        # ── Encoder (downsampling) ───────────────────────────────────────────
        downs: dict[str, nn.Module] = {}
        for i_level in range(len(num_res_block)):
            for i_block in range(num_res_block[i_level]):
                downs[f"{i_level}{i_block}"] = ResnetBlock(
                    dims[i_level], dims[i_level], time_emb_dim=hidden_size
                )
            downs[f"down{i_level}"] = nn.Conv3d(
                dims[i_level], dims[i_level + 1],
                kernel_size=3, stride=(1, 2, 2), padding=1,
            )
        self.downs = nn.ModuleDict(downs)

        # ── Transformer bottleneck ──────────────────────────────────────────
        self.img_embedding = PatchEmbed3D(
            patch_size=patch_size,
            in_chans=dims[-1],
            embed_dim=hidden_size,
            bias=True,
        )

        n_down = len(num_res_block)
        latent_size = input_spatial_size[:1] + [s // (2**n_down) for s in input_spatial_size[1:]]
        self.latent_grid_size = [s // patch_size for s in latent_size]
        assert math.prod(self.latent_grid_size) > 0, (
            f"latent_grid_size {self.latent_grid_size} contains a zero; "
            "check that input_spatial_size is divisible by 2^n_down * patch_size"
        )

        img_pos_embed = (
            torch.from_numpy(get_3d_sincos_pos_embed(hidden_size, self.latent_grid_size))
            .float()
            .unsqueeze(0)
        )
        self.img_pos_embed = nn.Parameter(img_pos_embed, requires_grad=False)

        self.mids = nn.ModuleList([
            TransformerBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                dim_head=dim_head,
                dropout=dropout,
                final_dropout=final_dropout,
                time_embed_dim=hidden_size,
            )
            for _ in range(num_hidden_layers)
        ])

        self.img_proj_out = FinalLayer(
            hidden_size=hidden_size,
            patch_size=patch_size,
            out_channels=dims[-1],
            time_embed_dim=hidden_size,
        )

        # ── Decoder (upsampling) ────────────────────────────────────────────
        ups: dict[str, nn.Module] = {}
        for i_level in reversed(range(len(num_res_block))):
            ups[f"up{i_level}"] = nn.ConvTranspose3d(
                dims[i_level + 1],
                dims[i_level],
                kernel_size=(1, 3, 3),
                stride=(1, 2, 2),
                padding=(0, 1, 1),
                output_padding=(0, 1, 1),
            )
            for i_block in range(num_res_block[i_level]):
                ups[f"{i_level}{i_block}"] = ResnetBlock(
                    dims[i_level] * 2, dims[i_level], time_emb_dim=hidden_size
                )
        self.ups = nn.ModuleDict(ups)

        # ── Output projection ───────────────────────────────────────────────
        self.outconv = nn.Conv3d(dims[0], in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict velocity field for flow-matching.

        Parameters
        ----------
        x : torch.Tensor
            Noisy target volume of shape ``(B, in_channels, D, H, W)``.
        cond : torch.Tensor
            Phase contrast conditioning of shape ``(B, 1, D, H, W)``.
        t : torch.Tensor
            Diffusion timesteps of shape ``(B,)``.

        Returns
        -------
        torch.Tensor
            Predicted velocity field of shape ``(B, in_channels, D, H, W)``.
        """
        assert list(x.shape[2:]) == self.input_spatial_size, (
            f"x spatial size {list(x.shape[2:])} does not match expected {self.input_spatial_size}"
        )
        assert list(cond.shape[2:]) == self.input_spatial_size, (
            f"cond spatial size {list(cond.shape[2:])} does not match expected {self.input_spatial_size}"
        )

        time_embeds = self.t_embedding(t)
        h = self.inconv(x) + self.cond_inconv(cond)

        # ── Encode ──────────────────────────────────────────────────────────
        skips: list[torch.Tensor] = []
        for i_level in range(len(self.num_res_block)):
            for i_block in range(self.num_res_block[i_level]):
                h = self.downs[f"{i_level}{i_block}"](h, time_embeds)
                skips.append(h)
            h = self.downs[f"down{i_level}"](h)

        # ── Transformer bottleneck ──────────────────────────────────────────
        h_embeds = self.img_embedding(h) + self.img_pos_embed
        for block in self.mids:
            h_embeds = block(h_embeds, time_embeds)
        h = self.img_proj_out(h_embeds, time_embeds)
        h = unpatchify(h, self._dims[-1], self.latent_grid_size, self._patch_size)

        # ── Decode ──────────────────────────────────────────────────────────
        for i_level in reversed(range(len(self.num_res_block))):
            h = self.ups[f"up{i_level}"](h)
            for i_block in range(self.num_res_block[i_level]):
                h = torch.cat((h, skips.pop()), dim=1)
                h = self.ups[f"{i_level}{i_block}"](h, time_embeds)

        return self.outconv(h)


class CellDiff3DVS(nn.Module):
    """Flow-matching virtual staining model.

    Wraps a :class:`CellDiffNet` backbone with a flow-matching transport to
    provide training loss computation and inference (generation) methods.

    Parameters
    ----------
    net : CellDiffNet
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
        net: CellDiffNet,
        path_type: str = "Linear",
        prediction: str = "velocity",
        loss_weight: str | None = None,
        train_eps: float | None = None,
        sample_eps: float | None = None,
    ):
        super().__init__()
        self.net = net
        self.transport = create_transport(path_type, prediction, loss_weight, train_eps, sample_eps)
        self.transport_sampler = Sampler(self.transport)

    def forward(self, phase: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute flow-matching training loss.

        Parameters
        ----------
        phase : torch.Tensor
            Phase contrast input of shape ``(B, 1, D, H, W)``.
        target : torch.Tensor
            Fluorescence target of shape ``(B, C, D, H, W)``.

        Returns
        -------
        torch.Tensor
            Scalar training loss.
        """
        t, x0, x1 = self.transport.sample(target)
        t, xt, ut = self.transport.path_sampler.plan(t, x0, x1)
        pred = self.net(xt, phase, t)
        loss_dict = self.transport.training_losses(pred, x0, x1, xt, ut, t)
        return loss_dict["loss"].mean()

    def generate(self, phase: torch.Tensor, num_steps: int = 100) -> torch.Tensor:
        """Generate virtual staining via ODE sampling.

        Parameters
        ----------
        phase : torch.Tensor
            Phase contrast input of shape ``(B, 1, D, H, W)``.
        num_steps : int
            Number of ODE integration steps.

        Returns
        -------
        torch.Tensor
            Predicted fluorescence of shape ``(B, 1, D, H, W)``.
        """
        target = torch.randn_like(phase)
        sample_fn = self.transport_sampler.sample_ode(num_steps=num_steps)

        def fn(xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return self.net(xt, phase, t)

        with torch.no_grad():
            target = sample_fn(target, fn)[-1]

        return target

    def generate_non_overlapping(self, phase: torch.Tensor, num_steps: int = 100) -> torch.Tensor:
        """Generate virtual staining via non-overlapping tiling.

        Tiles the full input into non-overlapping patches matching
        ``net.input_spatial_size``, generates each patch independently,
        and assembles the results.

        Parameters
        ----------
        phase : torch.Tensor
            Phase contrast input of shape ``(..., D, H, W)``.
        num_steps : int
            Number of ODE integration steps per patch.

        Returns
        -------
        torch.Tensor
            Predicted fluorescence of shape ``(..., D, H, W)``.
        """
        spatial = tuple(phase.shape[-3:])
        patch_spatial = tuple(self.net.input_spatial_size)
        n_spatial = 3

        for i in range(n_spatial):
            assert spatial[i] >= patch_spatial[i], (
                f"spatial dim {i} ({spatial[i]}) must be >= patch dim ({patch_spatial[i]})"
            )

        out = torch.empty_like(phase)
        sample_fn = self.transport_sampler.sample_ode(num_steps=num_steps)

        start_lists = []
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

                def fn(xt_: torch.Tensor, t_: torch.Tensor, _p: torch.Tensor = phase_patch) -> torch.Tensor:
                    return self.net(xt_, _p, t_)

                out[tuple(slicer)] = sample_fn(xt, fn)[-1]

        return out

    def generate_sliding_window(
        self,
        phase: torch.Tensor,
        num_steps: int = 100,
        overlap_size: int | tuple = 256,
    ) -> torch.Tensor:
        """Generate virtual staining via overlapping sliding window.

        Uses overlapping patches for generation, anchoring already-computed
        values in the overlap region to guide subsequent patches.

        Parameters
        ----------
        phase : torch.Tensor
            Phase contrast input of shape ``(..., D, H, W)``.
        num_steps : int
            Number of ODE integration steps per patch.
        overlap_size : int or tuple of int
            Overlap in each spatial dimension ``(od, oh, ow)``.
            A single int applies the same overlap to all three dimensions.

        Returns
        -------
        torch.Tensor
            Predicted fluorescence of shape ``(..., D, H, W)``.
        """
        spatial = tuple(phase.shape[-3:])
        patch_spatial = tuple(self.net.input_spatial_size)
        n_spatial = 3

        if isinstance(overlap_size, int):
            overlap = (overlap_size,) * n_spatial
        else:
            overlap = tuple(overlap_size)
            assert len(overlap) == n_spatial, "overlap_size must be int or a 3-tuple"

        for i in range(n_spatial):
            S, P, O = spatial[i], patch_spatial[i], overlap[i]
            assert S >= P, f"spatial dim {i} ({S}) must be >= patch dim ({P})"
            assert 0 <= O < P, (
                f"overlap at dim {i} must satisfy 0 <= overlap < patch (got {O} vs patch {P})"
            )

        out = torch.full_like(phase, float("nan"))
        sample_fn = self.transport_sampler.sample_ode(num_steps=num_steps)

        start_lists = []
        for i in range(n_spatial):
            S, P, O = spatial[i], patch_spatial[i], overlap[i]
            stride = P - O
            last = S - P
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
                    xt_: torch.Tensor,
                    t_: torch.Tensor,
                    _p: torch.Tensor = phase_patch,
                    _out: torch.Tensor = out_patch,
                    _mask: torch.Tensor = known_mask,
                ) -> torch.Tensor:
                    v = self.net(xt_, _p, t_)
                    # reshape t_ from (B,) to (B, 1, 1, 1, 1) for broadcasting
                    t_exp = t_.reshape(t_.shape[0], *([1] * (xt_.dim() - 1)))
                    x0_ = xt_ - t_exp * v
                    v_out = _out - x0_
                    return torch.where(_mask, v_out, v)

                patch_out = sample_fn(xt, fn)[-1]
                # preserve already-computed values in the overlap region
                patch_out = torch.where(known_mask, out_patch, patch_out)
                out[tuple(slicer)] = patch_out

        return out
