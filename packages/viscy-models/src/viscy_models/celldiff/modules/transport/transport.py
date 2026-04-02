"""Transport and Sampler classes for flow-matching.

Implements the core flow-matching training and inference logic:
loss computation, drift/score extraction, and ODE/SDE sampling.
"""

import enum
import math
from collections.abc import Callable

import torch
from torch import Tensor

from viscy_models.celldiff.modules.transport import path as _path
from viscy_models.celldiff.modules.transport.integrators import ODESolver, SDESolver
from viscy_models.celldiff.modules.transport.utils import mean_flat


class ModelType(enum.Enum):
    """Model prediction target."""

    NOISE = enum.auto()
    SCORE = enum.auto()
    VELOCITY = enum.auto()
    DENOISED = enum.auto()


class PathType(enum.Enum):
    """Flow path interpolation type."""

    LINEAR = enum.auto()
    GVP = enum.auto()
    VP = enum.auto()


class WeightType(enum.Enum):
    """Training loss weighting scheme."""

    NONE = enum.auto()
    VELOCITY = enum.auto()
    LIKELIHOOD = enum.auto()


class Transport:
    """Flow-matching transport for training and inference.

    Manages path sampling, loss computation, and drift/score extraction.

    Parameters
    ----------
    model_type : ModelType
        What the model predicts.
    path_type : PathType
        Which coupling plan to use.
    loss_type : WeightType
        Loss weighting scheme.
    train_eps : float
        Epsilon for training time interval stability.
    sample_eps : float
        Epsilon for sampling time interval stability.
    """

    def __init__(
        self,
        *,
        model_type: ModelType,
        path_type: PathType,
        loss_type: WeightType,
        train_eps: float,
        sample_eps: float,
    ) -> None:
        path_options = {
            PathType.LINEAR: _path.ICPlan,
            PathType.GVP: _path.GVPCPlan,
            PathType.VP: _path.VPCPlan,
        }
        self.loss_type = loss_type
        self.model_type = model_type
        self.path_sampler = path_options[path_type]()
        self.train_eps = train_eps
        self.sample_eps = sample_eps

    def prior_logp(self, z: Tensor) -> Tensor:
        """Compute log-probability under the standard normal prior.

        Parameters
        ----------
        z : Tensor
            Batched latent samples of shape ``(B, ...)``.

        Returns
        -------
        Tensor
            Log-probabilities of shape ``(B,)``.
        """
        shape = torch.tensor(z.size())
        n_dims = torch.prod(shape[1:])

        def _logp_single(x: Tensor) -> Tensor:
            return -n_dims / 2.0 * math.log(2 * math.pi) - torch.sum(x**2) / 2.0

        return torch.vmap(_logp_single)(z)

    def check_interval(
        self,
        train_eps: float,
        sample_eps: float,
        *,
        diffusion_form: str = "SBDM",
        sde: bool = False,
        reverse: bool = False,
        is_eval: bool = False,
        last_step_size: float = 0.0,
    ) -> tuple[float, float]:
        """Determine the integration time interval ``[t0, t1]``.

        Parameters
        ----------
        train_eps : float
            Training epsilon.
        sample_eps : float
            Sampling epsilon.
        diffusion_form : str
            SDE diffusion form.
        sde : bool
            Whether using SDE (vs ODE).
        reverse : bool
            Whether integrating in reverse.
        is_eval : bool
            Whether in evaluation mode.
        last_step_size : float
            Size of the final SDE step.

        Returns
        -------
        t0 : float
            Start time.
        t1 : float
            End time.
        """
        t0 = 0
        t1 = 1
        eps = train_eps if not is_eval else sample_eps

        if isinstance(self.path_sampler, _path.VPCPlan):
            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size

        elif isinstance(self.path_sampler, (_path.ICPlan, _path.GVPCPlan)) and (
            self.model_type != ModelType.VELOCITY or sde
        ):
            t0 = eps if (diffusion_form == "SBDM" and sde) or self.model_type != ModelType.VELOCITY else 0
            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size

        if reverse:
            t0, t1 = 1 - t0, 1 - t1

        return t0, t1

    def sample(self, x1: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Sample noise ``x0``, time ``t``, and return with data ``x1``.

        Parameters
        ----------
        x1 : Tensor
            Data batch of shape ``(B, ...)``.

        Returns
        -------
        t : Tensor
            Sampled times of shape ``(B,)``.
        x0 : Tensor
            Noise samples of same shape as ``x1``.
        x1 : Tensor
            Original data (unchanged).
        """
        x0 = torch.randn_like(x1)
        t0, t1 = self.check_interval(self.train_eps, self.sample_eps)
        t = torch.rand((x1.shape[0],)) * (t1 - t0) + t0
        t = t.to(x1)
        return t, x0, x1

    def training_losses(
        self,
        model_output: Tensor,
        x0: Tensor,
        x1: Tensor,
        xt: Tensor,
        ut: Tensor,
        t: Tensor,
    ) -> dict[str, Tensor]:
        """Compute flow-matching training loss.

        Parameters
        ----------
        model_output : Tensor
            Model prediction.
        x0 : Tensor
            Noise sample.
        x1 : Tensor
            Data sample.
        xt : Tensor
            Interpolated state.
        ut : Tensor
            Velocity target.
        t : Tensor
            Time vector.

        Returns
        -------
        dict[str, Tensor]
            Dictionary with ``"pred"`` and ``"loss"`` keys.
        """
        terms: dict[str, Tensor] = {}
        terms["pred"] = model_output
        if self.model_type == ModelType.VELOCITY:
            terms["loss"] = mean_flat((model_output - ut) ** 2)
        elif self.model_type == ModelType.DENOISED:
            terms["loss"] = mean_flat((model_output - x1) ** 2)
        else:
            _, drift_var = self.path_sampler.compute_drift(xt, t)
            sigma_t, _ = self.path_sampler.compute_sigma_t(_path.expand_t_like_x(t, xt))
            if self.loss_type == WeightType.VELOCITY:
                weight = (drift_var / sigma_t) ** 2
            elif self.loss_type == WeightType.LIKELIHOOD:
                weight = drift_var / (sigma_t**2)
            elif self.loss_type == WeightType.NONE:
                weight = 1
            else:
                raise NotImplementedError(f"Loss type {self.loss_type} not implemented")

            if self.model_type == ModelType.NOISE:
                terms["loss"] = mean_flat(weight * ((model_output - x0) ** 2))
            else:
                terms["loss"] = mean_flat(weight * ((model_output * sigma_t + x0) ** 2))

        return terms

    def get_drift(self) -> Callable:
        """Return the ODE drift function for the configured model type.

        Returns
        -------
        callable
            Drift function ``f(x, t, model, **kwargs) -> Tensor``.
        """

        def score_ode(x, t, model, **model_kwargs):
            drift_mean, drift_var = self.path_sampler.compute_drift(x, t)
            model_output = model(x, t, **model_kwargs)
            return -drift_mean + drift_var * model_output

        def noise_ode(x, t, model, **model_kwargs):
            drift_mean, drift_var = self.path_sampler.compute_drift(x, t)
            sigma_t, _ = self.path_sampler.compute_sigma_t(_path.expand_t_like_x(t, x))
            model_output = model(x, t, **model_kwargs)
            score = model_output / -sigma_t
            return -drift_mean + drift_var * score

        def velocity_ode(x, t, model, **model_kwargs):
            return model(x, t, **model_kwargs)

        def denoised_ode(x, t, model, **model_kwargs):
            drift_mean, drift_var = self.path_sampler.compute_drift(x, t)
            x1_hat = model(x, t, **model_kwargs)
            alpha_t, _ = self.path_sampler.compute_alpha_t(_path.expand_t_like_x(t, x))
            sigma_t, _ = self.path_sampler.compute_sigma_t(_path.expand_t_like_x(t, x))
            score = (alpha_t * x1_hat - x) / (sigma_t**2)
            return -drift_mean + drift_var * score

        drift_fns = {
            ModelType.NOISE: noise_ode,
            ModelType.SCORE: score_ode,
            ModelType.DENOISED: denoised_ode,
            ModelType.VELOCITY: velocity_ode,
        }
        drift_fn = drift_fns[self.model_type]

        def body_fn(x, t, model, **model_kwargs):
            model_output = drift_fn(x, t, model, **model_kwargs)
            if model_output.shape != x.shape:
                raise ValueError(f"ODE drift output shape {model_output.shape} does not match input shape {x.shape}")
            return model_output

        return body_fn

    def get_score(self) -> Callable:
        """Return the score function for the configured model type.

        Returns
        -------
        callable
            Score function ``s(x, t, model, **kwargs) -> Tensor``.
        """
        ps = self.path_sampler

        def _noise_score(x, t, model, **kwargs):
            sigma_t = ps.compute_sigma_t(_path.expand_t_like_x(t, x))[0]
            return model(x, t, **kwargs) / -sigma_t

        def _score_score(x, t, model, **kwargs):
            return model(x, t, **kwargs)

        def _velocity_score(x, t, model, **kwargs):
            return ps.get_score_from_velocity(model(x, t, **kwargs), x, t)

        def _denoised_score(x, t, model, **kwargs):
            return ps.get_score_from_denoised(model(x, t, **kwargs), x, t)

        score_fns = {
            ModelType.NOISE: _noise_score,
            ModelType.SCORE: _score_score,
            ModelType.VELOCITY: _velocity_score,
            ModelType.DENOISED: _denoised_score,
        }
        if self.model_type not in score_fns:
            raise NotImplementedError(f"Model type {self.model_type} not implemented")
        return score_fns[self.model_type]


class Sampler:
    """Sampler for ODE and SDE inference with a flow-matching transport.

    Parameters
    ----------
    transport : Transport
        Configured transport object.
    """

    def __init__(self, transport: Transport) -> None:
        self.transport = transport
        self.drift = self.transport.get_drift()
        self.score = self.transport.get_score()

    def _get_sde_diffusion_and_drift(
        self,
        *,
        diffusion_form: str = "SBDM",
        diffusion_norm: float = 1.0,
    ) -> tuple[Callable, Callable]:
        """Build SDE drift and diffusion functions."""

        def diffusion_fn(x, t):
            return self.transport.path_sampler.compute_diffusion(x, t, form=diffusion_form, norm=diffusion_norm)

        def sde_drift(x, t, model, **kwargs):
            return self.drift(x, t, model, **kwargs) + diffusion_fn(x, t) * self.score(x, t, model, **kwargs)

        return sde_drift, diffusion_fn

    def _get_last_step(
        self,
        sde_drift: Callable,
        *,
        last_step: str | None,
        last_step_size: float,
    ) -> Callable:
        """Build the final SDE step function."""
        if last_step is None:
            return lambda x, t, model, **model_kwargs: x
        elif last_step == "Mean":
            return lambda x, t, model, **model_kwargs: x + sde_drift(x, t, model, **model_kwargs) * last_step_size
        elif last_step == "Tweedie":
            alpha = self.transport.path_sampler.compute_alpha_t
            sigma = self.transport.path_sampler.compute_sigma_t
            return lambda x, t, model, **model_kwargs: (
                x / alpha(t)[0][0] + (sigma(t)[0][0] ** 2) / alpha(t)[0][0] * self.score(x, t, model, **model_kwargs)
            )
        elif last_step == "Euler":
            return lambda x, t, model, **model_kwargs: x + self.drift(x, t, model, **model_kwargs) * last_step_size
        else:
            raise NotImplementedError(f"Last step type {last_step!r} not implemented")

    def sample_sde(
        self,
        *,
        sampling_method: str = "Euler",
        diffusion_form: str = "SBDM",
        diffusion_norm: float = 1.0,
        last_step: str | None = "Mean",
        last_step_size: float = 0.04,
        num_steps: int = 250,
    ) -> Callable:
        """Return an SDE sampling function.

        Parameters
        ----------
        sampling_method : str
            Step type: ``"Euler"`` or ``"Heun"``.
        diffusion_form : str
            Diffusion coefficient form.
        diffusion_norm : float
            Diffusion magnitude.
        last_step : str or None
            Final step type: ``None``, ``"Mean"``, ``"Tweedie"``, ``"Euler"``.
        last_step_size : float
            Final step size.
        num_steps : int
            Number of integration steps.

        Returns
        -------
        callable
            Sampling function ``f(init, model, **kwargs) -> list[Tensor]``.
        """
        if last_step is None:
            last_step_size = 0.0

        sde_drift, sde_diffusion = self._get_sde_diffusion_and_drift(
            diffusion_form=diffusion_form,
            diffusion_norm=diffusion_norm,
        )

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            diffusion_form=diffusion_form,
            sde=True,
            is_eval=True,
            reverse=False,
            last_step_size=last_step_size,
        )

        _sde = SDESolver(
            sde_drift,
            sde_diffusion,
            t0=t0,
            t1=t1,
            num_steps=num_steps,
            sampler_type=sampling_method,
        )

        last_step_fn = self._get_last_step(sde_drift, last_step=last_step, last_step_size=last_step_size)

        def _sample(init: Tensor, model: Callable, **model_kwargs) -> list[Tensor]:
            x_final = _sde.sample(init, model, **model_kwargs)
            ts = torch.ones(init.size(0), device=init.device) * t1
            x = last_step_fn(x_final, ts, model, **model_kwargs)
            return [x]

        return _sample

    def sample_ode(
        self,
        *,
        sampling_method: str = "dopri5",
        num_steps: int = 50,
        atol: float = 1e-6,
        rtol: float = 1e-3,
        reverse: bool = False,
    ) -> Callable:
        """Return an ODE sampling function.

        Parameters
        ----------
        sampling_method : str
            ODE solver method (e.g. ``"dopri5"``, ``"euler"``).
        num_steps : int
            Number of output time points.
        atol : float
            Absolute error tolerance.
        rtol : float
            Relative error tolerance.
        reverse : bool
            Whether to integrate in reverse (data to noise).

        Returns
        -------
        callable
            Sampling function ``f(init, model, **kwargs) -> Tensor``.
        """
        if reverse:

            def drift(x, t, model, **kwargs):
                return self.drift(x, torch.ones_like(t) * (1 - t), model, **kwargs)

        else:
            drift = self.drift

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            sde=False,
            is_eval=True,
            reverse=reverse,
            last_step_size=0.0,
        )

        _ode = ODESolver(
            drift=drift,
            t0=t0,
            t1=t1,
            sampler_type=sampling_method,
            num_steps=num_steps,
            atol=atol,
            rtol=rtol,
        )

        return _ode.sample

    def sample_ode_likelihood(
        self,
        *,
        sampling_method: str = "dopri5",
        num_steps: int = 50,
        atol: float = 1e-6,
        rtol: float = 1e-3,
    ) -> Callable:
        """Return an ODE sampling function that also computes log-likelihood.

        Parameters
        ----------
        sampling_method : str
            ODE solver method.
        num_steps : int
            Number of output time points.
        atol : float
            Absolute error tolerance.
        rtol : float
            Relative error tolerance.

        Returns
        -------
        callable
            Function ``f(x, model, **kwargs) -> (logp, drift)``.
        """

        def _likelihood_drift(x, t, model, **model_kwargs):
            x, _ = x
            eps = torch.randint(2, x.size(), dtype=torch.float, device=x.device) * 2 - 1
            t = torch.ones_like(t) * (1 - t)
            with torch.enable_grad():
                x.requires_grad = True
                grad = torch.autograd.grad(torch.sum(self.drift(x, t, model, **model_kwargs) * eps), x)[0]
                logp_grad = torch.sum(grad * eps, dim=tuple(range(1, len(x.size()))))
                drift = self.drift(x, t, model, **model_kwargs)
            return (-drift, logp_grad)

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            sde=False,
            is_eval=True,
            reverse=False,
            last_step_size=0.0,
        )

        _ode = ODESolver(
            drift=_likelihood_drift,
            t0=t0,
            t1=t1,
            sampler_type=sampling_method,
            num_steps=num_steps,
            atol=atol,
            rtol=rtol,
        )

        def _sample_fn(x: Tensor, model: Callable, **model_kwargs) -> tuple[Tensor, Tensor]:
            init_logp = torch.zeros(x.size(0)).to(x)
            input_state = (x, init_logp)
            drift, delta_logp = _ode.sample(input_state, model, **model_kwargs)
            drift, delta_logp = drift[-1], delta_logp[-1]
            prior_logp = self.transport.prior_logp(drift)
            logp = prior_logp - delta_logp
            return logp, drift

        return _sample_fn
