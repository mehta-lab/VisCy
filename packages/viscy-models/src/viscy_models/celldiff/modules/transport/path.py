"""Coupling plan classes for flow-matching transport paths.

Provides linear, variance-preserving, and geometric vector path
interpolation plans used by :class:`Transport` during training and sampling.
"""

import math

import torch
from torch import Tensor

from viscy_models.celldiff.modules.transport.utils import expand_t_like_x


class ICPlan:
    """Linear interpolant coupling plan.

    Implements the linear path ``x_t = t * x_1 + (1-t) * x_0``.

    Parameters
    ----------
    sigma : float
        Noise scale (unused in current implementation, reserved for extensions).
    """

    def __init__(self, sigma: float = 0.0) -> None:
        self.sigma = sigma

    def compute_alpha_t(self, t: Tensor) -> tuple[Tensor, int]:
        """Compute the data coefficient along the path.

        Parameters
        ----------
        t : Tensor
            Time values, broadcastable with data.

        Returns
        -------
        alpha_t : Tensor
            Data coefficient ``t``.
        d_alpha_t : int
            Derivative of alpha w.r.t. time (constant ``1``).
        """
        return t, 1

    def compute_sigma_t(self, t: Tensor) -> tuple[Tensor, int]:
        """Compute the noise coefficient along the path.

        Parameters
        ----------
        t : Tensor
            Time values, broadcastable with data.

        Returns
        -------
        sigma_t : Tensor
            Noise coefficient ``1 - t``.
        d_sigma_t : int
            Derivative of sigma w.r.t. time (constant ``-1``).
        """
        return 1 - t, -1

    def compute_d_alpha_alpha_ratio_t(self, t: Tensor) -> Tensor:
        """Compute the ratio ``d_alpha_t / alpha_t = 1 / t``."""
        return 1 / t

    def compute_drift(self, x: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        """Compute the SDE drift in the score parametrization.

        Parameters
        ----------
        x : Tensor
            State tensor of shape ``(B, ...)``.
        t : Tensor
            Time vector of shape ``(B,)``.

        Returns
        -------
        neg_drift : Tensor
            Negative drift term.
        diffusion : Tensor
            Diffusion coefficient.
        """
        t = expand_t_like_x(t, x)
        alpha_ratio = self.compute_d_alpha_alpha_ratio_t(t)
        sigma_t, d_sigma_t = self.compute_sigma_t(t)
        drift = alpha_ratio * x
        diffusion = alpha_ratio * (sigma_t**2) - sigma_t * d_sigma_t
        return -drift, diffusion

    def compute_diffusion(self, x: Tensor, t: Tensor, form: str = "constant", norm: float = 1.0) -> Tensor | float:
        """Compute the diffusion term of the SDE.

        Parameters
        ----------
        x : Tensor
            State tensor of shape ``(B, ...)``.
        t : Tensor
            Time vector of shape ``(B,)``.
        form : str
            Diffusion form. One of ``"constant"``, ``"SBDM"``, ``"sigma"``,
            ``"linear"``, ``"decreasing"``, ``"increasing-decreasing"``.
        norm : float
            Magnitude scaling factor.

        Returns
        -------
        Tensor or float
            Diffusion coefficient.
        """
        t = expand_t_like_x(t, x)
        if form == "constant":
            return norm
        elif form == "SBDM":
            return norm * self.compute_drift(x, t)[1]
        elif form == "sigma":
            return norm * self.compute_sigma_t(t)[0]
        elif form == "linear":
            return norm * (1 - t)
        elif form == "decreasing":
            return 0.25 * (norm * torch.cos(math.pi * t) + 1) ** 2
        elif form == "increasing-decreasing":
            return norm * torch.sin(math.pi * t) ** 2
        else:
            raise NotImplementedError(f"Diffusion form {form!r} not implemented")

    def get_score_from_velocity(self, velocity: Tensor, x: Tensor, t: Tensor) -> Tensor:
        """Transform velocity prediction to score.

        Parameters
        ----------
        velocity : Tensor
            Velocity model output.
        x : Tensor
            State ``x_t``.
        t : Tensor
            Time vector of shape ``(B,)``.

        Returns
        -------
        Tensor
            Score estimate.
        """
        t = expand_t_like_x(t, x)
        alpha_t, d_alpha_t = self.compute_alpha_t(t)
        sigma_t, d_sigma_t = self.compute_sigma_t(t)
        mean = x
        reverse_alpha_ratio = alpha_t / d_alpha_t
        var = sigma_t**2 - reverse_alpha_ratio * d_sigma_t * sigma_t
        return (reverse_alpha_ratio * velocity - mean) / var

    def get_score_from_denoised(self, denoised: Tensor, x: Tensor, t: Tensor) -> Tensor:
        """Transform denoised prediction to score.

        Parameters
        ----------
        denoised : Tensor
            Denoised model output (predicted ``x_1``).
        x : Tensor
            State ``x_t``.
        t : Tensor
            Time vector of shape ``(B,)``.

        Returns
        -------
        Tensor
            Score estimate.
        """
        t = expand_t_like_x(t, x)
        alpha_t, _ = self.compute_alpha_t(t)
        sigma_t, _ = self.compute_sigma_t(t)
        return (alpha_t * denoised - x) / (sigma_t**2)

    def get_noise_from_velocity(self, velocity: Tensor, x: Tensor, t: Tensor) -> Tensor:
        """Transform velocity prediction to noise.

        Parameters
        ----------
        velocity : Tensor
            Velocity model output.
        x : Tensor
            State ``x_t``.
        t : Tensor
            Time vector of shape ``(B,)``.

        Returns
        -------
        Tensor
            Noise estimate.
        """
        t = expand_t_like_x(t, x)
        alpha_t, d_alpha_t = self.compute_alpha_t(t)
        sigma_t, d_sigma_t = self.compute_sigma_t(t)
        mean = x
        reverse_alpha_ratio = alpha_t / d_alpha_t
        var = reverse_alpha_ratio * d_sigma_t - sigma_t
        return (reverse_alpha_ratio * velocity - mean) / var

    def get_velocity_from_score(self, score: Tensor, x: Tensor, t: Tensor) -> Tensor:
        """Transform score prediction to velocity.

        Parameters
        ----------
        score : Tensor
            Score model output.
        x : Tensor
            State ``x_t``.
        t : Tensor
            Time vector of shape ``(B,)``.

        Returns
        -------
        Tensor
            Velocity estimate.
        """
        t = expand_t_like_x(t, x)
        drift, var = self.compute_drift(x, t)
        return var * score - drift

    def compute_mu_t(self, t: Tensor, x0: Tensor, x1: Tensor) -> Tensor:
        """Compute the mean of the time-dependent density ``p_t``.

        Parameters
        ----------
        t : Tensor
            Time vector of shape ``(B,)``.
        x0 : Tensor
            Noise sample.
        x1 : Tensor
            Data sample.

        Returns
        -------
        Tensor
            Mean ``alpha_t * x1 + sigma_t * x0``.
        """
        t = expand_t_like_x(t, x1)
        alpha_t, _ = self.compute_alpha_t(t)
        sigma_t, _ = self.compute_sigma_t(t)
        return alpha_t * x1 + sigma_t * x0

    def compute_xt(self, t: Tensor, x0: Tensor, x1: Tensor) -> Tensor:
        """Sample ``x_t`` from the time-dependent density ``p_t``.

        Parameters
        ----------
        t : Tensor
            Time vector of shape ``(B,)``.
        x0 : Tensor
            Noise sample.
        x1 : Tensor
            Data sample.

        Returns
        -------
        Tensor
            Interpolated state ``x_t``.
        """
        return self.compute_mu_t(t, x0, x1)

    def compute_ut(self, t: Tensor, x0: Tensor, x1: Tensor, xt: Tensor) -> Tensor:
        """Compute the velocity field ``d/dt x_t``.

        Parameters
        ----------
        t : Tensor
            Time vector of shape ``(B,)``.
        x0 : Tensor
            Noise sample.
        x1 : Tensor
            Data sample.
        xt : Tensor
            Interpolated state (unused for linear plan).

        Returns
        -------
        Tensor
            Velocity target.
        """
        t = expand_t_like_x(t, x1)
        _, d_alpha_t = self.compute_alpha_t(t)
        _, d_sigma_t = self.compute_sigma_t(t)
        return d_alpha_t * x1 + d_sigma_t * x0

    def plan(self, t: Tensor, x0: Tensor, x1: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Compute the training triple ``(t, x_t, u_t)``.

        Parameters
        ----------
        t : Tensor
            Time vector of shape ``(B,)``.
        x0 : Tensor
            Noise sample.
        x1 : Tensor
            Data sample.

        Returns
        -------
        t : Tensor
            Unchanged time vector.
        xt : Tensor
            Interpolated state.
        ut : Tensor
            Velocity target.
        """
        xt = self.compute_xt(t, x0, x1)
        ut = self.compute_ut(t, x0, x1, xt)
        return t, xt, ut


class VPCPlan(ICPlan):
    """Variance-preserving coupling plan.

    Uses exponential coefficient schedules derived from
    a variance-preserving SDE with configurable noise bounds.

    Parameters
    ----------
    sigma_min : float
        Minimum noise scale.
    sigma_max : float
        Maximum noise scale.
    """

    def __init__(self, sigma_min: float = 0.1, sigma_max: float = 20.0) -> None:
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.log_mean_coeff = lambda t: (
            -0.25 * ((1 - t) ** 2) * (self.sigma_max - self.sigma_min) - 0.5 * (1 - t) * self.sigma_min
        )
        self.d_log_mean_coeff = lambda t: 0.5 * (1 - t) * (self.sigma_max - self.sigma_min) + 0.5 * self.sigma_min

    def compute_alpha_t(self, t: Tensor) -> tuple[Tensor, Tensor]:
        """Compute data coefficient ``exp(log_mean_coeff(t))``."""
        alpha_t = self.log_mean_coeff(t)
        alpha_t = torch.exp(alpha_t)
        d_alpha_t = alpha_t * self.d_log_mean_coeff(t)
        return alpha_t, d_alpha_t

    def compute_sigma_t(self, t: Tensor) -> tuple[Tensor, Tensor]:
        """Compute noise coefficient ``sqrt(1 - exp(2 * log_mean_coeff(t)))``."""
        p_sigma_t = 2 * self.log_mean_coeff(t)
        sigma_t = torch.sqrt(1 - torch.exp(p_sigma_t))
        d_sigma_t = torch.exp(p_sigma_t) * (2 * self.d_log_mean_coeff(t)) / (-2 * sigma_t)
        return sigma_t, d_sigma_t

    def compute_d_alpha_alpha_ratio_t(self, t: Tensor) -> Tensor:
        """Compute numerically stable ``d_alpha_t / alpha_t``."""
        return self.d_log_mean_coeff(t)

    def compute_drift(self, x: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        """Compute the VP SDE drift."""
        t = expand_t_like_x(t, x)
        beta_t = self.sigma_min + (1 - t) * (self.sigma_max - self.sigma_min)
        return -0.5 * beta_t * x, beta_t / 2


class GVPCPlan(ICPlan):
    """Geometric vector path coupling plan.

    Uses sinusoidal coefficient schedules: ``alpha_t = sin(pi*t/2)``,
    ``sigma_t = cos(pi*t/2)``.

    Parameters
    ----------
    sigma : float
        Noise scale (unused, reserved for extensions).
    """

    def __init__(self, sigma: float = 0.0) -> None:
        super().__init__(sigma)

    def compute_alpha_t(self, t: Tensor) -> tuple[Tensor, Tensor]:
        """Compute data coefficient ``sin(pi*t/2)``."""
        alpha_t = torch.sin(t * math.pi / 2)
        d_alpha_t = math.pi / 2 * torch.cos(t * math.pi / 2)
        return alpha_t, d_alpha_t

    def compute_sigma_t(self, t: Tensor) -> tuple[Tensor, Tensor]:
        """Compute noise coefficient ``cos(pi*t/2)``."""
        sigma_t = torch.cos(t * math.pi / 2)
        d_sigma_t = -math.pi / 2 * torch.sin(t * math.pi / 2)
        return sigma_t, d_sigma_t

    def compute_d_alpha_alpha_ratio_t(self, t: Tensor) -> Tensor:
        """Compute ``d_alpha_t / alpha_t = pi / (2 * tan(pi*t/2))``."""
        return math.pi / (2 * torch.tan(t * math.pi / 2))
