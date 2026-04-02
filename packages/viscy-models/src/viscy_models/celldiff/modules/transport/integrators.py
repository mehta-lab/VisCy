"""ODE and SDE solver classes for flow-matching transport.

Uses ``torchdiffeq`` for adaptive ODE integration and provides
fixed-step SDE solvers (Euler-Maruyama, Heun).
"""

import torch
from torch import Tensor
from torchdiffeq import odeint


class SDESolver:
    """Fixed-step SDE solver.

    Parameters
    ----------
    drift : callable
        Drift function ``f(x, t, model, **kwargs) -> Tensor``.
    diffusion : callable
        Diffusion function ``g(x, t) -> Tensor``.
    t0 : float
        Start time.
    t1 : float
        End time (must be > ``t0``).
    num_steps : int
        Number of integration steps.
    sampler_type : str
        Solver type: ``"Euler"`` or ``"Heun"``.
    """

    def __init__(
        self,
        drift: callable,
        diffusion: callable,
        *,
        t0: float,
        t1: float,
        num_steps: int,
        sampler_type: str,
    ) -> None:
        if t0 >= t1:
            raise ValueError("SDE solver requires t0 < t1")
        self.num_timesteps = num_steps
        self.t = torch.linspace(t0, t1, num_steps)
        self.dt = self.t[1] - self.t[0]
        self.drift = drift
        self.diffusion = diffusion
        self.sampler_type = sampler_type

    def _euler_maruyama_step(
        self,
        x: Tensor,
        mean_x: Tensor,
        t: float,
        model: callable,
        **model_kwargs,
    ) -> tuple[Tensor, Tensor]:
        """Single Euler-Maruyama step."""
        w_cur = torch.randn_like(x)
        t_vec = x.new_ones(x.size(0)) * t
        dw = w_cur * torch.sqrt(self.dt)
        drift = self.drift(x, t_vec, model, **model_kwargs)
        diffusion = self.diffusion(x, t_vec)
        mean_x = x + drift * self.dt
        x = mean_x + torch.sqrt(2 * diffusion) * dw
        return x, mean_x

    def _heun_step(
        self,
        x: Tensor,
        _mean_x: Tensor,
        t: float,
        model: callable,
        **model_kwargs,
    ) -> tuple[Tensor, Tensor]:
        """Single Heun SDE step."""
        w_cur = torch.randn_like(x)
        dw = w_cur * torch.sqrt(self.dt)
        t_cur = x.new_ones(x.size(0)) * t
        diffusion = self.diffusion(x, t_cur)
        xhat = x + torch.sqrt(2 * diffusion) * dw
        k1 = self.drift(xhat, t_cur, model, **model_kwargs)
        xp = xhat + self.dt * k1
        k2 = self.drift(xp, t_cur + self.dt, model, **model_kwargs)
        return xhat + 0.5 * self.dt * (k1 + k2), xhat

    def _forward_fn(self) -> callable:
        """Select the step function based on sampler type."""
        sampler_dict = {
            "Euler": self._euler_maruyama_step,
            "Heun": self._heun_step,
        }
        if self.sampler_type not in sampler_dict:
            raise KeyError(f"Sampler type {self.sampler_type!r} not implemented. Choose from {set(sampler_dict)}.")
        return sampler_dict[self.sampler_type]

    def sample(self, init: Tensor, model: callable, **model_kwargs) -> Tensor:
        """Run the forward SDE loop and return the final state.

        Parameters
        ----------
        init : Tensor
            Initial state.
        model : callable
            Velocity/score model.
        **model_kwargs
            Extra arguments forwarded to ``model``.

        Returns
        -------
        Tensor
            Final state after integration.
        """
        x = init
        mean_x = init
        sampler = self._forward_fn()
        for ti in self.t[:-1]:
            with torch.no_grad():
                x, mean_x = sampler(x, mean_x, ti, model, **model_kwargs)
        return x


class ODESolver:
    """ODE solver using ``torchdiffeq.odeint``.

    Parameters
    ----------
    drift : callable
        Drift function ``f(x, t, model, **kwargs) -> Tensor``.
    t0 : float
        Start time.
    t1 : float
        End time (must be > ``t0``).
    sampler_type : str
        ODE solver method (e.g. ``"dopri5"``, ``"euler"``).
    num_steps : int
        Number of time points (for fixed-step methods, this is the step count;
        for adaptive methods, this controls the output grid).
    atol : float
        Absolute error tolerance.
    rtol : float
        Relative error tolerance.
    """

    def __init__(
        self,
        drift: callable,
        *,
        t0: float,
        t1: float,
        sampler_type: str,
        num_steps: int,
        atol: float,
        rtol: float,
    ) -> None:
        if t0 >= t1:
            raise ValueError("ODE solver requires t0 < t1")
        self.drift = drift
        self.t = torch.linspace(t0, t1, num_steps)
        self.atol = atol
        self.rtol = rtol
        self.sampler_type = sampler_type

    def sample(self, x: Tensor | tuple, model: callable, **model_kwargs) -> Tensor:
        """Integrate the ODE from ``t0`` to ``t1``.

        Parameters
        ----------
        x : Tensor or tuple[Tensor, ...]
            Initial state (tuple for likelihood computation).
        model : callable
            Velocity/score model.
        **model_kwargs
            Extra arguments forwarded to ``model``.

        Returns
        -------
        Tensor
            Trajectory of shape ``(num_steps, B, ...)``.
        """
        device = x[0].device if isinstance(x, tuple) else x.device

        def _fn(t: Tensor, x: Tensor | tuple) -> Tensor | tuple:
            batch_size = x[0].size(0) if isinstance(x, tuple) else x.size(0)
            t_vec = torch.ones(batch_size, device=device) * t
            return self.drift(x, t_vec, model, **model_kwargs)

        t = self.t.to(device)
        atol = [self.atol] * len(x) if isinstance(x, tuple) else [self.atol]
        rtol = [self.rtol] * len(x) if isinstance(x, tuple) else [self.rtol]
        return odeint(
            _fn,
            x,
            t,
            method=self.sampler_type,
            atol=atol,
            rtol=rtol,
        )
