import numpy as np


def sample_tau(
    tau_min: int,
    tau_max: int,
    rng: np.random.Generator,
    decay_rate: float = 2.0,
) -> int:
    """Sample a temporal offset using exponential decay.

    Probabilities are proportional to exp(-decay_rate * (tau - tau_min) / (tau_max - tau_min)),
    favoring small temporal offsets near tau_min.

    Parameters
    ----------
    tau_min : int
        Minimum tau value (inclusive).
    tau_max : int
        Maximum tau value (inclusive).
    rng : numpy.random.Generator
        Random number generator for reproducibility.
    decay_rate : float
        Exponential decay rate. 0.0 = uniform. Higher = stronger bias toward tau_min. Default: 2.0.

    Returns
    -------
    int
        Sampled tau value in [tau_min, tau_max].
    """
    if tau_min == tau_max:
        return int(tau_min)
    taus = np.arange(tau_min, tau_max + 1)
    weights = np.exp(-decay_rate * (taus - tau_min) / (tau_max - tau_min))
    weights /= weights.sum()
    return int(rng.choice(taus, p=weights))
