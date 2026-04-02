"""Flow-matching transport module.

Factory for creating configured :class:`Transport` objects with
path sampling, loss computation, and ODE/SDE inference.

Requires optional dependency: ``pip install viscy-models[celldiff]``
(includes ``torchdiffeq``).
"""

from viscy_models.celldiff.modules.transport.transport import (
    ModelType,
    PathType,
    Sampler,
    Transport,
    WeightType,
)

__all__ = [
    "ModelType",
    "PathType",
    "Sampler",
    "Transport",
    "WeightType",
    "create_transport",
]


def create_transport(
    path_type: str = "Linear",
    prediction: str = "velocity",
    loss_weight: str | None = None,
    train_eps: float | None = None,
    sample_eps: float | None = None,
) -> Transport:
    """Create a configured Transport object.

    Parameters
    ----------
    path_type : str
        Path interpolation type: ``"Linear"``, ``"GVP"``, or ``"VP"``.
    prediction : str
        Model prediction target: ``"velocity"``, ``"noise"``, ``"score"``,
        or ``"denoised"``.
    loss_weight : str or None
        Loss weighting: ``None``, ``"velocity"``, or ``"likelihood"``.
    train_eps : float or None
        Training epsilon. Auto-set based on path/model type if ``None``.
    sample_eps : float or None
        Sampling epsilon. Auto-set based on path/model type if ``None``.

    Returns
    -------
    Transport
        Configured transport object.
    """
    model_type_map = {
        "noise": ModelType.NOISE,
        "score": ModelType.SCORE,
        "velocity": ModelType.VELOCITY,
        "denoised": ModelType.DENOISED,
    }
    if prediction not in model_type_map:
        raise ValueError(f"Unknown prediction {prediction!r}, expected one of {set(model_type_map)}")
    model_type = model_type_map[prediction]

    loss_type_map: dict[str | None, WeightType] = {
        None: WeightType.NONE,
        "velocity": WeightType.VELOCITY,
        "likelihood": WeightType.LIKELIHOOD,
    }
    if loss_weight not in loss_type_map:
        raise ValueError(f"Unknown loss_weight {loss_weight!r}, expected one of {set(loss_type_map)}")
    loss_type = loss_type_map[loss_weight]

    path_choice = {
        "Linear": PathType.LINEAR,
        "GVP": PathType.GVP,
        "VP": PathType.VP,
    }
    resolved_path_type = path_choice[path_type]

    if resolved_path_type == PathType.VP:
        train_eps = 1e-5 if train_eps is None else train_eps
        sample_eps = 1e-3 if sample_eps is None else sample_eps
    elif resolved_path_type in (PathType.GVP, PathType.LINEAR) and model_type != ModelType.VELOCITY:
        train_eps = 1e-3 if train_eps is None else train_eps
        sample_eps = 1e-3 if sample_eps is None else sample_eps
    else:
        # velocity + [GVP, LINEAR] is stable everywhere
        train_eps = 0 if train_eps is None else train_eps
        sample_eps = 0 if sample_eps is None else sample_eps

    return Transport(
        model_type=model_type,
        path_type=resolved_path_type,
        loss_type=loss_type,
        train_eps=train_eps,
        sample_eps=sample_eps,
    )
