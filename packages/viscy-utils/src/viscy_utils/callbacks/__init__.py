import importlib

# Resolved lazily (PEP 562). VisCyCLI injects the health callbacks into every fit
# through this package, and OnlineEvalCallback needs scikit-learn, which only the
# ``eval`` extra installs, so an eager import would break fits on a base install.
_LAZY_ATTRS = {
    "EmbeddingSnapshotCallback": "viscy_utils.callbacks.embedding_snapshot",
    "EmbeddingWriter": "viscy_utils.callbacks.embedding_writer",
    "LinearLRCooldown": "viscy_utils.callbacks.linear_lr_cooldown",
    "MonitorHealthCheck": "viscy_utils.callbacks.monitor_health",
    "MonitorHealthError": "viscy_utils.callbacks.monitor_health",
    "OnlineEvalCallback": "viscy_utils.callbacks.online_eval",
    "OptimizerHealthCheck": "viscy_utils.callbacks.optimizer_health",
    "OptimizerHealthError": "viscy_utils.callbacks.optimizer_health",
    "HCSPredictionWriter": "viscy_utils.callbacks.prediction_writer",
}

__all__ = list(_LAZY_ATTRS)


def __getattr__(name: str):
    if name not in _LAZY_ATTRS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(_LAZY_ATTRS[name]), name)


def __dir__() -> list[str]:
    return sorted([*globals(), *__all__])
