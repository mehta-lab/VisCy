import importlib

# Resolved lazily (PEP 562) so importing a light submodule such as
# ``viscy_utils.prediction_metadata`` does not pull in torch and lightning.
_LAZY_ATTRS = {
    "configure_adamw_scheduler": "viscy_utils.optimizers",
    "detach_sample": "viscy_utils.log_images",
    "get_val_stats": "viscy_utils.mp_utils",
    "hist_clipping": "viscy_utils.normalize",
    "mp_wrapper": "viscy_utils.mp_utils",
    "render_images": "viscy_utils.log_images",
    "to_numpy": "viscy_utils.tensor_utils",
    "unzscore": "viscy_utils.normalize",
    "zscore": "viscy_utils.normalize",
}

__all__ = list(_LAZY_ATTRS)


def __getattr__(name: str):
    if name not in _LAZY_ATTRS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(_LAZY_ATTRS[name]), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(list(globals()) + __all__)
