"""Dynacell: benchmark virtual staining application."""

__all__ = ["DynacellFlowMatching", "DynacellUNet"]


def __getattr__(name: str):
    # Lazy imports to avoid pulling in heavy training deps on every import.
    if name == "DynacellFlowMatching":
        from dynacell.engine import DynacellFlowMatching

        return DynacellFlowMatching
    if name == "DynacellUNet":
        from dynacell.engine import DynacellUNet

        return DynacellUNet
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
