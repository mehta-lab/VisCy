"""Dynacell: benchmark virtual staining application."""

__all__ = ["DynacellFlowMatching", "DynacellUNet"]


def __getattr__(name: str):
    if name in {"DynacellFlowMatching", "DynacellUNet"}:
        from dynacell.engine import DynacellFlowMatching, DynacellUNet

        return {"DynacellFlowMatching": DynacellFlowMatching, "DynacellUNet": DynacellUNet}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
