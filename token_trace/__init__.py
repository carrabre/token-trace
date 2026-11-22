"""Forward-pass tracing for instrumentable PyTorch models."""

__all__ = [
    "Call", "Layer", "Plan", "Probe", "TracePoint", "linear_terms",
    "packed_token_row", "token_row", "trace", "trace_many",
]


def __getattr__(name):
    # Replay and CLI discovery are usable without importing the model runtime.
    if name in __all__:
        from . import core

        value = getattr(core, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
