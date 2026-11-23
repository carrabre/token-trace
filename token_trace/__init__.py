"""Forward-pass tracing for instrumentable PyTorch models."""

__all__ = [
    "Call", "Layer", "Plan", "Probe", "TracePoint", "inspect_model", "linear_terms",
    "packed_token_row", "parameter_value", "token_row", "trace", "trace_many",
]


def __getattr__(name):
    # Replay and CLI discovery are usable without importing the model runtime.
    if name in ("inspect_model", "parameter_value"):
        from . import inventory

        value = getattr(inventory, name)
    elif name in __all__:
        from . import core

        value = getattr(core, name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value
