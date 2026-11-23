"""Inspect registered model structure and logical parameter values without inference."""
from collections.abc import Sequence
from math import isfinite, prod
from numbers import Integral

import torch
from torch import Tensor, nn
from torch.nn.parameter import is_lazy


def _registered_modules(model: nn.Module, *, aliases: bool = True):
    """Yield registered paths in order, including shared modules and their descendants."""
    seen = {}
    pending = [("", None, model, frozenset())]
    while pending:
        path, parent, module, ancestors = pending.pop()
        identifier = id(module)
        if identifier in seen:
            if not aliases:
                continue
            alias_of = seen[identifier]
        else:
            seen[identifier] = path
            alias_of = None
        yield path, parent, module, alias_of
        if identifier in ancestors:
            continue
        ancestors = ancestors | {identifier}
        # named_children() deduplicates aliases at the same depth.
        for name, child in reversed(module._modules.items()):
            if child is not None:
                child_path = f"{path}.{name}" if path else name
                pending.append((child_path, path, child, ancestors))


def _qualified_class(module: nn.Module) -> str:
    cls = type(module)
    return f"{cls.__module__}.{cls.__qualname__}"


def _storage(tensor: Tensor) -> str:
    if is_lazy(tensor):
        return "uninitialized"
    if tensor.device.type == "meta":
        return "meta"
    if tensor.is_nested:
        return "nested"
    if tensor.is_quantized:
        return "quantized"
    return "strided" if tensor.layout == torch.strided else "non_strided"


def _tensor_metadata(name, module, tensor, alias_of):
    storage = _storage(tensor)
    shape = None
    if storage not in {"uninitialized", "nested"}:
        shape = [size if isinstance(size, int) else None for size in tensor.shape]
    return {
        "name": name,
        "module": module,
        "shape": shape,
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "device": str(tensor.device),
        "layout": str(tensor.layout).removeprefix("torch."),
        "numel": prod(shape) if shape is not None and None not in shape else None,
        "storage": storage,
        "alias_of": alias_of,
    }


def inspect_model(model: nn.Module) -> dict:
    """Return metadata for every registered module, parameter and buffer path.

    Aliases point to the first registered path for the same object. Distinct
    objects with overlapping storage are counted separately. Inventory has no
    information about which paths or weights a particular input exercises.
    """
    modules, parameters, buffers = [], [], []
    seen_parameters, seen_buffers = {}, {}
    for path, parent, module, alias_of in _registered_modules(model):
        modules.append({"path": path, "parent": parent, "class": _qualified_class(module),
                        "alias_of": alias_of})
        for members, records, seen in (
            (module._parameters, parameters, seen_parameters),
            (module._buffers, buffers, seen_buffers),
        ):
            for child, tensor in members.items():
                if tensor is None:
                    continue
                name = f"{path}.{child}" if path else child
                record = _tensor_metadata(name, path, tensor, seen.get(id(tensor)))
                seen.setdefault(id(tensor), name)
                if records is parameters:
                    record["requires_grad"] = tensor.requires_grad
                else:
                    record["persistent"] = child not in module._non_persistent_buffers_set
                records.append(record)
    sizes = [entry["numel"] for entry in parameters if entry["alias_of"] is None]
    return {
        "schema": "token-trace/model/1",
        "model_class": _qualified_class(model),
        "modules": modules,
        "parameters": parameters,
        "buffers": buffers,
        "totals": {
            "module_paths": len(modules),
            "unique_modules": sum(entry["alias_of"] is None for entry in modules),
            "parameter_paths": len(parameters),
            "unique_parameters": len(seen_parameters),
            "parameter_elements": sum(sizes) if None not in sizes else None,
            "buffer_paths": len(buffers),
            "unique_buffers": len(seen_buffers),
        },
    }


def _parameter_name(model: nn.Module, parameter: nn.Parameter) -> str:
    for path, _, module, _ in _registered_modules(model, aliases=False):
        for child, registered in module._parameters.items():
            if registered is parameter:
                return f"{path}.{child}" if path else child
    raise ValueError("Parameter is outside the registered model tree.")


def parameter_value(model: nn.Module, name: str, coordinate: Sequence[int]) -> dict:
    """Read a registered dense tensor at a full zero-based coordinate.

    A scalar parameter uses an empty coordinate. The value comes from the
    current runtime tensor and may not match a checkpoint's on-disk encoding.
    Packed or unavailable storage needs a runtime-specific adapter.
    """
    try:
        parameter = model.get_parameter(name)
    except (AttributeError, TypeError) as error:
        raise ValueError(f"Unknown parameter {name!r}.") from error
    canonical = _parameter_name(model, parameter)
    record = _tensor_metadata(name, name.rpartition(".")[0], parameter,
                              None if canonical == name else canonical)
    if record["storage"] != "strided":
        raise ValueError(f"{name!r} uses {record['storage']} storage and needs a parameter adapter.")
    indices = tuple(coordinate)
    shape = record["shape"]
    if len(indices) != len(shape) or any(
        size is None or not isinstance(index, Integral) or isinstance(index, bool)
        or not 0 <= index < size for index, size in zip(indices, shape)
    ):
        raise ValueError(f"Coordinate {indices} is outside {name!r} with shape {shape}.")
    indices = tuple(int(index) for index in indices)
    try:
        value = parameter.detach()[indices].item()
    except (RuntimeError, NotImplementedError, TypeError, ValueError) as error:
        raise ValueError(f"{name!r} cannot expose a scalar at {indices} in this runtime.") from error
    if isinstance(value, complex):
        if not (isfinite(value.real) and isfinite(value.imag)):
            raise ValueError(f"{name!r} contains a nonfinite value at {indices}.")
        value = {"real": value.real, "imag": value.imag}
    if isinstance(value, float) and not isfinite(value):
        raise ValueError(f"{name!r} contains a nonfinite value at {indices}.")
    return {**record, "requires_grad": parameter.requires_grad,
            "coordinate": list(indices), "value": value, "value_source": "runtime_parameter_tensor"}
