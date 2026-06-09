# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Worker-process eager tracer using nn.Module forward hooks.

This runs INSIDE the EngineCore worker process (unlike EagerTracer which
ran in the main process and couldn't see worker ops). Uses the same pattern
as vllm/utils/nvtx_pytorch_hooks.py (PytHooks).
"""

import os
from collections import defaultdict
from typing import Any

import torch
import torch.nn as nn

from vllm.profiler.graph_schema import (
    GraphDump,
    LayerDump,
    OpNode,
    TensorInfo,
    _get_model_hints,
    deduplicate_layers,
)

# Global state — accumulates tracer data during forward pass.
_trace_ops: list[OpNode] = []
_module_to_name: dict[nn.Module, str] = {}
_dumped: bool = False


def _tensor_info(t: torch.Tensor, name: str = "tensor") -> TensorInfo:
    """Convert a tensor to TensorInfo, handling symbolic shapes."""
    shape = []
    for s in t.shape:
        try:
            shape.append(int(s))
        except (TypeError, ValueError):
            shape.append(str(s))
    return TensorInfo(
        name=name,
        shape=shape,
        dtype=str(t.dtype),
        device=str(t.device),
    )


def _pre_hook(module: nn.Module, args: tuple) -> None:
    """Called before module forward. Records input shapes."""
    module_name = _module_to_name.get(module, type(module).__name__)
    inputs = []
    for i, arg in enumerate(args):
        if isinstance(arg, torch.Tensor):
            inputs.append(_tensor_info(arg, name=f"in_{i}"))
        elif isinstance(arg, (tuple, list)):
            for j, item in enumerate(arg):
                if isinstance(item, torch.Tensor):
                    inputs.append(_tensor_info(item, name=f"in_{i}_{j}"))

    op = OpNode(
        id=len(_trace_ops),
        op_type="eager",
        op_name=module_name,
        display_name=f"{module_name}(...→...)",
        inputs=inputs,
        outputs=[],
        metadata={"module_type": type(module).__name__},
    )
    _trace_ops.append(op)


def _hook(module: nn.Module, args: tuple, output: Any) -> Any:
    """Called after module forward. Records output shapes."""
    module_name = _module_to_name.get(module, type(module).__name__)
    # Find matching pre-hook OpNode and add outputs
    for op in reversed(_trace_ops):
        if op.op_name == module_name and not op.outputs:
            if isinstance(output, torch.Tensor):
                op.outputs.append(_tensor_info(output, name="out"))
            elif isinstance(output, (tuple, list)):
                for j, item in enumerate(output):
                    if isinstance(item, torch.Tensor):
                        op.outputs.append(_tensor_info(item, name=f"out_{j}"))
            # Update display name with output shapes
            out_shapes = "×".join(
                "×".join(str(d) for d in o.shape) for o in op.outputs
            )
            op.display_name = f"{module_name}(→{out_shapes})"
            break
    return output


_SKIP_TYPES = (
    torch.nn.Identity,
    torch.nn.Dropout,
    torch.nn.Dropout1d,
    torch.nn.Dropout2d,
    torch.nn.Dropout3d,
)


def register_tracer(model: nn.Module) -> None:
    """Walk model tree and register forward hooks on all modules."""
    for name, module in model.named_modules():
        if isinstance(module, _SKIP_TYPES):
            continue
        module.register_forward_pre_hook(_pre_hook)
        module.register_forward_hook(_hook)
        _module_to_name[module] = name


def dump_if_needed(output_dir: str = ".", filename: str = "eager_trace_dump.json") -> None:
    """Dump accumulated trace data to JSON. Called after first forward pass."""
    global _dumped
    if _dumped:
        return
    _dumped = True

    if not _trace_ops:
        return

    config_hints = _get_model_hints()

    # Group ops by layer
    by_layer: dict[str, list[OpNode]] = defaultdict(list)
    for op in _trace_ops:
        # Extract layer from module name: "model.layers.0.self_attn.q_proj" -> "model.layers.0"
        parts = op.op_name.split(".")
        layer_name = "other"
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts):
                layer_name = f"model.layers.{parts[i + 1]}"
                break
        by_layer[layer_name].append(op)

    layers_raw = dict(by_layer)
    deduped = deduplicate_layers(layers_raw, config_hints)

    dump = GraphDump(
        version="1.1",
        mode="eager",
        model_name=config_hints.get("model_type", "unknown"),
        model_config=config_hints,
        num_layers=config_hints.get("num_hidden_layers", 0),
        layers=deduped,
    )

    path = os.path.join(output_dir, filename)
    dump.to_json(path)
