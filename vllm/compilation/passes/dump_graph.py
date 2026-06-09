# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""FX Graph dump pass — walks the torch.fx.Graph and extracts op info."""

import os
import re
import sys
from collections import defaultdict

import torch
import torch.fx as fx
from torch.fx import GraphModule


def _safe_shape(t: torch.Tensor) -> list[int | str]:
    """Convert tensor shape to list[int|str], handling symbolic sizes."""
    result = []
    for s in t.shape:
        try:
            # Try to resolve symbolic size to int
            if hasattr(s, "node"):
                result.append(str(s))
            else:
                result.append(int(s))
        except (TypeError, ValueError):
            result.append(str(s))
    return result

from vllm.envs import VLLM_DUMP_FX_GRAPH, VLLM_DUMP_DIR
from vllm.profiler.graph_schema import (
    GraphDump,
    LayerDump,
    OpNode,
    TensorInfo,
    _get_model_hints,
)

LAYER_PATTERN = re.compile(r"model_layers_(\d+)")
EMBEDDING_KEYWORDS = {"embed", "tok_embeddings"}
LM_HEAD_KEYWORDS = {"lm_head", "head"}


def _resolve_layer_name(name: str) -> str:
    """Resolve a node name to its layer group."""
    m = LAYER_PATTERN.search(name)
    if m:
        return f"model.layers.{m.group(1)}"
    lower = name.lower()
    for kw in EMBEDDING_KEYWORDS:
        if kw in lower:
            return "embedding"
    for kw in LM_HEAD_KEYWORDS:
        if kw in lower:
            return "lm_head"
    return "other"


def _is_compute_op(node: fx.Node) -> bool:
    """Return True if this node represents a computation."""
    return node.op not in ("placeholder", "output", "get_attr")


def _extract_fake_tensor(node: fx.Node) -> list[TensorInfo]:
    """Extract fake tensor info from a node's metadata."""
    fake = node.meta.get("example_value", None)
    if fake is None:
        fake = node.meta.get("val", None)
    if fake is None:
        return []
    if isinstance(fake, (tuple, list)):
        results = []
        for i, f in enumerate(fake):
            if isinstance(f, torch.Tensor):
                results.append(TensorInfo(
                    name=f"{node.name}_out_{i}",
                    shape=_safe_shape(f),
                    dtype=str(f.dtype),
                    device=str(f.device),
                ))
        return results
    if isinstance(fake, torch.Tensor):
        return [TensorInfo(
            name="output",
            shape=_safe_shape(fake),
            dtype=str(fake.dtype),
            device=str(fake.device),
        )]
    return []


def _extract_inputs(node: fx.Node) -> list[TensorInfo]:
    """Extract input tensor info from an FX node's args."""
    inputs = []
    for arg in node.args:
        if isinstance(arg, fx.Node):
            fake = arg.meta.get("example_value", None)
            if fake is None:
                fake = arg.meta.get("val", None)
            if fake is not None and isinstance(fake, torch.Tensor):
                inputs.append(TensorInfo(
                    name=arg.name,
                    shape=_safe_shape(fake),
                    dtype=str(fake.dtype),
                    device=str(fake.device),
                ))
    return inputs


def _infer_subgraph(node: fx.Node) -> str:
    """Heuristically classify the node into a subgraph category."""
    name = node.name.lower()
    target = str(node.target).lower() if node.target else ""
    combined = f"{name} {target}"
    attn_kw = ("attn", "attention", "q_proj", "k_proj", "v_proj",
               "o_proj", "rotary")
    mlp_kw = ("mlp", "gate_proj", "up_proj", "down_proj", "feed_forward")
    norm_kw = ("norm", "rms")
    if any(k in combined for k in attn_kw):
        return "attention"
    if any(k in combined for k in mlp_kw):
        return "mlp"
    if any(k in combined for k in norm_kw):
        return "norm"
    if any(k in combined for k in ("embed",)):
        return "embedding"
    return "other"


_TARGET_NAME_MAP: dict[type, str] = {}
try:
    import operator
    for _name in ("getitem", "add", "mul", "truediv", "sub", "pow",
                  "getattr", "eq", "ne", "lt", "gt", "le", "ge"):
        _fn = getattr(operator, _name, None)
        if _fn is not None:
            _TARGET_NAME_MAP[_fn] = f"operator.{_name}"
except Exception:
    pass
_TARGET_NAME_MAP[int] = "int"
_TARGET_NAME_MAP[float] = "float"


def _format_op_name(node: fx.Node) -> str:
    """Return a clean operator name from an FX node target."""
    if node.target is None:
        return node.op
    target = node.target
    if isinstance(target, str):
        return target
    if callable(target):
        # Check the mapping first
        mapped = _TARGET_NAME_MAP.get(target) or _TARGET_NAME_MAP.get(type(target))
        if mapped:
            return mapped
        # For functions/modules, use __name__ or qualname
        name = getattr(target, "__name__", None)
        if name and name != "<lambda>":
            module = getattr(target, "__module__", "")
            if module and module != "builtins":
                return f"{module}.{name}"
            return name
    return str(target)[:80]


def _format_display_name(node: fx.Node) -> str:
    """Format a human-readable display name for the op."""
    name = _format_op_name(node)
    fake = node.meta.get("example_value", None)
    if fake is not None and isinstance(fake, torch.Tensor):
        shape_str = "×".join(str(s) for s in _safe_shape(fake))
        return f"{name}(→{shape_str})"
    return name


class DumpGraphPass:
    """Walk an FX graph and dump op information to JSON.

    Hooked into VllmBackend.compile() at split boundaries.
    """

    @staticmethod
    def _walk_graph(
        graph: fx.Graph,
        parent_module: torch.nn.Module | None,
        prefix: str,
    ) -> list[OpNode]:
        """Walk graph nodes, recursing into submodules when available."""
        ops: list[OpNode] = []
        op_id = 0
        for node in graph.nodes:
            if not _is_compute_op(node):
                continue
            # Recurse into call_module nodes
            if node.op == "call_module" and parent_module is not None:
                sub_name = node.target
                sub_graph = None
                sub_parent = None
                if hasattr(parent_module, sub_name):
                    sub_mod = getattr(parent_module, sub_name)
                    if isinstance(sub_mod, GraphModule):
                        sub_graph = sub_mod.graph
                        sub_parent = sub_mod
                    elif hasattr(sub_mod, "graph") and isinstance(
                        getattr(sub_mod, "graph"), GraphModule
                    ):
                        # PiecewiseBackend and similar wrappers
                        sub_graph = sub_mod.graph.graph
                        sub_parent = sub_mod.graph
                if sub_graph is not None:
                    sub_ops = DumpGraphPass._walk_graph(
                        sub_graph, sub_parent,
                        f"{prefix}/{sub_name}",
                    )
                    for s_op in sub_ops:
                        s_op.id = op_id
                        s_op.metadata["parent_module"] = str(node.target)
                        ops.append(s_op)
                        op_id += 1
                    continue
            op = OpNode(
                id=op_id,
                op_type="fx",
                op_name=_format_op_name(node),
                display_name=_format_display_name(node),
                inputs=_extract_inputs(node),
                outputs=_extract_fake_tensor(node),
                metadata={
                    "subgraph": _infer_subgraph(node),
                    "node_name": str(node.name),
                },
            )
            ops.append(op)
            op_id += 1
        return ops

    @staticmethod
    def dump(
        graph_or_module: fx.Graph | fx.GraphModule,
        stage: str,
    ) -> str | None:
        """Walk the graph and write JSON dump. Returns path or None if disabled."""
        if not VLLM_DUMP_FX_GRAPH:
            return None

        config_hints = _get_model_hints()
        dump = GraphDump(
            version="1.1",
            mode="fx_compile",
            model_name=config_hints.get("model_type", "unknown"),
            model_config=config_hints,
            num_layers=config_hints.get("num_hidden_layers", 0),
            layers=[],
        )

        if isinstance(graph_or_module, GraphModule):
            graph = graph_or_module.graph
            parent = graph_or_module
        else:
            graph = graph_or_module
            parent = None

        ops = DumpGraphPass._walk_graph(graph, parent, "")

        # Put all ops in a single group (FX graph nodes lack layer metadata)
        dump.layers = [LayerDump(
            name="graph",
            count=1,
            representative_index=0,
            ops=ops,
        )]

        output_dir = VLLM_DUMP_DIR or "."
        path = os.path.join(output_dir, f"fx_graph_dump__{stage}.json")
        dump.to_json(path)
        return path
