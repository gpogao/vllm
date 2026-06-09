# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unified data model for computation graph dumps."""

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class TensorInfo:
    name: str
    shape: list[int | str]
    dtype: str
    device: str


@dataclass
class OpNode:
    id: int
    op_type: str  # "fx" or "eager"
    op_name: str
    display_name: str
    inputs: list[TensorInfo]
    outputs: list[TensorInfo]
    metadata: dict[str, Any]


@dataclass
class LayerDump:
    name: str
    count: int
    representative_index: int
    ops: list[OpNode]


@dataclass
class GraphDump:
    version: str
    mode: str  # "fx_compile" or "eager"
    model_name: str
    model_config: dict[str, Any]
    num_layers: int
    layers: list[LayerDump]

    def to_json(self, path: str) -> None:
        data = {
            "version": self.version,
            "mode": self.mode,
            "model_name": self.model_name,
            "model_config": self.model_config,
            "num_layers": self.num_layers,
            "layers": [
                {
                    "name": l.name,
                    "count": l.count,
                    "representative_index": l.representative_index,
                    "ops": [asdict(op) for op in l.ops],
                }
                for l in self.layers
            ],
        }
        Path(path).write_text(json.dumps(data, indent=2, default=str))

    @classmethod
    def from_json(cls, path: str) -> "GraphDump":
        data = json.loads(Path(path).read_text())
        layers = []
        for l in data["layers"]:
            ops = []
            for op_data in l["ops"]:
                inputs = [TensorInfo(**i) for i in op_data["inputs"]]
                outputs = [TensorInfo(**o) for o in op_data["outputs"]]
                ops.append(OpNode(
                    id=op_data["id"],
                    op_type=op_data["op_type"],
                    op_name=op_data["op_name"],
                    display_name=op_data["display_name"],
                    inputs=inputs,
                    outputs=outputs,
                    metadata=op_data.get("metadata", {}),
                ))
            layers.append(LayerDump(
                name=l["name"],
                count=l["count"],
                representative_index=l["representative_index"],
                ops=ops,
            ))
        return cls(
            version=data["version"],
            mode=data["mode"],
            model_name=data["model_name"],
            model_config=data.get("model_config", {}),
            num_layers=data["num_layers"],
            layers=layers,
        )


def _get_model_hints() -> dict[str, Any]:
    """best-effort: extract model dimension info from runtime config."""
    try:
        from vllm.config import get_current_vllm_config
        cfg = get_current_vllm_config()
    except Exception:
        return {}
    if cfg is None or cfg.model_config is None:
        return {}
    hf = cfg.model_config.hf_config
    hints: dict[str, Any] = {}
    for key in ("hidden_size", "intermediate_size", "num_attention_heads",
                 "num_key_value_heads", "head_dim", "vocab_size",
                 "num_hidden_layers", "model_type", "architectures"):
        if hasattr(hf, key):
            hints[key] = getattr(hf, key)
    for sub in ("text_config", "vision_config", "audio_config"):
        sub_cfg = getattr(hf, sub, None)
        if sub_cfg is not None:
            for key in ("hidden_size", "intermediate_size", "num_attention_heads",
                         "num_key_value_heads", "head_dim", "num_hidden_layers"):
                if hasattr(sub_cfg, key):
                    hints[f"{sub}.{key}"] = getattr(sub_cfg, key)
    return hints


def hash_ops(ops: list[OpNode]) -> str:
    parts = []
    for op in ops:
        shape_strs = []
        for inp in op.inputs:
            shape_strs.append(",".join(str(d) for d in inp.shape))
        parts.append(f"{op.op_name}[{'|'.join(shape_strs)}]")
    content = "".join(parts)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def format_range(indices: list[int]) -> str:
    """Format a list of sorted ints into a compact range string."""
    if not indices:
        return "{}"
    sorted_ix = sorted(indices)
    if len(sorted_ix) == 1:
        return str(sorted_ix[0])
    # Check if consecutive
    is_consecutive = all(
        sorted_ix[i] == sorted_ix[0] + i for i in range(len(sorted_ix))
    )
    if is_consecutive:
        return f"{sorted_ix[0]},{sorted_ix[1]},...,{sorted_ix[-1]}"
    return ",".join(str(i) for i in sorted_ix)


def deduplicate_layers(
    layers: dict[str, list[OpNode]],
    config_hints: dict[str, Any] | None = None,
) -> list[LayerDump]:
    groups: dict[str, list[int]] = defaultdict(list)
    for name, ops in sorted(layers.items()):
        m = re.search(r"layers\.(\d+)", name)
        if m:
            idx = int(m.group(1))
            groups[hash_ops(ops)].append(idx)
        else:
            # non-layer entries kept as-is
            groups[f"_{name}"].append(-1)

    layer_types: list[str] | None = None
    if config_hints:
        layer_types = (
            config_hints.get("text_config.layer_types")
            or config_hints.get("layer_types")
        )

    result: list[LayerDump] = []
    for hash_key, indices in groups.items():
        if indices == [-1]:
            continue  # skip unknown entries

        # best-effort: enrich display name with layer_types
        extra = ""
        if layer_types and isinstance(layer_types, list):
            types_in_group = sorted({
                layer_types[i] for i in indices
                if 0 <= i < len(layer_types)
            })
            if len(types_in_group) == 1:
                extra = f" [{types_in_group[0]}]"
            elif types_in_group:
                extra = f" [mixed: {types_in_group}]"

        first = indices[0]
        range_str = format_range(indices)
        result.append(LayerDump(
            name=f"model.layers.{{{range_str}}}{extra}",
            count=len(indices),
            representative_index=first,
            ops=layers[f"model.layers.{first}"],
        ))
    return result
