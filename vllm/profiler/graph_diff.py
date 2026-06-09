# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Phase 3: diff tool for comparing FX graph dump vs eager trace dump."""

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field

from vllm.profiler.graph_schema import GraphDump, LayerDump, OpNode

MATCH_THRESHOLD = 35  # balanced for cross-granularity semantic matching


@dataclass
class DiffSummary:
    total_fx_ops: int = 0
    total_eager_ops: int = 0
    matched: int = 0
    fused: int = 0
    decomposed: int = 0
    compile_only: int = 0
    eager_only: int = 0
    shape_mismatches: int = 0


@dataclass
class DiffEntry:
    type: str  # match | fused | decomposed | compile_only | eager_only | shape_mismatch
    fx_op: OpNode | None = None
    eager_op: OpNode | None = None
    score: float | None = None


@dataclass
class LayerDiff:
    layer_name: str
    entries: list[DiffEntry] = field(default_factory=list)


@dataclass
class DiffReport:
    model_name: str
    source_fx: str
    source_eager: str
    summary: DiffSummary
    layers: list[LayerDiff] = field(default_factory=list)


# Keywords extracted from op names for cross-granularity semantic matching.
# Phase 1 uses Inductor function names (rms_norm, linear, silu...)
# Phase 2 uses module names and types (input_layernorm, QKVParallelLinear...)
_FX_KEYWORD_ALIASES: dict[str, str] = {
    "rms_norm": "rmsnorm",
    "rmsnorm": "rmsnorm",
    "layernorm": "rmsnorm",
    "layer_norm": "rmsnorm",
    "gemmarmsnorm": "rmsnorm",
    "input_layernorm": "rmsnorm",
    "post_attention_layernorm": "rmsnorm",
    "linear": "linear",
    "silu": "activation",
    "gelu": "activation",
    "silu_and_mul": "activation",
    "attention": "attention",
    "attn": "attention",
    "flash_attn": "attention",
    "gdn_attention": "attention",
    "self_attn": "attention",
    "embedding": "embedding",
    "embed": "embedding",
    "addmm": "linear",
    "matmul": "linear",
    "q_proj": "linear",
    "k_proj": "linear",
    "v_proj": "linear",
    "o_proj": "linear",
    "qkv_proj": "linear",
    "gate_proj": "linear",
    "up_proj": "linear",
    "down_proj": "linear",
    "in_proj": "linear",
    "rope": "rope",
    "rotary": "rope",
    "reshape": "reshape",
    "view": "reshape",
    "contiguous": "memory",
    "rsqrt": "rmsnorm",
    "pow": "rmsnorm",
    "mean": "rmsnorm",
    "getitem": "index",
    "setitem": "index",
    "split": "split",
    "chunk": "split",
    "cat": "cat",
    "unsqueeze": "reshape",
    "flatten": "reshape",
    "empty_like": "memory",
    "zeros": "memory",
    "kv_cache_update": "kvcache",
    "kv_cache": "kvcache",
    "mlp": "mlp",
}

# Noise words to filter out
_NOISE_WORDS = {
    "", "model", "language", "torch", "ops", "vllm", "ir", "default",
    "built", "function", "method", "operator", "c", "nn", "aten",
    "autograd", "pybind11", "get", "data", "attr", "sym", "size",
    "int", "type", "object", "of", "the",
}


def _norm_keywords(name: str) -> set[str]:
    """Extract normalized keywords from an op/module name."""
    name = name.lower().replace("-", "_").replace(".", "_").replace("/", "_")
    words = set(name.split("_"))
    # Also try camelCase splitting
    import re as _re
    camel = _re.findall(r"[a-z]+|[A-Z][a-z]*", name)
    words.update(w.lower() for w in camel)
    # Filter noise
    words.difference_update(_NOISE_WORDS)
    # Remove pure numbers
    words = {w for w in words if not w.isdigit()}
    return words


def _fx_keywords(op: OpNode) -> set[str]:
    """Get semantic keywords for an FX op."""
    kw = set()
    name = op.op_name
    # Apply alias mapping — map raw keywords to canonical form
    raw = _norm_keywords(name)
    for word in raw:
        if word in _FX_KEYWORD_ALIASES:
            kw.add(_FX_KEYWORD_ALIASES[word])
        else:
            kw.add(word)
    # Try substring aliases
    for alias, canonical in _FX_KEYWORD_ALIASES.items():
        if alias in name.lower().replace(".", "_"):
            kw.add(canonical)
    # From metadata subgraph
    sub = op.metadata.get("subgraph", "")
    if sub and sub != "other":
        kw.add(sub)
    return kw


def _eager_keywords(op: OpNode) -> set[str]:
    """Get semantic keywords for an eager (module) op."""
    kw = set()
    # Module type: "GemmaRMSNorm" → {gemma, rms, norm} → canonical
    mod_type = op.metadata.get("module_type", "")
    mod_raw = _norm_keywords(mod_type)
    for word in mod_raw:
        if word in _FX_KEYWORD_ALIASES:
            kw.add(_FX_KEYWORD_ALIASES[word])
        else:
            kw.add(word)
    # Module name
    name_raw = _norm_keywords(op.op_name)
    for word in name_raw:
        if word in _FX_KEYWORD_ALIASES:
            kw.add(_FX_KEYWORD_ALIASES[word])
        else:
            kw.add(word)
    # Aliases from full strings
    for alias, canonical in _FX_KEYWORD_ALIASES.items():
        combined = (mod_type + op.op_name).lower().replace(".", "_")
        if alias in combined:
            kw.add(canonical)
    return kw


def compute_match_score(fx_op: OpNode, eager_op: OpNode) -> float:
    score = 0.0

    # Exact name match (unlikely across granularities)
    if fx_op.op_name == eager_op.op_name:
        score += 50
    elif fx_op.op_name in eager_op.op_name or eager_op.op_name in fx_op.op_name:
        score += 20

    # Keyword overlap — key for cross-granularity matching
    fx_kw = _fx_keywords(fx_op)
    eager_kw = _eager_keywords(eager_op)
    overlap = fx_kw & eager_kw
    if overlap:
        score += len(overlap) * 12  # up to ~48 for 4 overlapping keywords

    # Input count match
    if len(fx_op.inputs) == len(eager_op.inputs):
        score += 10

    # Shape rank match
    for fx_in, e_in in zip(fx_op.inputs, eager_op.inputs):
        if len(fx_in.shape) == len(e_in.shape):
            score += 5
        if fx_in.dtype == e_in.dtype:
            score += 3

    return score


def align_ops(
    fx_ops: list[OpNode],
    eager_ops: list[OpNode],
) -> list[DiffEntry]:
    entries: list[DiffEntry] = []
    eager_matched: set[int] = set()

    for fx_op in fx_ops:
        best: OpNode | None = None
        best_score = 0.0
        best_idx: int | None = None

        for idx, e_op in enumerate(eager_ops):
            if idx in eager_matched:
                continue
            score = compute_match_score(fx_op, e_op)
            if score > best_score:
                best_score = score
                best = e_op
                best_idx = idx

        if best and best_score >= MATCH_THRESHOLD:
            entries.append(DiffEntry(
                type="match", fx_op=fx_op, eager_op=best, score=best_score,
            ))
            eager_matched.add(best_idx)
        elif best and best_score >= MATCH_THRESHOLD // 2:
            entries.append(DiffEntry(
                type="shape_mismatch", fx_op=fx_op, eager_op=best,
                score=best_score,
            ))
            eager_matched.add(best_idx)
        else:
            entries.append(DiffEntry(type="compile_only", fx_op=fx_op))

    for idx, e_op in enumerate(eager_ops):
        if idx not in eager_matched:
            entries.append(DiffEntry(type="eager_only", eager_op=e_op))

    return entries


def diff(dump_fx: GraphDump, dump_eager: GraphDump) -> DiffReport:
    # Flatten all ops — Phase 1 and Phase 2 may have different layer groupings
    all_fx_ops: list[OpNode] = []
    for l in dump_fx.layers:
        all_fx_ops.extend(l.ops)
    all_eager_ops: list[OpNode] = []
    for l in dump_eager.layers:
        all_eager_ops.extend(l.ops)

    report = DiffReport(
        model_name=dump_fx.model_name,
        source_fx="",
        source_eager="",
        summary=DiffSummary(),
    )

    entries = align_ops(all_fx_ops, all_eager_ops)
    # Group entries by eager op's layer for readable output
    layer_entries: dict[str, list[DiffEntry]] = defaultdict(list)
    for e in entries:
        if e.eager_op:
            parts = e.eager_op.op_name.split(".")
            for i, p in enumerate(parts):
                if p == "layers" and i + 1 < len(parts):
                    layer_entries[f"layer_{parts[i + 1]}"].append(e)
                    break
            else:
                layer_entries["other"].append(e)
        else:
            layer_entries["global"].append(e)

    for key in sorted(layer_entries.keys()):
        report.layers.append(LayerDiff(layer_name=key, entries=layer_entries[key]))

    report.summary.total_fx_ops = len(all_fx_ops)
    report.summary.total_eager_ops = len(all_eager_ops)
    report.summary.matched = sum(1 for e in entries if e.type == "match")
    report.summary.shape_mismatches = sum(1 for e in entries if e.type == "shape_mismatch")
    report.summary.compile_only = sum(1 for e in entries if e.type == "compile_only")
    report.summary.eager_only = sum(1 for e in entries if e.type == "eager_only")
    return report


def _format_report(report: DiffReport) -> str:
    from collections import Counter

    s = report.summary
    total = max(s.matched + s.compile_only + s.eager_only, 1)
    pct = s.matched * 100 // total
    lines = []
    lines.append("┌" + "─" * 70 + "┐")
    lines.append(f"│{'Diff Report: ' + report.model_name:^70}│")
    lines.append("├" + "─" * 70 + "┤")
    lines.append(
        f"│  FX (Inductor graph):  {s.total_fx_ops:5d} ops  "
        f"Eager (modules):  {s.total_eager_ops:5d} calls      │"
    )
    lines.append(
        f"│  Semantically matched: {s.matched:4d} ({pct:d}%)  "
        f"FX-only: {s.compile_only:4d}  Mod-only: {s.eager_only:4d}     │"
    )
    lines.append("├" + "─" * 70 + "┤")

    # Show matched pairs (sample)
    all_matches = []
    for layer in report.layers:
        for entry in layer.entries:
            if entry.type == "match":
                all_matches.append(entry)

    lines.append(f"│  Matched pairs ({len(all_matches)}):                                  │")
    for entry in all_matches[:15]:
        fx_short = entry.fx_op.op_name.split(".")[-1][:30]
        eager_short = entry.eager_op.op_name.split(".")[-1][:30]
        mt = entry.eager_op.metadata.get("module_type", "")[:15]
        lines.append(
            f"│    ✓ {fx_short:<28} ⇄ {eager_short:<18} [{mt:<15}] │"
        )
    if len(all_matches) > 15:
        lines.append(f"│    ... ({len(all_matches) - 15} more matches)                       │")

    # Show top FX-only keyword groups
    lines.append("├" + "─" * 70 + "┤")
    lines.append("│  FX-only (compile decomposes/transforms):                          │")
    fx_kw_counts: Counter = Counter()
    for layer in report.layers:
        for entry in layer.entries:
            if entry.type == "compile_only":
                for k in _fx_keywords(entry.fx_op):
                    if k not in ("other",):
                        fx_kw_counts[k] += 1
    for kw, cnt in fx_kw_counts.most_common(10):
        lines.append(f"│    {kw:<18s} {cnt:4d}x                                          │")

    # Show module-only types
    lines.append("├" + "─" * 70 + "┤")
    lines.append("│  Module calls (no matching FX op):                                 │")
    mod_counts: Counter = Counter()
    for layer in report.layers:
        for entry in layer.entries:
            if entry.type == "eager_only":
                mt = entry.eager_op.metadata.get("module_type", "?")
                mod_counts[mt] += 1
    for mt, cnt in mod_counts.most_common(8):
        lines.append(f"│    {mt:<18s} {cnt:4d}x                                          │")

    lines.append("└" + "─" * 70 + "┘")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Diff FX graph dump vs eager trace dump",
    )
    parser.add_argument("fx_dump", type=str, help="Path to fx_graph_dump.json")
    parser.add_argument("eager_dump", type=str, help="Path to eager_trace_dump.json")
    parser.add_argument(
        "--output", "-o", type=str, default=None, help="Output JSON path",
    )
    parser.add_argument("--threshold", type=int, default=None)
    args = parser.parse_args()

    global MATCH_THRESHOLD
    if args.threshold is not None:
        MATCH_THRESHOLD = args.threshold

    fx = GraphDump.from_json(args.fx_dump)
    eager = GraphDump.from_json(args.eager_dump)
    report = diff(fx, eager)
    report.source_fx = args.fx_dump
    report.source_eager = args.eager_dump

    print(_format_report(report))

    if args.output:
        data = {
            "model_name": report.model_name,
            "summary": {
                "total_fx_ops": report.summary.total_fx_ops,
                "total_eager_ops": report.summary.total_eager_ops,
                "matched": report.summary.matched,
                "fused": report.summary.fused,
                "decomposed": report.summary.decomposed,
                "compile_only": report.summary.compile_only,
                "eager_only": report.summary.eager_only,
                "shape_mismatches": report.summary.shape_mismatches,
            },
        }
        with open(args.output, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Report saved to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
