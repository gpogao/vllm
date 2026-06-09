# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from vllm.profiler.graph_schema import OpNode, TensorInfo
from vllm.profiler.graph_diff import (
    compute_match_score,
    DiffEntry,
    align_ops,
)


def make_op(name: str, inputs: list[list[int]], dtype="torch.float16") -> OpNode:
    return OpNode(
        id=0, op_type="fx", op_name=name, display_name=name,
        inputs=[TensorInfo(name=f"in_{i}", shape=s, dtype=dtype, device="cuda:0")
                for i, s in enumerate(inputs)],
        outputs=[],
        metadata={},
    )


class TestComputeMatchScore:
    def test_exact_match(self):
        a = make_op("aten::addmm", [[1, 2560], [10240, 2560]])
        b = make_op("aten::addmm", [[1, 2560], [10240, 2560]])
        score = compute_match_score(a, b)
        assert score >= 70  # name match 50 + input count 20

    def test_partial_name_match(self):
        a = make_op("aten::addmm", [[1, 2560]])
        b = make_op("aten::addmm.default", [[1, 2560]])
        score = compute_match_score(a, b)
        assert score >= 20  # partial name match

    def test_no_match(self):
        a = make_op("aten::addmm", [[1, 2560]])
        b = make_op("aten::relu", [[1, 10240]])
        score = compute_match_score(a, b)
        assert score < 50  # name mismatch but same input count + rank + dtype


class TestAlignOps:
    def test_one_to_one_match(self):
        fx = [make_op("vllm_ir.rms_norm", [[1, 2560]])]
        eager = [make_op("aten::rms_norm", [[1, 2560]])]
        results = align_ops(fx, eager)
        # Different names → compile_only + eager_only
        assert len(results) == 2

    def test_no_fx_ops_all_eager_only(self):
        fx = []
        eager = [make_op("aten::addmm", [[1, 2560], [10240, 2560]])]
        results = align_ops(fx, eager)
        assert all(r.type == "eager_only" for r in results)

    def test_no_eager_ops_all_compile_only(self):
        fx = [make_op("vllm_ir.relu", [[1, 2560]])]
        eager = []
        results = align_ops(fx, eager)
        assert all(r.type == "compile_only" for r in results)
