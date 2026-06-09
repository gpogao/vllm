# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import pytest
from vllm.profiler.graph_schema import (
    TensorInfo,
    OpNode,
    LayerDump,
    GraphDump,
    _get_model_hints,
    hash_ops,
    deduplicate_layers,
)


class TestTensorInfo:
    def test_serialize_roundtrip(self):
        t = TensorInfo(name="x", shape=[1, 2560], dtype="torch.bfloat16", device="cuda:0")
        d = t.__dict__
        assert d == {"name": "x", "shape": [1, 2560], "dtype": "torch.bfloat16", "device": "cuda:0"}

    def test_symbolic_shape(self):
        t = TensorInfo(name="x", shape=["s0", 2560], dtype="torch.float16", device="cuda:0")
        assert t.shape == ["s0", 2560]


class TestOpNode:
    def test_minimal(self):
        op = OpNode(
            id=0, op_type="fx", op_name="vllm_ir.rms_norm",
            display_name="RMSNorm(2560)",
            inputs=[TensorInfo(name="x", shape=[1, 2560], dtype="torch.bfloat16", device="cuda:0")],
            outputs=[TensorInfo(name="output", shape=[1, 2560], dtype="torch.bfloat16", device="cuda:0")],
            metadata={},
        )
        assert op.id == 0
        assert op.op_name == "vllm_ir.rms_norm"


class TestGraphDump:
    def test_to_from_json_roundtrip(self, tmp_path):
        layer = LayerDump(name="model.layers.{0}", count=1, representative_index=0, ops=[])
        dump = GraphDump(
            version="1.1", mode="fx_compile", model_name="test",
            model_config={}, num_layers=1, layers=[layer],
        )
        path = tmp_path / "test_dump.json"
        dump.to_json(str(path))
        loaded = GraphDump.from_json(str(path))
        assert loaded.model_name == "test"
        assert len(loaded.layers) == 1


class TestDeduplicateLayers:
    def test_identical_layers_merge(self):
        ops = [OpNode(id=0, op_type="fx", op_name="op", display_name="op",
                       inputs=[], outputs=[], metadata={})]
        layers = {
            "model.layers.0": ops,
            "model.layers.1": ops,
            "model.layers.2": ops,
        }
        result = deduplicate_layers(layers)
        assert len(result) == 1
        assert result[0].count == 3
        assert result[0].representative_index == 0

    def test_different_layers_split(self):
        ops_a = [OpNode(id=0, op_type="fx", op_name="op_a", display_name="op_a",
                         inputs=[], outputs=[], metadata={})]
        ops_b = [OpNode(id=0, op_type="fx", op_name="op_b", display_name="op_b",
                         inputs=[], outputs=[], metadata={})]
        layers = {
            "model.layers.0": ops_a,
            "model.layers.1": ops_b,
        }
        result = deduplicate_layers(layers)
        assert len(result) == 2

    def test_with_layer_types_annotation(self):
        ops = [OpNode(id=0, op_type="fx", op_name="op", display_name="op",
                       inputs=[], outputs=[], metadata={})]
        layers = {f"model.layers.{i}": ops for i in range(5)}
        config_hints = {"layer_types": ["sliding_attention"] * 5}
        result = deduplicate_layers(layers, config_hints)
        assert "[sliding_attention]" in result[0].name
        assert result[0].count == 5


class TestHashOps:
    def test_identical_ops_same_hash(self):
        ops_a = [OpNode(id=0, op_type="fx", op_name="linear",
                         display_name="l",
                         inputs=[TensorInfo(name="x", shape=[1, 10], dtype="fp16", device="cuda:0")],
                         outputs=[], metadata={})]
        ops_b = [OpNode(id=0, op_type="fx", op_name="linear",
                         display_name="l",
                         inputs=[TensorInfo(name="x", shape=[1, 10], dtype="fp16", device="cuda:0")],
                         outputs=[], metadata={})]
        assert hash_ops(ops_a) == hash_ops(ops_b)

    def test_different_ops_different_hash(self):
        ops_a = [OpNode(id=0, op_type="fx", op_name="linear",
                         display_name="l",
                         inputs=[TensorInfo(name="x", shape=[1, 10], dtype="fp16", device="cuda:0")],
                         outputs=[], metadata={})]
        ops_b = [OpNode(id=0, op_type="fx", op_name="rms_norm",
                         display_name="r",
                         inputs=[TensorInfo(name="x", shape=[1, 20], dtype="fp16", device="cuda:0")],
                         outputs=[], metadata={})]
        assert hash_ops(ops_a) != hash_ops(ops_b)


class TestGetModelHints:
    def test_no_config_returns_empty(self):
        # Monkeypatch at the definition site so the import inside _get_model_hints
        # returns our stub.
        import vllm.config
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(vllm.config, "get_current_vllm_config", lambda: None)
            hints = _get_model_hints()
            assert hints == {}
