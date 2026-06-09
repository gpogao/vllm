# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.fx as fx

from vllm.compilation.passes.dump_graph import _resolve_layer_name, _is_compute_op


class TestResolveLayer:
    def test_model_layer_pattern(self):
        assert _resolve_layer_name("model_layers_0_self_attn_q_proj") == "model.layers.0"
        assert _resolve_layer_name("model_layers_41_mlp_down_proj") == "model.layers.41"

    def test_embedding(self):
        assert _resolve_layer_name("embed_tokens") == "embedding"
        assert _resolve_layer_name("tok_embeddings") == "embedding"

    def test_lm_head(self):
        assert _resolve_layer_name("lm_head") == "lm_head"
        assert _resolve_layer_name("model_head") == "lm_head"

    def test_other(self):
        assert _resolve_layer_name("unknown_tensor") == "other"


class TestIsComputeOp:
    def test_placeholder_not_compute(self):
        graph = fx.Graph()
        node = graph.placeholder("input")
        assert not _is_compute_op(node)

    def test_output_not_compute(self):
        graph = fx.Graph()
        node = graph.output(())
        assert not _is_compute_op(node)

    def test_getattr_not_compute(self):
        graph = fx.Graph()
        node = graph.get_attr("weight")
        assert not _is_compute_op(node)

    def test_call_function_is_compute(self):
        graph = fx.Graph()
        node = graph.create_node("call_function", torch.add, args=())
        assert _is_compute_op(node)

    def test_call_module_is_compute(self):
        graph = fx.Graph()
        node = graph.create_node("call_module", "linear_1", args=())
        assert _is_compute_op(node)
