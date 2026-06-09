# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Phase 1 entry point: dump FX computation graph using vLLM offline API."""

import argparse
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Dump vLLM FX computation graph")
    parser.add_argument(
        "--model", type=str, required=True,
        help="Model path or HuggingFace name",
    )
    parser.add_argument(
        "--prompt", type=str, default="Hello",
        help="Prompt to trigger compilation",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=1,
        help="Max tokens to generate",
    )
    args = parser.parse_args()

    # Set env var if not already set
    if not os.environ.get("VLLM_DUMP_FX_GRAPH"):
        os.environ["VLLM_DUMP_FX_GRAPH"] = "1"

    from vllm import LLM
    from vllm.sampling_params import SamplingParams

    print(f"Loading model: {args.model}")
    llm = LLM(
        model=args.model,
        enforce_eager=False,         # compile mode
        language_model_only=True,    # only text inference
        gpu_memory_utilization=0.85,
        max_model_len=256,
        max_num_seqs=1,
    )

    print("Running generate to trigger compilation...")
    output = llm.generate(
        args.prompt,
        sampling_params=SamplingParams(
            max_tokens=args.max_tokens, temperature=0,
        ),
    )
    print(f"Generated: {output[0].outputs[0].text[:50]}...")

    # Check what got dumped
    from vllm.envs import VLLM_DUMP_DIR
    output_dir = VLLM_DUMP_DIR or "."
    for stage in ("before_split", "after_split", "after_lowering"):
        path = os.path.join(output_dir, f"fx_graph_dump__{stage}.json")
        if os.path.exists(path):
            size_kb = os.path.getsize(path) / 1024
            print(f"✓ {path} ({size_kb:.1f} KB)")
        else:
            print(f"✗ {path} (not found)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
