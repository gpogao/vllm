# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Phase 2 entry point: dump eager module trace via worker-side hooks."""

import argparse
import json
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Dump vLLM eager trace")
    parser.add_argument(
        "--model", type=str, required=True,
        help="Model path or HuggingFace name",
    )
    parser.add_argument(
        "--prompt", type=str, default="Hello",
        help="Prompt for inference",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=1,
        help="Max tokens to generate",
    )
    args = parser.parse_args()

    if not os.environ.get("VLLM_DUMP_EAGER_TRACE"):
        os.environ["VLLM_DUMP_EAGER_TRACE"] = "1"

    from vllm import LLM
    from vllm.sampling_params import SamplingParams
    from vllm.envs import VLLM_DUMP_DIR

    print(f"Loading model (eager mode): {args.model}")
    llm = LLM(
        model=args.model,
        enforce_eager=True,           # eager mode
        language_model_only=True,    # only text inference
        gpu_memory_utilization=0.85,
        max_model_len=256,
        max_num_seqs=1,
    )

    print("Running generate (hooks will dump from worker process)...")
    output = llm.generate(
        args.prompt,
        sampling_params=SamplingParams(
            max_tokens=args.max_tokens, temperature=0,
        ),
    )

    # Worker process writes the dump file during its first forward pass
    output_dir = VLLM_DUMP_DIR or "."
    output_path = os.path.join(output_dir, "eager_trace_dump.json")

    if os.path.exists(output_path):
        size_kb = os.path.getsize(output_path) / 1024
        with open(output_path) as f:
            d = json.load(f)
        total_ops = sum(len(l["ops"]) for l in d["layers"])
        print(
            f"✓ Dumped {d['model_name']}: {len(d['layers'])} layer groups, "
            f"{total_ops} ops total ({size_kb:.1f} KB)"
        )
        print(f"  Saved to: {output_path}")
    else:
        print(f"✗ Dump file not found: {output_path}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
