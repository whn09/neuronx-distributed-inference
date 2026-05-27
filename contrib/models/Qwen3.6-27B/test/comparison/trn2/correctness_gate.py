#!/usr/bin/env python3
"""Offline correctness gate for a freshly compiled Qwen3.6-27B artifact.

Loads the compiled NEFFs, runs greedy generation on two short prompts, and
exits non-zero if the output looks broken. Used by drive_b1.sh between the
compile step and the bench sweep so we never archive numbers from an artifact
that produces gibberish.

Pass criteria (all must hold for *both* prompts):
  - generation produced >= 8 new tokens
  - decoded text is not a single repeated word (no run of >=5 identical tokens
    in a row)
  - decoded text contains at least 3 distinct alphanumeric word-chars chunks

This is intentionally generic: we don't pin a specific expected string,
because greedy-deterministic decode can still drift across precisions. The
goal is to catch the obvious failure mode "all tokens are the same" or "EOS
on token 1", which is what numerically-broken FP8 paths tend to produce.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path


PROMPTS = [
    "The capital of France is",
    "Hello, I am a language model",
]

MIN_NEW_TOKENS = 8
MAX_REPEAT_RUN = 5
MIN_DISTINCT_WORDS = 3


def _looks_repetitive(token_ids: list[int]) -> bool:
    if len(token_ids) < MAX_REPEAT_RUN:
        return False
    for i in range(len(token_ids) - MAX_REPEAT_RUN + 1):
        window = token_ids[i : i + MAX_REPEAT_RUN]
        if len(set(window)) == 1:
            return True
    return False


def _distinct_word_count(text: str) -> int:
    words = re.findall(r"[A-Za-z0-9]+", text)
    return len(set(w.lower() for w in words))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--compiled-path", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--repo", default=None)
    args = parser.parse_args()

    repo = Path(args.repo).resolve() if args.repo else Path(__file__).resolve().parents[5]
    contrib_model_dir = repo / "contrib" / "models" / "Qwen3.6-27B"
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(contrib_model_dir))

    import transformers
    from transformers import AutoTokenizer, GenerationConfig

    from neuronx_distributed_inference.utils.hf_adapter import (
        HuggingFaceGenerationAdapter,
    )
    from src.modeling_qwen35 import NeuronQwen35ForCausalLM

    print(f"GATE_LOAD compiled_path={args.compiled_path}", flush=True)
    model = NeuronQwen35ForCausalLM(args.compiled_path)
    model.load(args.compiled_path)
    print("GATE_LOAD_DONE", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    gen_cfg = GenerationConfig(
        do_sample=False,
        top_k=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        transformers_version=transformers.__version__,
    )
    gen_model = HuggingFaceGenerationAdapter(model)
    gen_model.generation_config.transformers_version = transformers.__version__

    failures: list[str] = []
    for prompt in PROMPTS:
        inputs = tokenizer(prompt, padding=True, return_tensors="pt")
        in_len = int(inputs.input_ids.shape[-1])
        out = gen_model.generate(
            inputs.input_ids,
            generation_config=gen_cfg,
            attention_mask=inputs.attention_mask,
            max_new_tokens=args.max_new_tokens,
        )
        out_ids = out[0].tolist()
        new_ids = out_ids[in_len:]
        text = tokenizer.decode(out_ids, skip_special_tokens=True)
        new_text = text[len(prompt) :].strip()
        distinct = _distinct_word_count(new_text)
        repetitive = _looks_repetitive(new_ids)
        ok = (
            len(new_ids) >= MIN_NEW_TOKENS
            and not repetitive
            and distinct >= MIN_DISTINCT_WORDS
        )
        status = "PASS" if ok else "FAIL"
        # Truncate for log readability.
        snippet = new_text.replace("\n", " ")[:120]
        print(
            f"GATE {status} prompt={prompt!r} new_tokens={len(new_ids)} "
            f"distinct={distinct} repetitive={repetitive} text={snippet!r}",
            flush=True,
        )
        if not ok:
            failures.append(prompt)

    if failures:
        print(f"GATE_FAIL prompts={failures}", flush=True)
        return 1
    print("GATE_PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
