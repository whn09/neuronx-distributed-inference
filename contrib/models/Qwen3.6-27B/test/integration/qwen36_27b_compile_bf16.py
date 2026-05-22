#!/usr/bin/env python3
"""Compile Qwen3.6-27B BF16 (no quantization) for direct A100-vs-Trn2 comparison.

Mirror of qwen36_27b_compile_fp8.py with NeuronConfig.quantized=False and the
FP8-specific options removed. Supports multi-bucket context encoding so short
prompts don't pad to the largest bucket.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch


def _repo_root(path: str | None) -> Path:
    if path:
        return Path(path).expanduser().resolve()
    return Path(__file__).resolve().parents[5]


def _load_text_config(model_path: Path) -> dict:
    with (model_path / "config.json").open() as f:
        full_config = json.load(f)
    text_config = full_config.get("text_config", full_config)
    config_dict = dict(text_config)
    config_dict["pad_token_id"] = text_config.get("eos_token_id", 248044)
    if "rope_parameters" in text_config:
        config_dict["rope_theta"] = text_config["rope_parameters"].get(
            "rope_theta", 10000000
        )
    config_dict.setdefault("tie_word_embeddings", False)
    return config_dict


def _parse_buckets(args: argparse.Namespace) -> list[int]:
    if args.context_encoding_buckets:
        raw = [int(x) for x in args.context_encoding_buckets.split(",") if x.strip()]
        if not raw:
            raise ValueError("--context-encoding-buckets must list at least one int")
        return sorted(set(raw))
    return [args.cte_bucket]


def _build_config(args: argparse.Namespace):
    from neuronx_distributed_inference.models.config import (  # noqa: WPS433
        NeuronConfig,
        OnDeviceSamplingConfig,
    )
    from src.modeling_qwen35 import Qwen35InferenceConfig  # noqa: WPS433

    model_path = Path(args.model_path).expanduser().resolve()
    config_dict = _load_text_config(model_path)

    buckets = _parse_buckets(args)
    max_ctx = max(buckets)
    enable_bucketing = len(buckets) > 1

    neuron_config = NeuronConfig(
        tp_degree=args.tp_degree,
        batch_size=1,
        ctx_batch_size=1,
        tkg_batch_size=1,
        seq_len=args.seq_len,
        max_context_length=max_ctx,
        max_length=args.seq_len,
        context_encoding_buckets=buckets,
        torch_dtype=torch.bfloat16,
        on_device_sampling_config=OnDeviceSamplingConfig(
            do_sample=False,
            top_k=1,
            top_p=1.0,
            temperature=1.0,
        ),
        enable_bucketing=enable_bucketing,
        logical_nc_config=args.logical_nc_config,
        save_sharded_checkpoint=True,
    )

    config_dict.setdefault("use_hybrid_cache_manager", True)
    config_dict.setdefault("use_qwen_hybrid_chunked_prefill", True)
    config_dict.setdefault("use_qwen_hybrid_chunked_prefill_nki", True)

    inf_config = Qwen35InferenceConfig(neuron_config=neuron_config, **config_dict)
    return inf_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--compiled-path", required=True)
    parser.add_argument("--seq-len", type=int, default=9216)
    parser.add_argument("--cte-bucket", type=int, default=8192)
    parser.add_argument(
        "--context-encoding-buckets",
        default=None,
        help=(
            "Comma-separated list of CTE bucket sizes "
            "(e.g. '1024,2048,4096,8192'). When set, --cte-bucket is ignored "
            "and bucketing is enabled."
        ),
    )
    parser.add_argument("--tp-degree", type=int, default=4)
    parser.add_argument("--logical-nc-config", type=int, default=2)
    parser.add_argument("--load-after-compile", action="store_true")
    args = parser.parse_args()

    repo = _repo_root(args.repo_root)
    contrib_model_dir = repo / "contrib" / "models" / "Qwen3.6-27B"
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(contrib_model_dir))

    from src.modeling_qwen35 import NeuronQwen35ForCausalLM  # noqa: WPS433

    model_path = Path(args.model_path).expanduser().resolve()
    compiled_path = Path(args.compiled_path).expanduser().resolve()

    inf_config = _build_config(args)

    buckets = _parse_buckets(args)
    print("BF16_MODE", flush=True)
    print("MODEL_PATH", str(model_path), flush=True)
    print("COMPILED_PATH", str(compiled_path), flush=True)
    print(
        "CONTEXT_TRACE_SHAPE",
        json.dumps(
            {
                "seq_len": args.seq_len,
                "max_context_length": max(buckets),
                "context_encoding_buckets": buckets,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    print("COMPILE_START", flush=True)
    model = NeuronQwen35ForCausalLM(str(model_path), inf_config)
    model.compile(str(compiled_path))
    del model
    gc.collect()
    print("COMPILE_DONE", flush=True)

    if args.load_after_compile:
        model = NeuronQwen35ForCausalLM(str(compiled_path))
        model.load(str(compiled_path))
        print("LOAD_AFTER_COMPILE_OK", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
