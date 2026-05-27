#!/usr/bin/env python3
"""Compile Qwen3.6-27B 64K with a scoped FP8 quantization ablation.

This script starts from the validated 64K hybrid/chunked-prefill baseline and
changes only quantization scope. Three tiers are supported via
``--quantization-tier``:

- ``weight_only_mlp`` (default): MLP linear weights are converted to FP8 e4m3
  per-channel; attention, DeltaNet, norms, embeddings, lm_head, KV cache, and
  activations all stay BF16. ``activation_quantization_type`` stays ``None`` so
  FP8 weights are dequantized to BF16 at compute time. This is a memory-
  footprint experiment, not an FP8 compute experiment.

- ``dynamic_mlp``: same checkpoint as ``weight_only_mlp``, but
  ``activation_quantization_type='dynamic'`` so MLP matmul becomes real
  FP8 x FP8 via the native ``convert()`` quantize path. Attention, DeltaNet,
  norms, KV cache stay BF16.

- ``dynamic_mlp_attn``: ``dynamic_mlp`` plus removes ``self_attn`` from
  ``modules_to_not_convert`` so standard self-attention QKV/O on the
  ``full_attention`` layers also goes FP8. DeltaNet (``linear_attn``), norms,
  KV cache stay BF16. The checkpoint is built so the standard self-attention
  weights are pre-quantized to FP8 alongside the MLP weights.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
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


QUANT_TIERS = ("weight_only_mlp", "dynamic_mlp", "dynamic_mlp_attn")


def _modules_to_not_convert(num_layers: int, tier: str) -> list[str]:
    """Build the FP8-exclude list for the requested quantization tier.

    All tiers exclude embeddings, lm_head, norms, rotary cache, DeltaNet
    (``linear_attn``), and per-layer norms. ``self_attn`` is excluded for the
    MLP-only tiers and *included* (i.e. allowed to be FP8-converted) for
    ``dynamic_mlp_attn``.
    """
    if tier not in QUANT_TIERS:
        raise ValueError(f"unknown quantization tier: {tier}")
    modules = [
        "embed_tokens",
        "model.embed_tokens",
        "lm_head",
        "norm",
        "model.norm",
        "rotary_emb",
        "model.rotary_emb",
    ]
    for layer_idx in range(num_layers):
        for prefix in ("layers", "model.layers"):
            modules.extend(
                [
                    f"{prefix}.{layer_idx}.linear_attn",
                    f"{prefix}.{layer_idx}.input_layernorm",
                    f"{prefix}.{layer_idx}.post_attention_layernorm",
                ]
            )
            if tier != "dynamic_mlp_attn":
                modules.append(f"{prefix}.{layer_idx}.self_attn")
    return modules


def _quantized_checkpoint_ready(path: Path) -> bool:
    if path.is_file():
        return True
    if path.is_dir():
        return any(path.iterdir())
    return False


def _is_mlp_weight(name: str) -> bool:
    parts = name.split(".")
    return (
        len(parts) >= 4
        and parts[-3] == "mlp"
        and parts[-2] in {"gate_proj", "up_proj", "down_proj"}
        and parts[-1] == "weight"
    )


def _is_self_attn_qkvo_weight(name: str) -> bool:
    """Match standard self-attention projections on full_attention layers.

    Names look like ``layers.<idx>.self_attn.q_proj.weight`` etc. DeltaNet
    layers use ``linear_attn`` (different prefix) and are excluded.
    """
    parts = name.split(".")
    return (
        len(parts) >= 4
        and parts[-3] == "self_attn"
        and parts[-2] in {"q_proj", "k_proj", "v_proj", "o_proj"}
        and parts[-1] == "weight"
    )


def _should_quantize_weight(name: str, tier: str) -> bool:
    if tier == "dynamic_mlp_attn":
        return _is_mlp_weight(name) or _is_self_attn_qkvo_weight(name)
    return _is_mlp_weight(name)


def _scale_name(weight_name: str) -> str:
    return weight_name[: -len(".weight")] + ".weight_scale"


def _clear_quantized_checkpoint_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.name.endswith(".safetensors") or child.name.endswith(".json"):
            child.unlink()


def _save_fp8_state_dict(
    model_path: Path, output_path: Path, tier: str = "weight_only_mlp"
) -> None:
    """Create a sharded FP8 checkpoint directly from HF safetensors.

    Loading the HF architecture requires a newer Transformers than the Neuron
    venv uses internally. For this scoped ablation, we do not need model
    execution: the checkpoint transform is a direct tensor rewrite. ``tier``
    controls which weights are converted to FP8; everything else is copied
    through unchanged.
    """
    from safetensors.torch import load_file, save_file  # noqa: WPS433
    from neuronx_distributed.quantization.quantization_utils import (  # noqa: WPS433
        quantize_fp8_per_channel,
    )

    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open() as f:
            source_index = json.load(f)
        source_weight_map = source_index["weight_map"]
        filenames = sorted(set(source_weight_map.values()))
    elif (model_path / "model.safetensors").exists():
        source_weight_map = None
        filenames = ["model.safetensors"]
    else:
        raise FileNotFoundError(f"No safetensors checkpoint found in {model_path}")

    _clear_quantized_checkpoint_dir(output_path)
    output_weight_map: dict[str, str] = {}
    total_size = 0
    quantized_count = 0

    for filename in filenames:
        shard = load_file(str(model_path / filename))
        output_shard = {}
        for name, tensor in shard.items():
            if _should_quantize_weight(name, tier):
                weight, scale = quantize_fp8_per_channel(
                    tensor,
                    torch.float8_e4m3fn,
                    channel_axis=0,
                )
                output_shard[name] = weight
                output_shard[_scale_name(name)] = scale
                output_weight_map[_scale_name(name)] = filename
                total_size += weight.numel() * weight.element_size()
                total_size += scale.numel() * scale.element_size()
                quantized_count += 1
            else:
                output_shard[name] = tensor
                total_size += tensor.numel() * tensor.element_size()
            output_weight_map[name] = filename

        save_file(output_shard, str(output_path / filename), metadata={"format": "pt"})
        del shard
        del output_shard
        gc.collect()

    if source_weight_map is not None:
        with (output_path / "model.safetensors.index.json").open("w") as f:
            json.dump(
                {
                    "metadata": {"total_size": total_size},
                    "weight_map": output_weight_map,
                },
                f,
                indent=2,
                sort_keys=True,
            )

    print(
        f"MANUAL_FP8_QUANT_COUNT tier={tier} count={quantized_count}",
        flush=True,
    )


def _parse_buckets(args: argparse.Namespace) -> list[int]:
    if args.context_encoding_buckets:
        raw = [int(x) for x in args.context_encoding_buckets.split(",") if x.strip()]
        if not raw:
            raise ValueError("--context-encoding-buckets must list at least one int")
        return sorted(set(raw))
    return [args.cte_bucket]


def _patch_scale_dequantize_for_3d_activations() -> None:
    """Workaround for NxDI's scale_dequantize broadcasting on 3-D activations.

    Why: NxDI's `scale_dequantize(tensor, scale, ...)` always does
    `scale.unsqueeze(len(scale.shape)-1)` before the in-place multiply. That
    is correct for the 2-D weight scale `[1, out]` -> `[1, 1, out]` against a
    3-D output `(B, S, out)`. But on the *input* side of the dynamic FP8
    path, `quantize_fp8_per_channel(x, channel_axis=1)` over a 3-D input
    `(B, S, H)` produces a per-S scale shaped `[1, S, 1]` (3-D). Unsqueezing
    that to `[1, S, 1, 1]` and broadcasting onto the 3-D matmul output fails
    neuronx-cc's rank check:
        RuntimeError: Check failed: input_sizes.size() <= output_sizes.size()

    Fix: only unsqueeze when the scale rank is strictly less than the tensor
    rank. This keeps the original 2-D weight-scale path untouched.
    """
    from neuronx_distributed.quantization import dequantize as _dq
    from neuronx_distributed.quantization import quantization_layers as _ql

    def _patched(tensor, scale, upcast_dtype):  # noqa: WPS430
        upcast_tensor = tensor.to(torch.float32)
        if scale.ndim < tensor.ndim:
            scale = scale.unsqueeze(scale.ndim - 1)
        upcast_tensor = upcast_tensor * scale
        return upcast_tensor.to(upcast_dtype)

    _dq.scale_dequantize = _patched
    _ql.scale_dequantize = _patched


def _build_config(args: argparse.Namespace):
    from neuronx_distributed_inference.models.config import (  # noqa: WPS433
        NeuronConfig,
        OnDeviceSamplingConfig,
    )
    from src.modeling_qwen35 import Qwen35InferenceConfig  # noqa: WPS433

    model_path = Path(args.model_path).expanduser().resolve()
    config_dict = _load_text_config(model_path)
    num_layers = int(config_dict["num_hidden_layers"])
    tier = args.quantization_tier
    modules_to_not_convert = _modules_to_not_convert(num_layers, tier)

    # Tier semantics:
    #   weight_only_mlp -> FP8 weights, BF16 compute (dequant pre-matmul)
    #   dynamic_mlp     -> FP8 weights + DYNAMIC FP8 activations on MLP
    #   dynamic_mlp_attn-> same dynamic, also covers self_attn QKV/O
    activation_quantization_type = (
        "dynamic" if tier in ("dynamic_mlp", "dynamic_mlp_attn") else None
    )

    buckets = _parse_buckets(args)
    max_ctx = max(buckets)
    enable_bucketing = len(buckets) > 1

    neuron_config = NeuronConfig(
        tp_degree=args.tp_degree,
        batch_size=args.batch_size,
        ctx_batch_size=args.batch_size,
        tkg_batch_size=args.batch_size,
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
        quantized=True,
        quantized_checkpoints_path=str(
            Path(args.quantized_checkpoints_path).expanduser().resolve()
        ),
        quantization_type="per_channel_symmetric",
        quantization_dtype="f8e4m3",
        modules_to_not_convert=modules_to_not_convert,
        kv_cache_quant=False,
        quantized_mlp_kernel_enabled=False,
        activation_quantization_type=activation_quantization_type,
    )

    config_dict.setdefault("use_hybrid_cache_manager", True)
    config_dict.setdefault("use_qwen_hybrid_chunked_prefill", True)
    config_dict.setdefault("use_qwen_hybrid_chunked_prefill_nki", True)

    inf_config = Qwen35InferenceConfig(neuron_config=neuron_config, **config_dict)
    return inf_config, modules_to_not_convert


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--compiled-path", required=True)
    parser.add_argument("--quantized-checkpoints-path", required=True)
    parser.add_argument("--seq-len", type=int, default=65536)
    parser.add_argument("--cte-bucket", type=int, default=512)
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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help=(
            "Compiled (max) batch size. Sets batch_size, ctx_batch_size, and "
            "tkg_batch_size to the same value because Qwen3.6's "
            "perform_qwen_chunked_prefill assumes ctx and tkg share the same "
            "batch dim (it indexes the full kv-cache during prefill). The "
            "vLLM server must be started with --max-num-seqs <= this value."
        ),
    )
    parser.add_argument("--force-quantize", action="store_true")
    parser.add_argument("--quantize-only", action="store_true")
    parser.add_argument("--load-after-compile", action="store_true")
    parser.add_argument(
        "--quantization-tier",
        choices=QUANT_TIERS,
        default="weight_only_mlp",
        help=(
            "weight_only_mlp (default): FP8 weights, BF16 compute. "
            "dynamic_mlp: FP8 weights + DYNAMIC FP8 activations on MLP. "
            "dynamic_mlp_attn: also covers standard self-attention QKV/O."
        ),
    )
    args = parser.parse_args()

    repo = _repo_root(args.repo_root)
    contrib_model_dir = repo / "contrib" / "models" / "Qwen3.6-27B"
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(contrib_model_dir))

    from src.modeling_qwen35 import NeuronQwen35ForCausalLM  # noqa: WPS433

    model_path = Path(args.model_path).expanduser().resolve()
    compiled_path = Path(args.compiled_path).expanduser().resolve()
    quantized_path = Path(args.quantized_checkpoints_path).expanduser().resolve()

    if args.quantization_tier in ("dynamic_mlp", "dynamic_mlp_attn"):
        _patch_scale_dequantize_for_3d_activations()
        print("PATCH_SCALE_DEQUANTIZE_APPLIED", flush=True)

    inf_config, modules_to_not_convert = _build_config(args)

    print(f"FP8_TIER {args.quantization_tier}", flush=True)
    print("MODEL_PATH", str(model_path), flush=True)
    print("COMPILED_PATH", str(compiled_path), flush=True)
    print("QUANTIZED_CHECKPOINTS_PATH", str(quantized_path), flush=True)
    print("MODULES_TO_NOT_CONVERT_COUNT", len(modules_to_not_convert), flush=True)
    buckets = _parse_buckets(args)
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

    if args.force_quantize or not _quantized_checkpoint_ready(quantized_path):
        print(f"QUANTIZE_START tier={args.quantization_tier}", flush=True)
        _save_fp8_state_dict(model_path, quantized_path, tier=args.quantization_tier)
        print("QUANTIZE_DONE", flush=True)
    else:
        print("QUANTIZE_SKIP existing checkpoint found", flush=True)

    if args.quantize_only:
        return 0

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
