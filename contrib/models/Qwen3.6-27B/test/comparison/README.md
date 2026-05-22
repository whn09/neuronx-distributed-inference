# Qwen3.6-27B GPU vs Trn2 comparison harness

Apples-to-apples benchmark scripts for comparing the contrib Qwen3.6-27B model on
A100 (and other GPUs) against the same model on Trn2. Both sides use vLLM's
standard `vllm bench serve` client so the request methodology, sampling, and
metrics are identical.

```
test/comparison/
├── gpu/    # vLLM CUDA backend (FP8 + BF16) + bench harness
└── trn2/   # vLLM Neuron backend (FP8 + BF16) + bench harness
```

The two sides share the same prompt distribution (`--dataset-name random
--random-range-ratio`) and the same sweep grids:

* **Prefill sweep**: ISL ∈ {8K, 16K, 32K, 64K, 128K, 256K}, OSL = 1, concurrency = 1.
* **Decode sweep**: ISL = 1K, OSL = 1K, concurrency ∈ {1, 2, 4, 8, 16, 32, 64}.

Stop conditions per side: TTFT > 120 s for prefill, TTFT > 10 s for decode.

See `gpu/README.md` and `trn2/README.md` for one-host run instructions.
