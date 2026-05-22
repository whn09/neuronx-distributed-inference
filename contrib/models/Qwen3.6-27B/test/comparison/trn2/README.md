# Trn2 side: Qwen3.6-27B vLLM/Neuron benchmark harness

Drives prefill + decode sweeps against a Trn2 vLLM server (NxDI/Neuron backend)
using the same `vllm bench serve` client as the GPU side. Use multi-bucket
compiled artifacts so short prompts don't pad to the largest CTE bucket.

## 1. Compile multi-bucket artifacts

From the repo root, run the BF16 compile script with a bucket list:

```bash
python contrib/models/Qwen3.6-27B/test/integration/qwen36_27b_compile_bf16.py \
  --model-path  /opt/dlami/nvme/models/Qwen3.6-27B-BF16 \
  --compiled-path /opt/dlami/nvme/qwen36/artifacts/bf16_multi_8k \
  --seq-len 9216 \
  --context-encoding-buckets "1024,2048,4096,8192" \
  --tp-degree 4 --logical-nc-config 2
```

The FP8 compile script accepts the same flag:

```bash
python contrib/models/Qwen3.6-27B/test/integration/qwen36_27b_compile_fp8.py \
  --model-path  /opt/dlami/nvme/models/Qwen3.6-27B \
  --compiled-path /opt/dlami/nvme/qwen36/artifacts/fp8_multi_8k \
  --quantized-checkpoints-path /opt/dlami/nvme/qwen36/quantized/fp8_mlp \
  --seq-len 9216 \
  --context-encoding-buckets "1024,2048,4096,8192" \
  --tp-degree 4 --logical-nc-config 2
```

Each additional bucket re-traces the context encoder once, so a 4-bucket compile
takes roughly 4× the single-bucket compile time.

## 2. Start the Neuron vLLM server

The contrib server script (`contrib/.../vllm/start_vllm_server.sh`) accepts
`--context-encoding-buckets` for multi-bucket artifacts:

```bash
contrib/models/Qwen3.6-27B/vllm/start_vllm_server.sh \
  --model-path /opt/dlami/nvme/models/Qwen3.6-27B-BF16 \
  --compiled-artifacts /opt/dlami/nvme/qwen36/artifacts/bf16_multi_8k \
  --max-model-len 9216 \
  --seq-len 9216 \
  --context-encoding-buckets 1024,2048,4096,8192 \
  --tensor-parallel-size 4 \
  --logical-nc-config 2 \
  --port 8000
```

## 3. Run the sweep

```bash
TAG=trn2_bf16 \
PORT=8000 \
SERVED_NAME=/opt/dlami/nvme/models/Qwen3.6-27B-BF16 \
OUTDIR=/opt/dlami/nvme/qwen36/sweeps/trn2_bf16 \
PREFILL_ISLS="1023 2047 4095 8191" \
DECODE_CONC="1 2 4 8" \
./run_sweeps.sh
```

Compared to the GPU sweep, the Trn2 sweep should:
* limit `PREFILL_ISLS` to the buckets you actually compiled, and
* limit `DECODE_CONC` to the `--max-num-seqs` the server was started with
  (Trn2 uses a fixed traced batch).

### Why `PREFILL_ISLS = bucket - 1`

NxDI's first_fit bucket selection
(`neuronx_distributed_inference/models/model_wrapper.py:_get_seq_bucket`) uses
**strict** less-than (`required_len < bucket`), so a request with `input_len`
exactly equal to a compiled bucket boundary falls through to the *next-larger*
bucket. Concretely, with `context_encoding_buckets=[1024, 2048, 4096, 8192]`:

| user ISL | bucket actually used |
|----------|----------------------|
| 1024     | 2048                 |
| 2048     | 4096                 |
| 4096     | 8192                 |
| 8192     | 8192 (via `largest` fallback) |

That is why we default to `bucket - 1` for every ISL except the very largest:
each value lands in the bucket of the same name, so `TTFT(ISL)` actually
reflects bucket-N performance instead of bucket-N+1 padding cost. If you
recompile with different bucket boundaries, update `PREFILL_ISLS` accordingly
(e.g. for buckets `[1024, 4096, 16384]`, use `1023 4095 16383`).

The output JSON files share the exact same schema as the GPU side, so cross-
side plotting is a simple `pandas.read_json` on both directories.
