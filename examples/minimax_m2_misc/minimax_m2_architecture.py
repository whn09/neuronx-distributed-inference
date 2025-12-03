"""
MiniMax M2 Architecture Visualization

This script prints a visual representation of the MiniMax M2 model architecture.
"""

def print_architecture():
    # Model configuration
    config = {
        "hidden_size": 3072,
        "num_hidden_layers": 62,
        "num_attention_heads": 48,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "intermediate_size": 1536,  # per expert
        "mlp_intermediate_size": 8192,
        "num_local_experts": 256,
        "num_experts_per_tok": 8,
        "vocab_size": 200064,
        "rotary_dim": 64,
    }

    print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                        MiniMax M2 Model Architecture                          ║
║                     (456B parameters, MoE with 256 experts)                   ║
╚═══════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────────┐
│                              INPUT TOKENS                                    │
│                         [batch, seq_len] int64                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           EMBEDDING LAYER                                    │
│                                                                              │
│   embed_tokens: [vocab_size=200064, hidden_size=3072]                       │
│                                                                              │
│   Input:  [batch, seq_len]                                                   │
│   Output: [batch, seq_len, 3072]                                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
╔═════════════════════════════════════════════════════════════════════════════╗
║                     DECODER LAYERS (x62)                                     ║
╠═════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  ┌───────────────────────────────────────────────────────────────────────┐  ║
║  │                     INPUT LAYERNORM (RMSNorm)                         │  ║
║  │                     weight: [3072]                                     │  ║
║  │                     Input/Output: [batch, seq, 3072]                   │  ║
║  └───────────────────────────────────────────────────────────────────────┘  ║
║                                    │                                         ║
║                                    ▼                                         ║
║  ┌───────────────────────────────────────────────────────────────────────┐  ║
║  │                    SELF-ATTENTION (GQA)                               │  ║
║  │                                                                        │  ║
║  │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                   │  ║
║  │   │  Q Proj     │  │  K Proj     │  │  V Proj     │                   │  ║
║  │   │ 3072→6144   │  │ 3072→1024   │  │ 3072→1024   │                   │  ║
║  │   │ (48 heads)  │  │ (8 heads)   │  │ (8 heads)   │                   │  ║
║  │   └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                   │  ║
║  │          │                │                │                           │  ║
║  │          ▼                ▼                │                           │  ║
║  │   ┌─────────────┐  ┌─────────────┐         │                           │  ║
║  │   │  Q Norm     │  │  K Norm     │         │                           │  ║
║  │   │ RMSNorm     │  │ RMSNorm     │         │                           │  ║
║  │   │ [6144]      │  │ [1024]      │         │                           │  ║
║  │   └──────┬──────┘  └──────┬──────┘         │                           │  ║
║  │          │                │                │                           │  ║
║  │          ▼                ▼                ▼                           │  ║
║  │   ┌─────────────────────────────────────────────┐                     │  ║
║  │   │              RESHAPE TO HEADS               │                     │  ║
║  │   │  Q: [B, 48, S, 128]                         │                     │  ║
║  │   │  K: [B, 8, S, 128]  (GQA: 6 Q heads per KV) │                     │  ║
║  │   │  V: [B, 8, S, 128]                          │                     │  ║
║  │   └──────────────────────┬──────────────────────┘                     │  ║
║  │                          │                                             │  ║
║  │                          ▼                                             │  ║
║  │   ┌─────────────────────────────────────────────┐                     │  ║
║  │   │           PARTIAL ROPE (rotary_dim=64)      │                     │  ║
║  │   │  Apply rotary to Q[:,:,:,:64], K[:,:,:,:64] │                     │  ║
║  │   │  Pass through Q[:,:,:,64:], K[:,:,:,64:]    │                     │  ║
║  │   └──────────────────────┬──────────────────────┘                     │  ║
║  │                          │                                             │  ║
║  │                          ▼                                             │  ║
║  │   ┌─────────────────────────────────────────────┐                     │  ║
║  │   │           SCALED DOT-PRODUCT ATTENTION      │                     │  ║
║  │   │                                             │                     │  ║
║  │   │  scores = Q @ K.T / sqrt(128)              │                     │  ║
║  │   │  attn = softmax(scores + mask) @ V          │                     │  ║
║  │   │                                             │                     │  ║
║  │   │  Output: [B, 48, S, 128]                    │                     │  ║
║  │   └──────────────────────┬──────────────────────┘                     │  ║
║  │                          │                                             │  ║
║  │                          ▼                                             │  ║
║  │   ┌─────────────────────────────────────────────┐                     │  ║
║  │   │              O PROJECTION                   │                     │  ║
║  │   │           [48*128=6144] → 3072              │                     │  ║
║  │   └──────────────────────┬──────────────────────┘                     │  ║
║  │                          │                                             │  ║
║  └──────────────────────────┼────────────────────────────────────────────┘  ║
║                             │                                                ║
║                             ▼                                                ║
║                    ┌────────────────┐                                        ║
║                    │  + RESIDUAL    │                                        ║
║                    └────────┬───────┘                                        ║
║                             │                                                ║
║                             ▼                                                ║
║  ┌───────────────────────────────────────────────────────────────────────┐  ║
║  │                 POST-ATTENTION LAYERNORM (RMSNorm)                    │  ║
║  │                     weight: [3072]                                     │  ║
║  └───────────────────────────────────────────────────────────────────────┘  ║
║                             │                                                ║
║                             ▼                                                ║
║  ┌───────────────────────────────────────────────────────────────────────┐  ║
║  │                    MOE FEED-FORWARD NETWORK                           │  ║
║  │                                                                        │  ║
║  │   ┌─────────────────────────────────────────────────────────────┐     │  ║
║  │   │                      ROUTER                                  │     │  ║
║  │   │   Linear: 3072 → 256 (num_experts)                          │     │  ║
║  │   │   Activation: Sigmoid (not softmax!)                        │     │  ║
║  │   │   + e_score_correction_bias [256]                           │     │  ║
║  │   │   Select Top-8 experts per token                            │     │  ║
║  │   └──────────────────────────┬──────────────────────────────────┘     │  ║
║  │                              │                                         │  ║
║  │                              ▼                                         │  ║
║  │   ┌─────────────────────────────────────────────────────────────┐     │  ║
║  │   │              EXPERTS (x256, Top-8 selected)                  │     │  ║
║  │   │                                                              │     │  ║
║  │   │   Each expert is a GLU MLP:                                 │     │  ║
║  │   │   ┌─────────────────────────────────────────────────────┐   │     │  ║
║  │   │   │  gate_proj: 3072 → 1536                             │   │     │  ║
║  │   │   │  up_proj:   3072 → 1536                             │   │     │  ║
║  │   │   │  down_proj: 1536 → 3072                             │   │     │  ║
║  │   │   │                                                      │   │     │  ║
║  │   │   │  output = down_proj(silu(gate_proj(x)) * up_proj(x))│   │     │  ║
║  │   │   └─────────────────────────────────────────────────────┘   │     │  ║
║  │   │                                                              │     │  ║
║  │   │   Final: weighted sum of 8 expert outputs                   │     │  ║
║  │   └──────────────────────────┬──────────────────────────────────┘     │  ║
║  │                              │                                         │  ║
║  └──────────────────────────────┼────────────────────────────────────────┘  ║
║                                 │                                            ║
║                                 ▼                                            ║
║                        ┌────────────────┐                                    ║
║                        │  + RESIDUAL    │                                    ║
║                        └────────┬───────┘                                    ║
║                                 │                                            ║
╚═════════════════════════════════╪════════════════════════════════════════════╝
                                  │
                        (Repeat 62 times)
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FINAL LAYERNORM (RMSNorm)                            │
│                            weight: [3072]                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              LM HEAD                                         │
│                                                                              │
│   Linear: 3072 → 200064 (vocab_size)                                        │
│                                                                              │
│   Output: [batch, seq_len, 200064] logits                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SAMPLING / GREEDY                                    │
│                     argmax or sample from logits[:,-1,:]                     │
│                              → next token                                    │
└─────────────────────────────────────────────────────────────────────────────┘


╔═══════════════════════════════════════════════════════════════════════════════╗
║                      KEY ARCHITECTURAL FEATURES                               ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  1. UNUSUAL HIDDEN SIZE vs ATTENTION DIM:                                    ║
║     - hidden_size = 3072                                                     ║
║     - Q output = 48 heads × 128 dim = 6144 (LARGER than hidden!)            ║
║     - K/V output = 8 heads × 128 dim = 1024                                  ║
║                                                                               ║
║  2. GROUPED QUERY ATTENTION (GQA):                                           ║
║     - 48 Q heads share 8 KV heads (ratio 6:1)                                ║
║     - Neuron TP=64: pad Q 48→64, replicate KV 8→64                          ║
║                                                                               ║
║  3. PARTIAL ROTARY EMBEDDING:                                                ║
║     - head_dim = 128, but rotary_dim = 64                                    ║
║     - Only first 64 dims get rotary, last 64 pass through                    ║
║                                                                               ║
║  4. QK NORMALIZATION:                                                        ║
║     - RMSNorm applied to full Q/K BEFORE reshape to heads                    ║
║     - q_norm: [6144], k_norm: [1024]                                         ║
║                                                                               ║
║  5. MOE WITH SIGMOID ROUTING:                                                ║
║     - 256 experts, top-8 selection per token                                 ║
║     - Sigmoid activation (not softmax) + bias correction                     ║
║     - Small expert FFN: 3072→1536→3072 (GLU)                                ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝


╔═══════════════════════════════════════════════════════════════════════════════╗
║                    TENSOR PARALLELISM (TP=64) MAPPING                         ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  EMBEDDING (shard_across_embedding=True):                                    ║
║    - Full: [200064, 3072]                                                    ║
║    - Per rank: handles vocab_size/64 = 3126 tokens                           ║
║    - Lookup: only owner rank returns non-zero, all-reduce sum                ║
║                                                                               ║
║  Q PROJECTION (ColumnParallel + interleaved padding):                        ║
║    - Full: [3072, 6144] → padded to [3072, 8192]                            ║
║    - Per rank: [3072, 128] (1 head)                                          ║
║    - Padding pattern: [6 real, 2 pad] × 8 groups                             ║
║                                                                               ║
║  K/V PROJECTION (ColumnParallel + replication):                              ║
║    - Full: [3072, 1024] → replicated to [3072, 8192]                        ║
║    - Per rank: [3072, 128] (1 head, replicated 8x)                           ║
║                                                                               ║
║  QK NORM (DistributedRMSNorm):                                               ║
║    - Compute local sum_sq, all-reduce to get global                          ║
║    - Divide by original dim (6144 for Q, 8192 for K after replication)      ║
║    - Each rank applies its weight slice [128]                                ║
║                                                                               ║
║  O PROJECTION (RowParallel):                                                 ║
║    - Full: [6144, 3072] → padded to [8192, 3072]                            ║
║    - Per rank: [128, 3072]                                                   ║
║    - All-reduce sum after matmul                                             ║
║                                                                               ║
║  MOE EXPERTS:                                                                 ║
║    - Sharded across Expert Parallelism or TP                                 ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    print_architecture()
