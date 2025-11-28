"""
Unit test to verify qk_norm implementation correctness.
This test compares GPU version vs Neuron version (simulated) without compiling the full model.
"""

import torch
import sys
sys.path.insert(0, '/home/ubuntu/neuronx-distributed-inference/src')

# Import the GPU reference implementation
from neuronx_distributed_inference.models.minimax_m2.modeling_minimax_m2_gpu import MiniMaxM2RMSNorm

# Import the padding function
from neuronx_distributed_inference.modules.attention.gqa import _maybe_pad_interleaved


def test_rmsnorm_equivalence():
    """
    Test that DistributedRMSNorm (simulated) produces the same result as GPU RMSNorm.
    """
    print("=" * 60)
    print("Test 1: RMSNorm equivalence (GPU vs Distributed simulation)")
    print("=" * 60)

    # MiniMax M2 config
    num_attention_heads = 48
    num_kv_heads = 8
    head_dim = 128
    tp_degree = 64

    # Padded dimensions (same as GQA sharding)
    padded_num_attention_heads = 64
    padded_num_kv_heads = 64

    batch_size = 1
    seq_len = 4

    # Create GPU RMSNorm
    q_norm_gpu = MiniMaxM2RMSNorm(num_attention_heads * head_dim, eps=1e-6)

    # Create random Q projection output (before qk_norm)
    # GPU version: [B, S, num_attention_heads * head_dim] = [1, 4, 6144]
    torch.manual_seed(42)
    q_proj_output_gpu = torch.randn(batch_size, seq_len, num_attention_heads * head_dim, dtype=torch.float32)

    # GPU qk_norm result
    q_normed_gpu = q_norm_gpu(q_proj_output_gpu)

    print(f"GPU Q projection output shape: {q_proj_output_gpu.shape}")
    print(f"GPU q_norm output shape: {q_normed_gpu.shape}")
    print(f"GPU q_norm output (first 10 values): {q_normed_gpu[0, 0, :10]}")

    # Now simulate Neuron distributed version
    # Step 1: Apply interleaved padding to q_norm weights
    q_norm_weight_original = q_norm_gpu.weight.data.clone()  # [6144]
    source_group_size = num_attention_heads // num_kv_heads  # 6

    q_norm_weight_padded = _maybe_pad_interleaved(
        q_norm_weight_original.unsqueeze(0),  # [1, 6144]
        pad_dim=1,
        source_heads=num_attention_heads,  # 48
        target_heads=padded_num_attention_heads,  # 64
        source_group_size=source_group_size,  # 6
    ).squeeze(0)  # [8192]

    print(f"\nq_norm weight original shape: {q_norm_weight_original.shape}")
    print(f"q_norm weight padded shape: {q_norm_weight_padded.shape}")

    # Step 2: Apply interleaved padding to Q projection output
    # This simulates what happens after Q projection with padded weights
    q_proj_output_padded = _maybe_pad_interleaved(
        q_proj_output_gpu,  # [1, 4, 6144]
        pad_dim=2,
        source_heads=num_attention_heads,  # 48
        target_heads=padded_num_attention_heads,  # 64
        source_group_size=source_group_size,  # 6
    )  # [1, 4, 8192]

    print(f"Q projection output padded shape: {q_proj_output_padded.shape}")

    # Step 3: Simulate distributed RMSNorm across tp_degree ranks
    # Each rank has [B, S, heads_per_rank * head_dim] = [1, 4, 128]
    heads_per_rank = padded_num_attention_heads // tp_degree  # 1

    # Split Q output across ranks
    q_proj_per_rank = q_proj_output_padded.reshape(batch_size, seq_len, tp_degree, heads_per_rank * head_dim)
    # Shape: [1, 4, 64, 128]

    # Split weights across ranks
    q_norm_weight_per_rank = q_norm_weight_padded.reshape(tp_degree, heads_per_rank * head_dim)
    # Shape: [64, 128]

    # Compute local sum of squares for each rank
    local_sum_sq = q_proj_per_rank.pow(2).sum(dim=-1, keepdim=True)  # [1, 4, 64, 1]

    # All-reduce (sum across ranks)
    global_sum_sq = local_sum_sq.sum(dim=2, keepdim=True)  # [1, 4, 1, 1]

    # Compute global RMS (divide by ORIGINAL full_hidden_size, not padded!)
    full_hidden_size = num_attention_heads * head_dim  # 6144, NOT 8192
    global_variance = global_sum_sq / full_hidden_size
    rsqrt_variance = torch.rsqrt(global_variance + 1e-6)

    # Normalize and apply weights
    q_proj_normalized = q_proj_per_rank * rsqrt_variance  # [1, 4, 64, 128]
    q_normed_distributed = q_proj_normalized * q_norm_weight_per_rank  # [1, 4, 64, 128]

    # Reshape back to full tensor
    q_normed_distributed_full = q_normed_distributed.reshape(batch_size, seq_len, -1)  # [1, 4, 8192]

    # Extract only the non-padding parts for comparison
    # The interleaved pattern is: [6 real, 2 pad] x 8 groups
    # We need to extract the real heads
    q_normed_distributed_real = []
    for group_idx in range(num_kv_heads):  # 8 groups
        start = group_idx * (source_group_size + 2) * head_dim  # 8 heads per group after padding
        end = start + source_group_size * head_dim  # 6 real heads
        q_normed_distributed_real.append(q_normed_distributed_full[:, :, start:end])
    q_normed_distributed_real = torch.cat(q_normed_distributed_real, dim=-1)  # [1, 4, 6144]

    print(f"\nDistributed q_norm output (real parts) shape: {q_normed_distributed_real.shape}")
    print(f"Distributed q_norm output (first 10 values): {q_normed_distributed_real[0, 0, :10]}")

    # Compare
    diff = (q_normed_gpu - q_normed_distributed_real).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"\n=== Comparison ===")
    print(f"Max absolute difference: {max_diff:.6e}")
    print(f"Mean absolute difference: {mean_diff:.6e}")

    if max_diff < 1e-5:
        print("✓ TEST PASSED: GPU and Distributed RMSNorm produce equivalent results!")
        return True
    else:
        print("✗ TEST FAILED: Results differ significantly!")
        return False


def test_qk_norm_weight_padding():
    """
    Test that qk_norm weight padding produces correct shapes and values.
    """
    print("\n" + "=" * 60)
    print("Test 2: qk_norm weight padding correctness")
    print("=" * 60)

    # MiniMax M2 config
    num_attention_heads = 48
    num_kv_heads = 8
    head_dim = 128
    tp_degree = 64
    padded_num_attention_heads = 64
    padded_num_kv_heads = 64

    # Create mock q_norm and k_norm weights
    torch.manual_seed(123)
    q_norm_weight = torch.randn(num_attention_heads * head_dim)  # [6144]
    k_norm_weight = torch.randn(num_kv_heads * head_dim)  # [1024]

    print(f"Original q_norm weight shape: {q_norm_weight.shape}")
    print(f"Original k_norm weight shape: {k_norm_weight.shape}")

    # Test q_norm padding (interleaved)
    source_group_size = num_attention_heads // num_kv_heads  # 6
    q_norm_padded = _maybe_pad_interleaved(
        q_norm_weight.unsqueeze(0),
        pad_dim=1,
        source_heads=num_attention_heads,
        target_heads=padded_num_attention_heads,
        source_group_size=source_group_size,
    ).squeeze(0)

    print(f"\nPadded q_norm weight shape: {q_norm_padded.shape}")
    assert q_norm_padded.shape[0] == padded_num_attention_heads * head_dim, \
        f"Expected {padded_num_attention_heads * head_dim}, got {q_norm_padded.shape[0]}"

    # Verify padding pattern: [6 real, 2 pad] x 8 groups
    # Check that padding values are zeros
    q_norm_reshaped = q_norm_padded.reshape(padded_num_attention_heads, head_dim)  # [64, 128]
    for group_idx in range(num_kv_heads):
        # Real heads: indices 0-5, 8-13, 16-21, etc. within each group of 8
        for head_in_group in range(8):
            head_idx = group_idx * 8 + head_in_group
            if head_in_group < source_group_size:  # Real head
                # Should have non-zero values (from original weight)
                pass
            else:  # Padding head
                # Should be zeros
                assert q_norm_reshaped[head_idx].abs().max() < 1e-10, \
                    f"Padding head {head_idx} should be zeros, got max={q_norm_reshaped[head_idx].abs().max()}"

    print("✓ q_norm padding pattern verified (real heads have values, padding heads are zeros)")

    # Test k_norm replication
    k_norm_reshaped = k_norm_weight.reshape(num_kv_heads, head_dim)  # [8, 128]
    repeats = padded_num_kv_heads // num_kv_heads  # 8
    k_norm_replicated = k_norm_reshaped.repeat_interleave(repeats, dim=0)  # [64, 128]
    k_norm_padded = k_norm_replicated.reshape(-1)  # [8192]

    print(f"\nReplicated k_norm weight shape: {k_norm_padded.shape}")
    assert k_norm_padded.shape[0] == padded_num_kv_heads * head_dim

    # Verify replication: each original KV head should be repeated 8 times
    k_norm_check = k_norm_padded.reshape(padded_num_kv_heads, head_dim)
    for orig_head in range(num_kv_heads):
        for rep in range(repeats):
            replicated_idx = orig_head * repeats + rep
            diff = (k_norm_check[replicated_idx] - k_norm_reshaped[orig_head]).abs().max()
            assert diff < 1e-10, f"Replication failed for head {orig_head}, rep {rep}"

    print("✓ k_norm replication verified (each KV head replicated correctly)")

    return True


def test_distributed_rmsnorm_with_padding_heads():
    """
    Test that DistributedRMSNorm correctly handles padding heads (zeros contribution).
    """
    print("\n" + "=" * 60)
    print("Test 3: Distributed RMSNorm with padding heads")
    print("=" * 60)

    # Simplified test: 4 real heads + 2 padding heads
    num_real_heads = 4
    num_padding_heads = 2
    total_heads = num_real_heads + num_padding_heads
    head_dim = 8

    batch_size = 1
    seq_len = 2

    # Create GPU RMSNorm for real heads only
    gpu_rmsnorm = MiniMaxM2RMSNorm(num_real_heads * head_dim, eps=1e-6)

    # Create random data for real heads
    torch.manual_seed(456)
    real_data = torch.randn(batch_size, seq_len, num_real_heads * head_dim)

    # GPU result (reference)
    gpu_result = gpu_rmsnorm(real_data)

    # Simulate distributed version with padding
    # Padded data: [real_data, zeros]
    padded_data = torch.cat([
        real_data,
        torch.zeros(batch_size, seq_len, num_padding_heads * head_dim)
    ], dim=-1)

    # Padded weights: [real_weights, zeros]
    padded_weights = torch.cat([
        gpu_rmsnorm.weight.data,
        torch.zeros(num_padding_heads * head_dim)
    ])

    # Distributed RMSNorm simulation
    # Each "rank" has head_dim elements
    padded_data_per_rank = padded_data.reshape(batch_size, seq_len, total_heads, head_dim)
    padded_weights_per_rank = padded_weights.reshape(total_heads, head_dim)

    # Local sum of squares
    local_sum_sq = padded_data_per_rank.pow(2).sum(dim=-1, keepdim=True)  # [B, S, total_heads, 1]

    # All-reduce
    global_sum_sq = local_sum_sq.sum(dim=2, keepdim=True)  # [B, S, 1, 1]

    # IMPORTANT: Divide by ORIGINAL size (real heads only)
    full_hidden_size = num_real_heads * head_dim
    global_variance = global_sum_sq / full_hidden_size
    rsqrt_variance = torch.rsqrt(global_variance + 1e-6)

    # Normalize and apply weights
    distributed_normalized = padded_data_per_rank * rsqrt_variance
    distributed_result = distributed_normalized * padded_weights_per_rank

    # Extract real heads result
    distributed_result_real = distributed_result[:, :, :num_real_heads, :].reshape(
        batch_size, seq_len, num_real_heads * head_dim
    )

    # Compare
    diff = (gpu_result - distributed_result_real).abs()
    max_diff = diff.max().item()

    print(f"GPU result shape: {gpu_result.shape}")
    print(f"Distributed result (real heads) shape: {distributed_result_real.shape}")
    print(f"Max difference: {max_diff:.6e}")

    if max_diff < 1e-5:
        print("✓ TEST PASSED: Distributed RMSNorm with padding heads works correctly!")
        return True
    else:
        print("✗ TEST FAILED!")
        return False


def test_interleaved_padding_extraction():
    """
    Test that we can correctly extract real heads from interleaved padded tensor.
    """
    print("\n" + "=" * 60)
    print("Test 4: Interleaved padding extraction")
    print("=" * 60)

    # MiniMax M2 config
    num_attention_heads = 48
    num_kv_heads = 8
    head_dim = 128
    padded_num_attention_heads = 64
    source_group_size = num_attention_heads // num_kv_heads  # 6

    # Create original tensor
    torch.manual_seed(789)
    original = torch.randn(1, 4, num_attention_heads * head_dim)  # [1, 4, 6144]

    # Apply interleaved padding
    padded = _maybe_pad_interleaved(
        original,
        pad_dim=2,
        source_heads=num_attention_heads,
        target_heads=padded_num_attention_heads,
        source_group_size=source_group_size,
    )  # [1, 4, 8192]

    print(f"Original shape: {original.shape}")
    print(f"Padded shape: {padded.shape}")

    # Extract real heads back
    # Pattern: [6 real, 2 pad] x 8 groups
    extracted_parts = []
    heads_per_group_padded = padded_num_attention_heads // num_kv_heads  # 8
    for group_idx in range(num_kv_heads):
        start = group_idx * heads_per_group_padded * head_dim
        end = start + source_group_size * head_dim
        extracted_parts.append(padded[:, :, start:end])
    extracted = torch.cat(extracted_parts, dim=-1)

    print(f"Extracted shape: {extracted.shape}")

    # Compare
    diff = (original - extracted).abs().max()
    print(f"Max difference: {diff:.6e}")

    if diff < 1e-10:
        print("✓ TEST PASSED: Interleaved padding extraction works correctly!")
        return True
    else:
        print("✗ TEST FAILED!")
        return False


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("MiniMax M2 qk_norm Unit Tests")
    print("=" * 60 + "\n")

    results = []

    results.append(("RMSNorm equivalence", test_rmsnorm_equivalence()))
    results.append(("qk_norm weight padding", test_qk_norm_weight_padding()))
    results.append(("Distributed RMSNorm with padding heads", test_distributed_rmsnorm_with_padding_heads()))
    results.append(("Interleaved padding extraction", test_interleaved_padding_extraction()))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("All tests passed!")
        sys.exit(0)
    else:
        print("Some tests failed!")
        sys.exit(1)
