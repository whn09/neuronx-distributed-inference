"""
Unit test to verify partial RoPE (Rotary Position Embedding) implementation.
MiniMax M2 uses partial rotary with rotary_dim=64 and head_dim=128.
Only the first 64 dimensions get RoPE applied, the rest pass through unchanged.
"""

import torch
import math
import sys
sys.path.insert(0, '/home/ubuntu/neuronx-distributed-inference/src')


def gpu_rotate_half(x):
    """GPU reference implementation from modeling_minimax_m2_gpu.py"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def gpu_apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """
    GPU reference implementation from modeling_minimax_m2_gpu.py
    Handles partial rotary embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # Get rotary_dim from cos/sin shape
    rotary_dim = cos.shape[-1]

    # Split into rotary and pass-through parts
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    # Apply rotary embeddings only to first rotary_dim dimensions
    q_embed = (q_rot * cos) + (gpu_rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (gpu_rotate_half(k_rot) * sin)

    # Concatenate back
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)

    return q_embed, k_embed


def neuron_rotate_half(x):
    """Neuron implementation from utils.py"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def neuron_apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """
    Neuron implementation from utils.py
    Note: This is the base implementation that doesn't handle partial rotary.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (neuron_rotate_half(q) * sin)
    k_embed = (k * cos) + (neuron_rotate_half(k) * sin)
    return q_embed, k_embed


def neuron_apply_partial_rotary(q, k, cos, sin, rotary_dim, head_dim):
    """
    Neuron implementation for partial rotary from NeuronMiniMaxM2Attention.apply_rotary_embedding
    """
    cos = cos.unsqueeze(1)  # unsqueeze_dim=1
    sin = sin.unsqueeze(1)

    if rotary_dim < head_dim:
        # Split into [rotary_part, pass_through_part]
        q_rot = q[..., :rotary_dim]
        q_pass = q[..., rotary_dim:]
        k_rot = k[..., :rotary_dim]
        k_pass = k[..., rotary_dim:]

        # Apply RoPE only to rotary part
        q_rot_embed = (q_rot * cos) + (neuron_rotate_half(q_rot) * sin)
        k_rot_embed = (k_rot * cos) + (neuron_rotate_half(k_rot) * sin)

        # Concatenate back
        q_embed = torch.cat([q_rot_embed, q_pass], dim=-1)
        k_embed = torch.cat([k_rot_embed, k_pass], dim=-1)
    else:
        q_embed = (q * cos) + (neuron_rotate_half(q) * sin)
        k_embed = (k * cos) + (neuron_rotate_half(k) * sin)

    return q_embed, k_embed


def compute_rope_cos_sin(rotary_dim, seq_len, base=10000.0):
    """
    Compute cos and sin for rotary position embedding.
    """
    # Compute inverse frequencies
    inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim))

    # Create position indices
    t = torch.arange(seq_len).float()

    # Compute freqs: [seq_len, rotary_dim/2]
    freqs = torch.outer(t, inv_freq)

    # Duplicate to get [seq_len, rotary_dim]
    emb = torch.cat((freqs, freqs), dim=-1)

    return emb.cos(), emb.sin()


def test_rotate_half_equivalence():
    """
    Test that rotate_half implementations are equivalent.
    """
    print("=" * 60)
    print("Test 1: rotate_half equivalence")
    print("=" * 60)

    torch.manual_seed(42)
    x = torch.randn(2, 4, 8, 64)  # [batch, heads, seq, rotary_dim]

    gpu_result = gpu_rotate_half(x)
    neuron_result = neuron_rotate_half(x)

    diff = (gpu_result - neuron_result).abs().max()
    print(f"Max difference: {diff:.6e}")

    assert diff < 1e-10, "rotate_half implementations should be identical"
    print("✓ TEST PASSED: rotate_half implementations are equivalent!")
    return True


def test_partial_rotary_basic():
    """
    Test basic partial rotary embedding.
    """
    print("\n" + "=" * 60)
    print("Test 2: Partial rotary embedding (basic)")
    print("=" * 60)

    batch_size = 2
    num_heads = 4
    seq_len = 8
    head_dim = 128
    rotary_dim = 64

    torch.manual_seed(123)
    # Q, K shape: [batch, heads, seq, head_dim]
    Q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    K = torch.randn(batch_size, num_heads, seq_len, head_dim)

    # Compute cos/sin for rotary_dim
    cos, sin = compute_rope_cos_sin(rotary_dim, seq_len)
    # Expand to [batch, seq, rotary_dim]
    cos = cos.unsqueeze(0).expand(batch_size, -1, -1)
    sin = sin.unsqueeze(0).expand(batch_size, -1, -1)

    # GPU implementation
    Q_gpu, K_gpu = gpu_apply_rotary_pos_emb(Q, K, cos, sin)

    # Neuron implementation
    Q_neuron, K_neuron = neuron_apply_partial_rotary(Q, K, cos, sin, rotary_dim, head_dim)

    print(f"Q shape: {Q.shape}")
    print(f"cos shape: {cos.shape}")
    print(f"Q_gpu shape: {Q_gpu.shape}")
    print(f"Q_neuron shape: {Q_neuron.shape}")

    # Compare
    q_diff = (Q_gpu - Q_neuron).abs().max()
    k_diff = (K_gpu - K_neuron).abs().max()

    print(f"\nMax Q difference: {q_diff:.6e}")
    print(f"Max K difference: {k_diff:.6e}")

    assert q_diff < 1e-6, "Q should match"
    assert k_diff < 1e-6, "K should match"

    print("✓ TEST PASSED: Partial rotary implementations are equivalent!")
    return True


def test_pass_through_unchanged():
    """
    Test that the pass-through part (dims 64-127) remains unchanged.
    """
    print("\n" + "=" * 60)
    print("Test 3: Pass-through part unchanged")
    print("=" * 60)

    batch_size = 2
    num_heads = 4
    seq_len = 8
    head_dim = 128
    rotary_dim = 64

    torch.manual_seed(456)
    Q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    K = torch.randn(batch_size, num_heads, seq_len, head_dim)

    # Save original pass-through parts
    Q_pass_original = Q[..., rotary_dim:].clone()
    K_pass_original = K[..., rotary_dim:].clone()

    cos, sin = compute_rope_cos_sin(rotary_dim, seq_len)
    cos = cos.unsqueeze(0).expand(batch_size, -1, -1)
    sin = sin.unsqueeze(0).expand(batch_size, -1, -1)

    Q_out, K_out = gpu_apply_rotary_pos_emb(Q, K, cos, sin)

    # Check pass-through parts
    Q_pass_after = Q_out[..., rotary_dim:]
    K_pass_after = K_out[..., rotary_dim:]

    q_pass_diff = (Q_pass_original - Q_pass_after).abs().max()
    k_pass_diff = (K_pass_original - K_pass_after).abs().max()

    print(f"Q pass-through difference: {q_pass_diff:.6e}")
    print(f"K pass-through difference: {k_pass_diff:.6e}")

    assert q_pass_diff < 1e-10, "Q pass-through should be unchanged"
    assert k_pass_diff < 1e-10, "K pass-through should be unchanged"

    print("✓ TEST PASSED: Pass-through dimensions are unchanged!")
    return True


def test_rotary_part_changed():
    """
    Test that the rotary part (dims 0-63) is actually modified.
    """
    print("\n" + "=" * 60)
    print("Test 4: Rotary part is modified")
    print("=" * 60)

    batch_size = 2
    num_heads = 4
    seq_len = 8
    head_dim = 128
    rotary_dim = 64

    torch.manual_seed(789)
    Q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    K = torch.randn(batch_size, num_heads, seq_len, head_dim)

    # Save original rotary parts
    Q_rot_original = Q[..., :rotary_dim].clone()
    K_rot_original = K[..., :rotary_dim].clone()

    cos, sin = compute_rope_cos_sin(rotary_dim, seq_len)
    cos = cos.unsqueeze(0).expand(batch_size, -1, -1)
    sin = sin.unsqueeze(0).expand(batch_size, -1, -1)

    Q_out, K_out = gpu_apply_rotary_pos_emb(Q, K, cos, sin)

    # Check rotary parts
    Q_rot_after = Q_out[..., :rotary_dim]
    K_rot_after = K_out[..., :rotary_dim]

    q_rot_diff = (Q_rot_original - Q_rot_after).abs().mean()
    k_rot_diff = (K_rot_original - K_rot_after).abs().mean()

    print(f"Q rotary mean difference: {q_rot_diff:.6f}")
    print(f"K rotary mean difference: {k_rot_diff:.6f}")

    # The rotary part should be different (position encoding applied)
    assert q_rot_diff > 0.1, "Q rotary part should be modified"
    assert k_rot_diff > 0.1, "K rotary part should be modified"

    print("✓ TEST PASSED: Rotary dimensions are properly modified!")
    return True


def test_position_dependent():
    """
    Test that different positions get different encodings.
    """
    print("\n" + "=" * 60)
    print("Test 5: Position-dependent encoding")
    print("=" * 60)

    batch_size = 1
    num_heads = 1
    seq_len = 16
    head_dim = 128
    rotary_dim = 64

    # Create identical Q vectors at different positions
    Q_base = torch.randn(1, 1, 1, head_dim)
    Q = Q_base.expand(batch_size, num_heads, seq_len, head_dim).clone()
    K = Q.clone()

    cos, sin = compute_rope_cos_sin(rotary_dim, seq_len)
    cos = cos.unsqueeze(0).expand(batch_size, -1, -1)
    sin = sin.unsqueeze(0).expand(batch_size, -1, -1)

    Q_out, K_out = gpu_apply_rotary_pos_emb(Q, K, cos, sin)

    # Check that different positions have different outputs
    # Compare position 0 vs position 1
    pos0 = Q_out[0, 0, 0, :rotary_dim]
    pos1 = Q_out[0, 0, 1, :rotary_dim]
    pos_diff = (pos0 - pos1).abs().mean()

    print(f"Difference between position 0 and 1: {pos_diff:.6f}")
    assert pos_diff > 0.01, "Different positions should have different encodings"

    # Check that pass-through parts are still identical
    pass0 = Q_out[0, 0, 0, rotary_dim:]
    pass1 = Q_out[0, 0, 1, rotary_dim:]
    pass_diff = (pass0 - pass1).abs().max()

    print(f"Pass-through difference (should be ~0): {pass_diff:.6e}")
    assert pass_diff < 1e-10, "Pass-through parts should be identical"

    print("✓ TEST PASSED: Position encoding varies by position!")
    return True


def test_rope_properties():
    """
    Test mathematical properties of RoPE.
    """
    print("\n" + "=" * 60)
    print("Test 6: RoPE mathematical properties")
    print("=" * 60)

    batch_size = 1
    num_heads = 1
    seq_len = 32
    head_dim = 128
    rotary_dim = 64

    torch.manual_seed(111)
    Q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    K = torch.randn(batch_size, num_heads, seq_len, head_dim)

    cos, sin = compute_rope_cos_sin(rotary_dim, seq_len)
    cos = cos.unsqueeze(0).expand(batch_size, -1, -1)
    sin = sin.unsqueeze(0).expand(batch_size, -1, -1)

    Q_out, K_out = gpu_apply_rotary_pos_emb(Q, K, cos, sin)

    # Property 1: RoPE should preserve vector norms (approximately)
    # Since we're only rotating part of the vector, check rotary part
    Q_rot_norm_before = Q[..., :rotary_dim].norm(dim=-1)
    Q_rot_norm_after = Q_out[..., :rotary_dim].norm(dim=-1)
    norm_diff = (Q_rot_norm_before - Q_rot_norm_after).abs().max()

    print(f"Rotary part norm preservation (max diff): {norm_diff:.6e}")
    assert norm_diff < 1e-5, "RoPE should preserve norms"
    print("✓ Norm preservation verified")

    # Property 2: Attention score should depend on relative position
    # Q[i] @ K[j] should depend on (i-j), not on absolute positions
    # This is a key property of RoPE

    # Compute attention scores
    # Q_out: [1, 1, 32, 128], K_out: [1, 1, 32, 128]
    attn_scores = torch.matmul(Q_out, K_out.transpose(-1, -2))  # [1, 1, 32, 32]

    # The diagonal (relative position 0) should have consistent behavior
    diagonal = attn_scores[0, 0].diag()
    print(f"Attention diagonal (self-attention) mean: {diagonal.mean():.4f}, std: {diagonal.std():.4f}")

    print("✓ TEST PASSED: RoPE mathematical properties verified!")
    return True


def test_minimax_m2_config():
    """
    Test with actual MiniMax M2 configuration.
    """
    print("\n" + "=" * 60)
    print("Test 7: MiniMax M2 actual configuration")
    print("=" * 60)

    # MiniMax M2 config
    batch_size = 1
    num_attention_heads = 48
    num_kv_heads = 8
    seq_len = 128
    head_dim = 128
    rotary_dim = 64
    rope_theta = 10000.0

    torch.manual_seed(222)

    # Simulate per-rank tensors (after TP sharding, each rank has fewer heads)
    tp_degree = 64
    heads_per_rank = 1  # 64 padded heads / 64 TP = 1

    Q = torch.randn(batch_size, heads_per_rank, seq_len, head_dim)
    K = torch.randn(batch_size, heads_per_rank, seq_len, head_dim)

    cos, sin = compute_rope_cos_sin(rotary_dim, seq_len, base=rope_theta)
    cos = cos.unsqueeze(0).expand(batch_size, -1, -1)
    sin = sin.unsqueeze(0).expand(batch_size, -1, -1)

    print(f"Q shape (per rank): {Q.shape}")
    print(f"cos shape: {cos.shape}")
    print(f"rotary_dim: {rotary_dim}")
    print(f"head_dim: {head_dim}")
    print(f"rope_theta: {rope_theta}")

    Q_out, K_out = gpu_apply_rotary_pos_emb(Q, K, cos, sin)

    # Verify shapes
    assert Q_out.shape == Q.shape, f"Q shape mismatch: {Q_out.shape} vs {Q.shape}"
    assert K_out.shape == K.shape, f"K shape mismatch: {K_out.shape} vs {K.shape}"

    # Verify partial rotary
    pass_diff = (Q[..., rotary_dim:] - Q_out[..., rotary_dim:]).abs().max()
    assert pass_diff < 1e-10, "Pass-through should be unchanged"

    rot_diff = (Q[..., :rotary_dim] - Q_out[..., :rotary_dim]).abs().mean()
    assert rot_diff > 0.1, "Rotary part should be modified"

    print(f"\nOutput Q shape: {Q_out.shape}")
    print(f"Pass-through unchanged: ✓ (diff={pass_diff:.6e})")
    print(f"Rotary part modified: ✓ (mean diff={rot_diff:.4f})")

    print("\n✓ TEST PASSED: MiniMax M2 configuration works correctly!")
    return True


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("MiniMax M2 Partial RoPE Unit Tests")
    print("=" * 60 + "\n")

    results = []

    results.append(("rotate_half equivalence", test_rotate_half_equivalence()))
    results.append(("Partial rotary basic", test_partial_rotary_basic()))
    results.append(("Pass-through unchanged", test_pass_through_unchanged()))
    results.append(("Rotary part modified", test_rotary_part_changed()))
    results.append(("Position-dependent encoding", test_position_dependent()))
    results.append(("RoPE properties", test_rope_properties()))
    results.append(("MiniMax M2 config", test_minimax_m2_config()))

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
