#!/usr/bin/env python3
"""
Comprehensive unit tests for MiniMax M2 model adaptation to Trainium2.
Run all tests to verify the implementation before compiling the full model.

Usage:
    python test_minimax_m2_all.py
"""

import subprocess
import sys


def run_test(test_name, test_file):
    """Run a test file and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {test_name}")
    print(f"{'='*60}")

    result = subprocess.run(
        [sys.executable, test_file],
        capture_output=False,
        text=True
    )

    return result.returncode == 0


def main():
    print("=" * 60)
    print("MiniMax M2 Comprehensive Unit Tests")
    print("=" * 60)
    print("\nThis will verify all critical components:")
    print("  1. qk_norm (RMSNorm with distributed all-reduce)")
    print("  2. MoE Router (sigmoid + e_score_correction_bias)")
    print("  3. Partial RoPE (rotary_dim=64, head_dim=128)")
    print("\n")

    tests = [
        ("QK Norm Tests", "test_qk_norm.py"),
        ("MoE Router Tests", "test_moe_router.py"),
        ("Partial RoPE Tests", "test_rope_partial.py"),
    ]

    results = []
    for test_name, test_file in tests:
        passed = run_test(test_name, test_file)
        results.append((test_name, passed))

    # Print summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    all_passed = True
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("\n✓ ALL TESTS PASSED!")
        print("\nYou can now proceed to compile the model:")
        print("  1. Delete old compiled model: rm -rf /home/ubuntu/traced_model/MiniMax-M2-BF16-weights/")
        print("  2. Run: python generation_minimax_m2_demo.py")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED!")
        print("Please fix the failing tests before compiling.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
