#!/usr/bin/env python3
"""
Script to compare Python and Rust TDT model outputs.
"""

import numpy as np
from pathlib import Path
import json


def load_and_compare(python_file, rust_file, tolerance=1e-5):
    """Load and compare two numpy arrays."""
    py_path = Path("debug_outputs") / f"{python_file}.npy"
    rust_path = Path("debug_outputs") / f"{rust_file}.npy"

    if not py_path.exists():
        print(f"❌ Python file not found: {py_path}")
        return False

    if not rust_path.exists():
        print(f"❌ Rust file not found: {rust_path}")
        return False

    py_data = np.load(py_path)
    rust_data = np.load(rust_path)

    print(f"\n=== Comparing {python_file} vs {rust_file} ===")
    print(f"Python shape: {py_data.shape}, Rust shape: {rust_data.shape}")
    print(f"Python dtype: {py_data.dtype}, Rust dtype: {rust_data.dtype}")

    if py_data.shape != rust_data.shape:
        print("❌ Shape mismatch!")
        return False

    # Calculate differences
    diff = np.abs(py_data - rust_data)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    print(f"Max difference: {max_diff:.8f}")
    print(f"Mean difference: {mean_diff:.8f}")
    print(f"Tolerance: {tolerance}")

    if max_diff <= tolerance:
        print("✅ Arrays match within tolerance!")
        return True
    else:
        print("❌ Arrays differ beyond tolerance!")

        # Show where the largest differences are
        max_diff_idx = np.unravel_index(np.argmax(diff), diff.shape)
        print(f"Largest diff at index {max_diff_idx}:")
        print(f"  Python: {py_data[max_diff_idx]}")
        print(f"  Rust: {rust_data[max_diff_idx]}")
        print(f"  Diff: {diff[max_diff_idx]}")

        return False


def main():
    print("=== TDT Model Output Comparison ===")

    # Define comparison pairs (python_file, rust_file)
    comparisons = [
        ("01_raw_audio", "rust_01_raw_audio"),
        ("02_audio_batch", "rust_02_audio_batch"),
        ("03_preprocessed_features", "rust_03_preprocessed_features"),
        ("04_features_lengths", "rust_04_features_lengths"),
        ("05_encoder_output", "rust_05_encoder_output"),
        ("06_encoder_lengths", "rust_06_encoder_lengths"),
    ]

    results = []
    for py_file, rust_file in comparisons:
        result = load_and_compare(py_file, rust_file)
        results.append((py_file, result))

    print(f"\n=== Summary ===")
    for filename, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {filename}")

    total_passed = sum(1 for _, passed in results if passed)
    print(f"\nPassed: {total_passed}/{len(results)}")


if __name__ == "__main__":
    main()
