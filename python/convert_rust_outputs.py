#!/usr/bin/env python3
"""
Convert Rust raw binary outputs to numpy format for comparison.
"""

import numpy as np
import struct
from pathlib import Path


def convert_rust_raw_to_npy(raw_file_path, npy_file_path):
    """Convert Rust raw binary format to numpy .npy format."""
    with open(raw_file_path, "rb") as f:
        # Read number of dimensions
        ndims_bytes = f.read(8)
        ndims = struct.unpack("<Q", ndims_bytes)[0]

        # Read shape
        shape = []
        for _ in range(ndims):
            dim_bytes = f.read(8)
            dim = struct.unpack("<Q", dim_bytes)[0]
            shape.append(dim)

        # Read data
        total_elements = np.prod(shape)
        data_bytes = f.read(total_elements * 4)  # 4 bytes per float32
        data = struct.unpack(f"<{total_elements}f", data_bytes)

        # Create numpy array
        array = np.array(data, dtype=np.float32).reshape(shape)

        # Save as .npy
        np.save(npy_file_path, array)

        print(f"Converted {raw_file_path} -> {npy_file_path}")
        print(f"  Shape: {shape}")
        print(f"  Range: [{array.min():.6f}, {array.max():.6f}]")

        return array


def main():
    print("=== Converting Rust Raw Outputs to NumPy Format ===")

    output_dir = Path("debug_outputs")

    # List of files to convert
    rust_files = [
        "rust_01_raw_audio",
        "rust_02_audio_batch",
        "rust_03_preprocessed_features",
        "rust_04_features_lengths",
        "rust_05_encoder_output",
        "rust_06_encoder_lengths",
    ]

    converted_files = []

    for filename in rust_files:
        raw_path = output_dir / f"{filename}.raw"
        npy_path = output_dir / f"{filename}.npy"

        if raw_path.exists():
            try:
                array = convert_rust_raw_to_npy(raw_path, npy_path)
                converted_files.append(filename)
            except Exception as e:
                print(f"Error converting {filename}: {e}")
        else:
            print(f"Raw file not found: {raw_path}")

    print(f"\n=== Conversion Complete ===")
    print(f"Converted {len(converted_files)} files:")
    for filename in converted_files:
        print(f"  - {filename}.npy")

    print(f"\nNow you can run: python compare_outputs.py")


if __name__ == "__main__":
    main()
