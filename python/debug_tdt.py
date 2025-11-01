#!/usr/bin/env python3
"""
Debug TDT model step by step.
"""

import sys

sys.path.append("onnx-asr/src")

import numpy as np
from pathlib import Path
from onnx_asr.models.nemo import NemoConformerTdt
import wave


def load_wav_simple(path):
    """Simple WAV loader."""
    with wave.open(path, "rb") as wav_file:
        sr = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        audio_data = wav_file.readframes(n_frames)

    # Convert to float32 array
    audio = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
    return audio, sr


def main():
    print("=== Python TDT Debug ===")

    # Load model
    model_files = {
        "encoder": Path("../models/encoder-model.int8.onnx"),
        "decoder_joint": Path("../models/decoder_joint-model.int8.onnx"),
        "vocab": Path("../models/vocab.txt"),
    }

    print("1. Loading model...")
    model = NemoConformerTdt(model_files, {})
    print(f"   Vocab size: {model._vocab_size}")
    print(f"   Blank idx: {model._blank_idx}")

    # Load audio
    audio_path = "../crates/vad/tests/audio/sample_1.wav"
    print(f"\n2. Loading audio: {audio_path}")
    audio, sr = load_wav_simple(audio_path)
    audio_batch = audio[np.newaxis, :]  # Shape: (1, samples)
    print(f"   Audio batch shape: {audio_batch.shape}")

    # Check methods
    print(f"\n3. Available methods:")
    all_methods = [m for m in dir(model) if not m.startswith("__")]
    print(f"   Methods: {all_methods}")

    # Try preprocessing
    print(f"\n4. Testing preprocessing...")
    try:
        # Use the preprocessor
        features, features_lens = model._preprocessor(audio_batch)
        print(f"   Features shape: {features.shape}")
        print(f"   Features lens: {features_lens}")
        print(f"   Features dtype: {features.dtype}")
        print(f"   Features range: [{features.min():.4f}, {features.max():.4f}]")

        # Try encoding
        print(f"\n5. Testing encoding...")
        encoder_out, encoder_out_lens = model._encode(features, features_lens)
        print(f"   Encoder output shape: {encoder_out.shape}")
        print(f"   Encoder output lens: {encoder_out_lens}")
        print(f"   Encoder output dtype: {encoder_out.dtype}")
        print(
            f"   Encoder output range: [{encoder_out.min():.4f}, {encoder_out.max():.4f}]"
        )

        # Try full recognition
        print(f"\n6. Testing full recognition...")
        results = list(model.recognize_batch([audio_batch]))
        print(f"   Results: {results}")
        if results:
            print(f"   First result: '{results[0].text}'")

    except Exception as e:
        print(f"   Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
