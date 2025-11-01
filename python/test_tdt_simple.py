#!/usr/bin/env python3
"""
Simple test script to trace through TDT model inference and compare shapes.
"""

import sys

sys.path.append("onnx-asr/src")

import numpy as np
from pathlib import Path
from onnx_asr.models.nemo import NemoConformerTdt
import wave


def load_wav_simple(path):
    """Simple WAV loader without librosa dependency."""
    with wave.open(path, "rb") as wav_file:
        sr = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        audio_data = wav_file.readframes(n_frames)

    # Convert to float32 array
    audio = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
    return audio, sr


def main():
    print("=== Python TDT Model Shape Analysis ===")

    # Load model
    model_files = {
        "encoder": Path("../models/encoder-model.int8.onnx"),
        "decoder_joint": Path("../models/decoder_joint-model.int8.onnx"),
        "vocab": Path("../models/vocab.txt"),
    }

    NemoConformerTdt._preprocessor_name = "nemo128"

    print("1. Loading TDT model...")
    model = NemoConformerTdt(model_files, {})
    print(f"   Model loaded successfully")
    print(f"   Vocab size: {model._vocab_size}")
    print(f"   Blank idx: {model._blank_idx}")

    # Load audio
    audio_path = "../crates/vad/tests/audio/sample_1.wav"
    print(f"\n2. Loading audio: {audio_path}")

    audio, sr = load_wav_simple(audio_path)
    print(f"   Audio shape: {audio.shape}")
    print(f"   Sample rate: {sr}")
    print(f"   Duration: {len(audio) / sr:.2f}s")

    # Add batch dimension for model input
    audio_batch = audio[np.newaxis, :]  # Shape: (1, samples)
    print(f"   Audio batch shape: {audio_batch.shape}")

    # Step 3: Full transcription to see if it works
    print(f"\n3. Full transcription test...")
    try:
        # Use the correct method from the base class
        results = list(model.recognize_batch(audio_batch, np.array([len(audio)]), ""))
        print(f"   Results: {results}")
        if results:
            print(f"   First result text: '{results[0].text}'")
    except Exception as e:
        print(f"   Error: {e}")
        import traceback

        traceback.print_exc()
        return

    # Step 4: Manual step-by-step analysis
    print(f"\n4. Step-by-step analysis...")

    # Get preprocessor - this should be the internal preprocessing
    print("   4a. Preprocessing...")
    features, features_lens = model._preprocessor(audio_batch, np.array([len(audio)]))

    print(f"   Features shape: {features.shape}")
    print(f"   Features lens: {features_lens}")
    print(f"   Features dtype: {features.dtype}")
    print(f"   Features range: [{features.min():.4f}, {features.max():.4f}]")

    # Encoding
    print("   4b. Encoding...")
    encoder_out, encoder_out_lens = model._encode(features, features_lens)
    print(f"   Encoder output shape: {encoder_out.shape}")
    print(f"   Encoder output lens: {encoder_out_lens}")
    print(f"   Encoder output dtype: {encoder_out.dtype}")
    print(
        f"   Encoder output range: [{encoder_out.min():.4f}, {encoder_out.max():.4f}]"
    )

    # Step 5: Decoding (first few steps)
    print(f"\n5. Decoding (first few steps)...")

    # Initialize state
    state = model._create_state()
    print(f"   Initial state shapes: {[s.shape for s in state]}")

    # Test first few decoding steps
    prev_tokens = []
    for t in range(
        min(5, encoder_out.shape[1])
    ):  # encoder_out is (batch, time, features)
        encoder_frame = encoder_out[0]  # Shape: (batch, features)
        print(f"   Step {t}: encoder frame shape: {encoder_frame.shape}")

        # Run decode step
        output, step, new_state = model._decode(prev_tokens, state, encoder_frame[t])
        print(f"   Step {t}: output shape: {output.shape}, step: {step}")
        print(f"   Step {t}: output range: [{output.min():.4f}, {output.max():.4f}]")

        # Get token prediction
        token_probs = output[: model._vocab_size]
        token_id = np.argmax(token_probs)
        print(
            f"   Step {t}: predicted token: {token_id}, blank_idx: {model._blank_idx}"
        )

        if token_id != model._blank_idx:
            prev_tokens.append(token_id)
            state = new_state
            print(f"   Step {t}: emitted token {token_id}")
        else:
            print(f"   Step {t}: blank token")

        if len(prev_tokens) >= 3:  # Stop after a few tokens
            break

    # Step 6: Full transcription


if __name__ == "__main__":
    main()
