# Parakeet Code Tour

This code tour explains how the Parakeet realtime speech-recognition library is organized and how to run the microphone demo. It mirrors the `.tours/parakeet-codetour.tour` steps but with more detail and runnable commands.

## Contract (what this library provides)

- Inputs: mono f32 audio (16 kHz by default) pushed into the audio producer ring buffer.
- Outputs: tokens (ids and optional decoded text) produced by the streaming engine via a token ring buffer.
- Error modes: missing model files, incompatible ONNX Runtime providers, audio device errors.

## High-level overview

1. Model loading: `ParakeetTDTModel::from_pretrained` finds the required ONNX files in a model directory and instantiates ONNX Runtime sessions.
2. Preprocessing & encoding: model preprocessor converts raw waveform to features; encoder turns features into time frames.
3. Decoding: the decoder/joint model returns token logits and duration step predictions; greedy logic in `decoding` (and streaming `decode_frame`) emits tokens and timestamps.
4. Streaming: `StreamingParakeetTDT` wraps the model, manages left/chunk/right context, and uses lock-free ring buffers to accept audio and push tokens.
5. Example app: `crates/parakeet-mic` wires it all together with CPAL (audio), three threads (recording/processing/output), and graceful shutdown via ring-buffer abandonment detection.

## Tour steps (what to open in the editor)

- `README.md` — Quick start, model download instructions, and example run commands.
- `crates/parakeet/src/lib.rs` — crate exports.
- `crates/parakeet/src/model.rs` — model lifecycle: `from_pretrained`, `preprocess`, `encode`, `decoding`.
- `crates/parakeet/src/decoder.rs` — token -> text with timestamps helper.
- `crates/parakeet/src/streaming.rs` — `ContextConfig`, `StreamingAudioBuffer`, `StreamingParakeetTDT` and FrameProcessor implementation.
- `crates/parakeet-mic/src/main.rs` — practical demo: argument parsing, audio capture, thread coordination, and example usage.

## How to run the microphone demo (quick)

1. Create a `models/` directory at the repo root and download required ONNX files (see `download_models.sh`):

   **Use the included script**
   ./download_models.sh

2. Build & run the microphone demo (default 16kHz, chunk 0.25s):

   cargo run --bin parakeet-mic --release -- --models models

3. Useful flags:
   - `--list-devices` — list available audio input devices
   - `--chunk-size` and `--right-context` — tune latency vs quality

## Important files and what to look for

- `model.rs::from_pretrained` — how session builders are created and configured with `ModelConfig` / `ExecutionProvider`.
- `model.rs::preprocess`, `encode`, `decode` — mapping between ONNX outputs and ndarray shapes; error checks ensure expected tensor ranks.
- `streaming.rs::StreamingAudioBuffer::get_next_chunk` — how left/right contexts are assembled and how the next processing chunk is determined.
- `streaming.rs::process_next_chunk` — the main per-chunk processing: preprocess → encode → decode frames → push tokens.
- `parakeet-mic/src/main.rs` — recording_thread, processing_thread and output_thread show how producers/consumers are created and how shutdown is coordinated.

## Edge cases & notes

- Missing models: `ParakeetTDTModel::find_encoder/decode` will return an error if required ONNX files can't be found; ensure `models/` contains the expected filenames.
- Latency tradeoffs: latency = chunk_size + right_context. Smaller chunk → more frequent processing but higher CPU overhead.
- Abandonment detection: dropping a producer/consumer signals the other side via `is_abandoned()` — the engine then drains remaining buffered data.

## Next steps & recommended small improvements

- Add a short `examples/` binary that runs a prerecorded WAV through the streaming engine to simplify CI and testing.
- Add a small integration test that runs `parakeet-mic` in CI with a tiny model or a mocked ONNX runtime.

---

If you'd like, I can now: add this tour to the CodeTour extension format (already added under `.tours/`), or create an interactive example that sends a short WAV through the streaming engine for a live demo. Which would you prefer next?
