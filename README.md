# Parakeet

A Rust library for real-time speech recognition using the Parakeet TDT (Token-and-Duration Transducer) model with ONNX Runtime.

## Features

- **Streaming inference** with configurable latency and context windows
- **Real-time audio processing** using lock-free ring buffers
- **Async/await support** for efficient processing
- **FrameProcessor trait** for easy integration into audio pipelines
- **Vocabulary support** for token-to-text decoding
- **Cross-platform** audio support via CPAL

## Quick Start

Add parakeet to your `Cargo.toml`:

```toml
[dependencies]
parakeet = { path = "crates/parakeet" }
```

### Basic Usage

```rust
use parakeet::execution::ModelConfig;
use parakeet::model::ParakeetTDTModel;
use parakeet::streaming::{ContextConfig, StreamingParakeetTDT};
use parakeet::vocab::Vocabulary;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load the model
    let exec_config = ModelConfig::default();
    let model = ParakeetTDTModel::from_pretrained("models", exec_config)?;

    // Load vocabulary (optional, for text output)
    let vocab = Vocabulary::from_file("models/vocab.txt").ok();

    // Configure streaming parameters
    let context = ContextConfig::new(
        1.0,    // left_context (seconds)
        0.25,   // chunk_size (seconds)
        0.25,   // right_context (seconds)
        16000   // sample_rate (Hz)
    );

    // Create streaming engine
    let (mut engine, audio_producer, token_consumer) =
        StreamingParakeetTDT::new_with_vocab(model, context, vocab);

    // Feed audio samples (f32, mono, 16kHz)
    for sample in audio_samples {
        let _ = audio_producer.push(sample);
    }

    // Process audio in a separate task
    tokio::spawn(async move {
        engine.process_loop().await
    });

    // Consume tokens
    while let Ok(token) = token_consumer.pop() {
        if let Some(text) = token.text {
            print!("{}", text);
        }
    }

    Ok(())
}
```

## Architecture

Parakeet uses a three-stage pipeline for real-time speech recognition:

```bash
Audio Input → [Ring Buffer] → Processing Engine → [Ring Buffer] → Token Output
```

### Components

1. **ParakeetTDTModel**: Core ONNX model wrapper
   - Preprocessor: Converts audio to features
   - Encoder: Processes audio features
   - Decoder/Joint: Generates tokens with timestamps

2. **StreamingParakeetTDT**: Streaming inference engine
   - Implements `FrameProcessor` trait
   - Manages context windows for low-latency processing
   - Handles automatic shutdown coordination

3. **Vocabulary**: Token-to-text decoder
   - Loads from vocab.txt files
   - Handles SentencePiece tokens

## Configuration

### Context Configuration

The `ContextConfig` controls latency and quality tradeoffs:

```rust
let context = ContextConfig::new(
    left_secs,   // Historical context (improves quality, no latency impact)
    chunk_secs,  // Processing chunk size (affects latency)
    right_secs,  // Future context (affects latency)
    sample_rate  // Audio sample rate in Hz
);
```

**Latency calculation**: `chunk_size + right_context`

Example configurations:

- **Low latency** (500ms): `chunk=0.25s, right=0.25s`
- **Balanced** (1000ms): `chunk=0.75s, right=0.25s`
- **High quality** (2000ms): `chunk=1.75s, right=0.25s`

### Execution Configuration

```rust
use parakeet::execution::{ModelConfig, ExecutionProvider};

let config = ModelConfig::new()
    .with_execution_provider(ExecutionProvider::Cpu)
    .with_intra_threads(4)
    .with_inter_threads(1);
```

## API Reference

### ParakeetTDTModel

```rust
// Load model from directory
let model = ParakeetTDTModel::from_pretrained("models", exec_config)?;

// Process audio (batch mode)
let (tokens, timestamps) = model.forward(audio, audio_lens).await?;

// Streaming components
let (features, features_len) = model.preprocess(audio, audio_lens).await?;
let (encoder_out, encoder_len) = model.encode(features, features_len).await?;
let (probs, step, state) = model.decode(&tokens, state, encoding).await?;
```

### StreamingParakeetTDT

```rust
// Create streaming engine
let (engine, audio_producer, token_consumer) =
    StreamingParakeetTDT::new(model, context);

// With vocabulary
let (engine, audio_producer, token_consumer) =
    StreamingParakeetTDT::new_with_vocab(model, context, Some(vocab));

// Process audio manually
engine.process_audio().await?;

// Or use FrameProcessor trait
engine.process_loop().await?;

// Finalize processing
engine.finalize().await;
```

### Vocabulary

```rust
// Load vocabulary
let vocab = Vocabulary::from_file("models/vocab.txt")?;

// Decode single token
let text = vocab.decode_token(token_id);

// Decode sequence
let text = vocab.decode_tokens(&token_ids);

// Encode token
let id = vocab.encode_token("hello");
```

## FrameProcessor Trait

Parakeet implements the `FrameProcessor` trait for easy integration:

```rust
use frame_processor::FrameProcessor;

// Check if more frames are available
if engine.has_next_frame() {
    // Process next frame
    engine.process_frame().await?;
}

// Process all available frames
engine.process_loop().await?;

// Check if processing is complete
if engine.is_finished() {
    engine.finalize().await?;
}
```

### Automatic Shutdown Coordination

The streaming engine uses ring buffer abandonment detection for graceful shutdown:

- When the audio producer is dropped, the engine processes remaining buffered audio
- When the token consumer is dropped, the engine stops producing tokens
- No explicit coordination or `mark_finished()` calls needed

## Model Requirements

Parakeet requires three ONNX models in the model directory:

1. **Preprocessor**: `nemo128.onnx`
2. **Encoder**: `encoder-model.onnx` or `encoder-model.int8.onnx`
3. **Decoder/Joint**: `decoder_joint-model.onnx` or `decoder_joint-model.int8.onnx`
4. **Vocabulary** (optional): `vocab.txt`

### Downloading Models

The easiest way to download the required models is using the provided script:

```bash
# Create models directory
mkdir -p models

# Download models from Hugging Face
./download_models.sh
```

This will download the Parakeet TDT 0.6B v3 model from [Hugging Face](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx):

- `encoder-model.onnx` - Encoder model (~600MB)
- `encoder-model.onnx.data` - Encoder weights
- `decoder_joint-model.onnx` - Decoder/Joint model
- `vocab.txt` - Vocabulary file for text decoding

**Manual download**: If you prefer to download manually, visit the [model repository](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx) and download the files to the `models/` directory.

## Example: parakeet-mic

See `crates/parakeet-mic` for a complete real-time microphone transcription example:

```bash
# Basic usage
cargo run --bin parakeet-mic

# List audio devices
cargo run --bin parakeet-mic -- --list-devices

# Custom configuration
cargo run --bin parakeet-mic -- \
  --device 0 \
  --sample-rate 16000 \
  --chunk-size 0.25 \
  --output transcription.txt
```

## Performance

- **Latency**: Configurable from 250ms to 1000ms+
- **Memory**: Optimized ring buffers for audio and token streams
- **CPU**: Depends on model complexity and chunk size
- **Threading**: Configurable intra/inter-op parallelism

## Requirements

- Rust 1.70+
- ONNX Runtime (automatically handled by `ort` crate)
- Audio input device (for microphone demo)
