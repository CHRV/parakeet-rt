# Parakeet Streaming Inference

This document explains the streaming inference implementation for Parakeet TDT models, based on the NVIDIA NeMo streaming inference approach.

## Key Concepts

### 1. Producer/Consumer Pattern (Silero-like)
Following the VAD crate pattern:
- **Audio Producer**: Lock-free feeding of input audio samples
- **Token Consumer**: Lock-free reading of transcription results
- **Processing Engine**: Handles inference and state management

### 2. Context-Based Chunking
Streaming inference uses three types of context:
- **Left Context**: Historical audio for better quality (doesn't affect latency)
- **Chunk**: Current processing window (affects latency)
- **Right Context**: Future audio for better accuracy (affects latency)

**Theoretical Latency = Chunk Size + Right Context**

### 3. Batch Size Always 1
Unlike the Python implementation that supports variable batch sizes, this Rust streaming version is optimized for single-stream processing (batch_size=1), which is typical for real-time applications.

### 4. State Management
The decoder maintains internal state (LSTM hidden/cell states) across chunks to ensure continuity in the transcription.

## Configuration

### Recommended Settings

```rust
// Low latency streaming (500ms total latency)
let context = ContextConfig::new(
    2.0,  // 2s left context (quality)
    0.25, // 250ms chunk (latency)
    0.25, // 250ms right context (latency)
    16000 // sample rate
);

// Balanced streaming (1s total latency)
let context = ContextConfig::new(
    5.0,  // 5s left context
    0.5,  // 500ms chunk
    0.5,  // 500ms right context
    16000
);

// High quality streaming (2s total latency)
let context = ContextConfig::new(
    10.0, // 10s left context
    1.0,  // 1s chunk
    1.0,  // 1s right context
    16000
);
```

### Context Guidelines
- **Left Context**: 2-10 seconds (more = better quality, no latency impact)
- **Chunk Size**: 0.1-1.0 seconds (smaller = lower latency, may reduce quality)
- **Right Context**: 0.25-2.0 seconds (more = better accuracy, increases latency)

## Usage Examples

### Basic Streaming (Silero-like API)

```rust
use parakeet::streaming::{ContextConfig, StreamingParakeetTDT, TokenResult};
use parakeet::parakeet_tdt::ParakeetTDTModel;

// Load model
let model = ParakeetTDTModel::from_pretrained("models", config)?;

// Configure streaming - returns (engine, audio_producer, token_consumer)
let context = ContextConfig::new(2.0, 0.5, 0.5, 16000);
let (mut engine, audio_producer, mut token_consumer) =
    StreamingParakeetTDT::new(model, context, 16000);

// Feed audio (can be done from another thread)
for chunk in audio_chunks {
    for &sample in chunk {
        if audio_producer.push(sample).is_err() {
            // Buffer full, handle overflow
            break;
        }
    }

    // Process audio in engine
    engine.process_audio()?;

    // Read tokens from consumer
    while let Ok(token_result) = token_consumer.pop() {
        println!("Token {} at {:.3}s (conf: {:.2})",
            token_result.token_id,
            token_result.timestamp as f32 / 16000.0,
            token_result.confidence);
    }
}

// Finish processing
engine.finalize();
```

### Multi-threaded Processing

```rust
use std::thread;

// Create streaming engine
let (mut engine, audio_producer, mut token_consumer) =
    StreamingParakeetTDT::new(model, context, 16000);

// Audio capture thread
let audio_thread = thread::spawn(move || {
    for audio_chunk in audio_stream {
        for &sample in &audio_chunk {
            if audio_producer.push(sample).is_err() {
                // Buffer full, skip sample
                break;
            }
        }
    }
});

// Token processing thread
let token_thread = thread::spawn(move || {
    loop {
        while let Ok(token_result) = token_consumer.pop() {
            // Handle transcription tokens
            handle_token(token_result);
        }
        thread::sleep(Duration::from_millis(10));
    }
});

// Main processing loop
loop {
    engine.process_audio()?;
    thread::sleep(Duration::from_millis(10));
}
```

### Real-time Simulation

```rust
// Simulate real-time by feeding small chunks
let chunk_size = sample_rate / 20; // 50ms chunks

for chunk_start in (0..audio.len()).step_by(chunk_size) {
    let chunk_end = std::cmp::min(chunk_start + chunk_size, audio.len());
    let chunk = &audio[chunk_start..chunk_end];

    engine.add_audio(chunk);
    let tokens = engine.process_available()?;

    // Process tokens immediately for real-time response
    handle_new_tokens(tokens);
}
```

## Examples

### 1. Simple Streaming (`simple_streaming.rs`)
Demonstrates basic concepts with synthetic audio:
```bash
cargo run --example simple_streaming --features audio
```

### 2. File Streaming (`streaming_transcribe.rs`)
Processes real audio files in streaming mode:
```bash
cargo run --example streaming_transcribe --features audio -- path/to/audio.wav
```

## Performance Characteristics

### Latency Analysis
- **Theoretical Latency**: chunk_size + right_context
- **Practical Latency**: Add model inference time (~10-50ms per chunk)
- **Memory Usage**: Proportional to total context size

### Quality vs Latency Trade-offs
| Configuration | Latency | Quality | Use Case |
|---------------|---------|---------|----------|
| 10-0.16-3.84s | 4.0s | High | Offline-like quality |
| 10-2-2s | 4.0s | High | Balanced streaming |
| 2-0.25-0.25s | 0.5s | Good | Low-latency streaming |
| 1-0.1-0.1s | 0.2s | Fair | Ultra-low latency |

## Implementation Details

### Audio Buffer Management
The `StreamingAudioBuffer` manages:
- Ring buffer (`rtrb::RingBuffer`) for efficient lock-free audio feeding
- Context window extraction with left/chunk/right segments
- Chunk boundary handling for continuous processing
- End-of-stream processing for final audio segments

### State Continuity
- LSTM states are preserved across chunks
- Token history is maintained for context
- Timestamps are calculated relative to stream start

### Frame-level Processing
- Audio is processed at the encoder frame level
- TDT-specific duration modeling is handled
- Blank token suppression follows TDT algorithm

## Differences from Python Implementation

1. **Batch Size**: Fixed at 1 (single stream)
2. **Memory Management**: More explicit buffer management
3. **State Handling**: Simplified state structure
4. **Error Handling**: Rust-style Result types
5. **Performance**: Lower-level optimizations possible

## Troubleshooting

### Common Issues

1. **No tokens detected**: Check audio levels and model compatibility
2. **High latency**: Reduce chunk size and right context
3. **Poor quality**: Increase left context and chunk size
4. **Memory usage**: Monitor buffer sizes for long streams

### Debug Tips

```rust
// Enable debug output
println!("Buffer size: {}", buffer.len());
println!("Context: {:?}", context);
println!("Tokens so far: {:?}", engine.get_tokens());
```

## Dependencies

The streaming implementation uses:
- `rtrb` - Lock-free ring buffer for efficient audio feeding
- `ndarray` - Multi-dimensional arrays for audio processing
- `ort` - ONNX Runtime for model inference

## Future Enhancements

- [ ] Multi-stream batching support
- [ ] Adaptive context sizing
- [ ] Voice activity detection integration
- [ ] Real-time audio input support
- [ ] WebRTC integration
- [ ] Streaming metrics and monitoring
- [ ] Automatic buffer size tuning
- [ ] Latency monitoring and optimization