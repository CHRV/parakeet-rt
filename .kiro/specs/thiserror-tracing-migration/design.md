# Design Document

## Overview

This design outlines the migration of the parakeet crate's error handling from manual trait implementations to the `thiserror` library, and the addition of comprehensive `tracing` instrumentation throughout the codebase. The migration will improve code maintainability, reduce boilerplate, and provide better observability for debugging and performance analysis.

## Architecture

### Error Handling Architecture

The current error handling uses manual implementations of `std::error::Error`, `Display`, and `From` traits. The new architecture will leverage `thiserror`'s derive macros to automatically generate these implementations.

**Current Structure:**
```rust
#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    Ort(ort::Error),
    Audio(String),
    Model(String),
    Tokenizer(String),
    Config(String),
}
```

**New Structure:**
```rust
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("ONNX Runtime error: {0}")]
    Ort(#[from] ort::Error),

    #[error("Audio processing error: {0}")]
    Audio(String),

    #[error("Model error: {0}")]
    Model(String),

    #[error("Tokenizer error: {0}")]
    Tokenizer(String),

    #[error("Config error: {0}")]
    Config(String),
}
```

### Tracing Architecture

Tracing will be added at multiple levels:

1. **Function-level spans**: Wrap public functions to track execution flow
2. **Operation-level events**: Log key operations (model loading, inference, state updates)
3. **Error-level events**: Emit error events before returning errors
4. **Performance spans**: Track timing for critical operations

**Tracing Hierarchy:**
```
model::from_pretrained [SPAN]
├─ model::find_encoder [EVENT: debug]
├─ model::find_decoder_joint [EVENT: debug]
├─ model::find_preprocessor [EVENT: debug]
└─ session loading [EVENT: info]

model::forward [SPAN]
├─ preprocess [SPAN]
│  └─ preprocessor inference [EVENT: trace]
├─ encode [SPAN]
│  └─ encoder inference [EVENT: trace]
└─ decoding [SPAN]
   └─ decode_frame [SPAN] (per frame)
      └─ decoder_joint inference [EVENT: trace]
```

## Components and Interfaces

### 1. Error Module (`error.rs`)

**Changes:**
- Add `thiserror` dependency
- Replace manual trait implementations with derive macros
- Use `#[from]` attribute for automatic conversions
- Keep the `Result<T>` type alias unchanged
- Remove the conditional `#[cfg(test)]` on `hound::Error` conversion (make it feature-gated instead)

**Interface:**
```rust
pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error)]
pub enum Error {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("ONNX Runtime error: {0}")]
    Ort(#[from] ort::Error),

    #[error("Audio processing error: {0}")]
    Audio(String),

    #[error("Model error: {0}")]
    Model(String),

    #[error("Tokenizer error: {0}")]
    Tokenizer(String),

    #[error("Config error: {0}")]
    Config(String),
}

// Feature-gated conversion for hound errors
#[cfg(feature = "audio")]
impl From<hound::Error> for Error {
    fn from(e: hound::Error) -> Self {
        Error::Audio(e.to_string())
    }
}
```

### 2. Model Module (`model.rs`)

**Tracing Points:**

1. `from_pretrained`: Span wrapping entire model loading
   - Events for finding each model file
   - Events for session creation
   - Error events if loading fails

2. `forward`: Span wrapping full inference pipeline
   - Child spans for preprocess, encode, decoding

3. `preprocess`: Span with input shape information
   - Trace event before/after inference

4. `encode`: Span with input shape information
   - Trace event before/after inference
   - Debug event for output shapes

5. `decode`: Span with frame information
   - Trace events for decoder_joint inference
   - Debug events for token predictions

6. `decoding`: Span wrapping greedy decoding loop
   - Debug events for token emissions
   - Info event for final token count

**Key Instrumentation:**
```rust
#[tracing::instrument(skip(self))]
pub async fn forward(&mut self, waves: Array2<f32>, waves_lens: Array1<i64>)
    -> Result<Vec<(Vec<i32>, Vec<usize>)>> {
    tracing::debug!("Starting forward pass with input shape: {:?}", waves.shape());
    // ... implementation
}
```

### 3. Streaming Module (`streaming.rs`)

**Tracing Points:**

1. `new`/`new_with_vocab`: Debug event for initialization
   - Log buffer sizes and context configuration

2. `process_audio`: Span wrapping audio processing loop
   - Debug events for chunk processing count

3. `process_next_chunk`: Span with chunk information
   - Trace events for preprocessing, encoding
   - Debug events for token emissions
   - Info events for frame processing progress

4. `decode_frame`: Span with frame index
   - Trace events for decoder inference

5. `finalize`: Info event for stream completion

6. `reset`: Debug event for state reset

**Key Instrumentation:**
```rust
#[tracing::instrument(skip(self))]
async fn process_next_chunk(&mut self) -> Result<Vec<(i32, usize)>> {
    tracing::trace!("Processing next audio chunk");
    // ... implementation
    tracing::debug!(tokens = new_tokens.len(), "Emitted tokens from chunk");
    Ok(new_tokens)
}
```

### 4. Decoder Module (`decoder.rs`)

**Tracing Points:**

1. `from_vocab`: Debug event for decoder creation

2. `decode_with_timestamps`: Span with token count
   - Debug events for timestamp calculations
   - Info event for final transcription length

**Key Instrumentation:**
```rust
#[tracing::instrument(skip(self, tokens, frame_indices, _durations))]
pub fn decode_with_timestamps(&self, ...) -> Result<TranscriptionResult> {
    tracing::debug!(token_count = tokens.len(), "Decoding tokens with timestamps");
    // ... implementation
}
```

### 5. Vocab Module (`vocab.rs`)

**Tracing Points:**

1. `from_file`: Span with file path
   - Info event for successful loading
   - Debug event with vocabulary size
   - Error event if loading fails

2. `decode_tokens`: Trace event (only if needed for debugging)

**Key Instrumentation:**
```rust
#[tracing::instrument]
pub fn from_file<P: AsRef<Path>>(vocab_path: P) -> Result<Self> {
    let path = vocab_path.as_ref();
    tracing::debug!("Loading vocabulary from: {}", path.display());
    // ... implementation
    tracing::info!(vocab_size = vocab.size(), "Vocabulary loaded successfully");
    Ok(vocab)
}
```

## Data Models

No changes to existing data models. The `Error` enum structure remains the same, only the implementation changes.

## Error Handling

### Error Propagation

All error propagation remains unchanged due to the `?` operator compatibility. The `thiserror` derive automatically implements the necessary traits.

### Error Context

For errors that need additional context, we'll continue using the string-based variants (Audio, Model, Tokenizer, Config) with formatted messages. The `#[error]` attribute provides the display formatting.

### Error Events

Before returning errors in critical paths, emit a tracing error event:

```rust
if shape.len() != 3 {
    let err = Error::Model(format!("Expected 3D encoder output, got shape: {shape:?}"));
    tracing::error!(?err, "Invalid encoder output shape");
    return Err(err);
}
```

## Testing Strategy

### Unit Tests

1. **Error Module Tests**
   - Verify error display messages match expected format
   - Test automatic `From` conversions
   - Verify error chain preservation

2. **Tracing Tests**
   - Use `tracing-subscriber` test utilities
   - Verify spans are created at expected points
   - Verify events contain expected fields
   - Test that error events are emitted

### Integration Tests

1. **End-to-End Tracing**
   - Set up a test subscriber
   - Run a full inference pipeline
   - Verify complete span hierarchy
   - Verify timing information is captured

2. **Error Propagation**
   - Trigger various error conditions
   - Verify error messages are correct
   - Verify error events are logged

### Manual Testing

1. **Tracing Output**
   - Run examples with `RUST_LOG=trace`
   - Verify readable and useful output
   - Check performance impact is minimal

2. **Error Messages**
   - Trigger errors in examples
   - Verify error messages are clear and actionable

## Implementation Notes

### Dependency Management

**Cargo.toml changes:**
```toml
[dependencies]
thiserror = "2.0"
tracing = "0.1"  # Already present, verify version
```

### Backward Compatibility

- The public API remains unchanged
- Error types remain the same
- Only internal implementations change
- No breaking changes for consumers

### Performance Considerations

1. **Tracing Overhead**
   - Use appropriate log levels (trace for hot paths)
   - Spans are cheap when not subscribed
   - Consider `#[instrument(skip_all)]` for large data structures

2. **Error Creation**
   - `thiserror` has zero runtime overhead compared to manual implementations
   - String formatting only occurs when errors are displayed

### Migration Strategy

1. Update `error.rs` with `thiserror` derives
2. Remove manual trait implementations
3. Add tracing to `vocab.rs` (simplest module)
4. Add tracing to `decoder.rs`
5. Add tracing to `model.rs` (most complex)
6. Add tracing to `streaming.rs`
7. Update tests
8. Verify with examples

## Tracing Levels Guide

- **ERROR**: Error conditions, always logged before returning errors
- **WARN**: Unexpected but recoverable situations (e.g., buffer full)
- **INFO**: High-level operations (model loading, stream completion)
- **DEBUG**: Detailed operation info (token emissions, shape info)
- **TRACE**: Very detailed info (per-frame processing, inference calls)

## Example Tracing Output

```
INFO parakeet::model: Loading model from directory path="models/"
DEBUG parakeet::model: Found encoder model path="models/encoder.onnx"
DEBUG parakeet::model: Found decoder_joint model path="models/decoder_joint.onnx"
INFO parakeet::model: Model loaded successfully
INFO parakeet::streaming: Initializing streaming engine left_samples=32000 chunk_samples=8000 right_samples=16000
DEBUG parakeet::streaming: Processing audio chunk chunk_length=8000
TRACE parakeet::streaming: Running preprocessor
TRACE parakeet::streaming: Running encoder
DEBUG parakeet::streaming: Emitted tokens from chunk tokens=5
INFO parakeet::decoder: Decoding tokens with timestamps token_count=42
DEBUG parakeet::decoder: Final transcription length=156
```
