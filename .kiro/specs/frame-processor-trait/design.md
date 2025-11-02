# Design Document

## Overview

This design document outlines the architecture for introducing a `FrameProcessor` trait to unify the streaming audio processing patterns used in both the Parakeet and VAD crates. The trait will provide a common interface for frame-by-frame audio processing with ring buffer-based producer-consumer patterns.

The refactoring will create a shared trait in a common location that both crates can use, while keeping the existing struct implementations largely intact. The trait will define the core processing lifecycle methods, and implementations will be provided directly in trait impl blocks for `StreamingParakeetTDT` and `StreamingVad`.

## Architecture

### Module Structure

```
crates/
├── frame-processor/              # New shared crate
│   ├── Cargo.toml
│   └── src/
│       └── lib.rs                # FrameProcessor trait definition
├── parakeet/
│   ├── Cargo.toml                # Add frame-processor dependency
│   └── src/
│       └── streaming.rs          # StreamingParakeetTDT + FrameProcessor impl
└── vad/
    ├── Cargo.toml                # Add frame-processor dependency
    └── src/
        └── streaming_vad.rs      # StreamingVad + FrameProcessor impl
```

The `FrameProcessor` trait will be defined in a new lightweight `frame-processor` crate. This approach:
- Keeps the trait independent from both implementations
- Allows using either `parakeet` or `vad` without pulling in the other
- Provides a clean separation of concerns
- Makes the trait reusable for future audio processing implementations

### Trait Design Philosophy

The trait follows these design principles:
1. **Simplicity**: Minimal required methods with sensible defaults
2. **Flexibility**: Associated types allow customization per implementation
3. **Lifecycle Management**: Clear start, process, and finish semantics
4. **Error Handling**: Consistent error propagation using `thiserror`
5. **Drop Safety**: Automatic finalization on drop

## Components and Interfaces

### FrameProcessor Trait

```rust
use async_trait::async_trait;
use thiserror::Error;

/// Error type for frame processing operations
#[derive(Error, Debug)]
pub enum FrameProcessorError {
    #[error("Processing error: {0}")]
    ProcessingError(String),

    #[error("Buffer error: {0}")]
    BufferError(String),

    #[error("Model error: {0}")]
    ModelError(#[from] Box<dyn std::error::Error + Send + Sync>),
}

/// Trait for frame-by-frame audio processing with streaming semantics
#[async_trait]
pub trait FrameProcessor {
    /// The error type returned by processing operations
    type Error: std::error::Error + Send + Sync + 'static;

    /// Check if another frame is available for processing
    ///
    /// Returns `true` if `process_frame` can be called to process the next frame.
    /// This should check the internal buffer state to determine availability.
    fn has_next_frame(&self) -> bool;

    /// Process the next available frame
    ///
    /// This method should:
    /// 1. Extract the next frame from the input buffer
    /// 2. Perform the necessary processing (VAD, transcription, etc.)
    /// 3. Emit results to the output buffer
    /// 4. Update internal state
    ///
    /// # Errors
    ///
    /// Returns an error if frame processing fails for any reason.
    async fn process_frame(&mut self) -> Result<(), Self::Error>;

    /// Check if the stream has been marked as finished
    ///
    /// Returns `true` if no more input will be provided and all frames
    /// have been processed.
    fn is_finished(&self) -> bool;

    /// Mark the stream as finished
    ///
    /// This signals that no more input will be provided. The processor
    /// should prepare to process any remaining buffered frames.
    fn mark_finished(&mut self);

    /// Finalize processing after the stream is finished
    ///
    /// This method is called after all frames have been processed to perform
    /// any cleanup or final operations. Implementations can override this
    /// to add custom finalization logic.
    ///
    /// # Errors
    ///
    /// Returns an error if finalization fails.
    async fn finalize(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }

    /// Process all available frames in a loop
    ///
    /// This default implementation continues processing while the stream
    /// is not finished, checking for available frames before each call
    /// to `process_frame`. Once the stream is marked as finished and all
    /// frames are processed, it calls `finalize()` and returns.
    ///
    /// # Errors
    ///
    /// Returns the first error encountered during frame processing or finalization.
    async fn process_loop(&mut self) -> Result<(), Self::Error> {
        while !self.is_finished() {
            if self.has_next_frame() {
                self.process_frame().await?;
            } else {
                // No frames available yet, yield to allow other tasks to run
                tokio::task::yield_now().await;
            }
        }

        // Process any remaining frames after stream is finished
        while self.has_next_frame() {
            self.process_frame().await?;
        }

        // Finalize processing
        self.finalize().await?;

        Ok(())
    }
}
```

### StreamingParakeetTDT Implementation

The `StreamingParakeetTDT` struct will implement `FrameProcessor` by adapting its existing methods:

```rust
#[async_trait]
impl FrameProcessor for StreamingParakeetTDT {
    type Error = crate::error::Error;

    fn has_next_frame(&self) -> bool {
        self.buffer.has_next_chunk()
    }

    async fn process_frame(&mut self) -> Result<(), Self::Error> {
        // Delegate to existing process_next_chunk method
        self.process_next_chunk().await?;
        Ok(())
    }

    fn is_finished(&self) -> bool {
        self.buffer.is_finished && !self.has_next_frame()
    }

    fn mark_finished(&mut self) {
        self.buffer.finish();
    }

    async fn finalize(&mut self) -> Result<(), Self::Error> {
        // Any cleanup needed after all frames are processed
        // Currently no additional finalization needed for Parakeet
        Ok(())
    }
}
```

### StreamingVad Implementation

The `StreamingVad` struct will implement `FrameProcessor` similarly:

```rust
#[async_trait]
impl FrameProcessor for StreamingVad {
    type Error = ort::Error;

    fn has_next_frame(&self) -> bool {
        self.audio_consumer.slots() >= self.params.frame_size_samples
    }

    async fn process_frame(&mut self) -> Result<(), Self::Error> {
        // Read one frame from the ring buffer
        if let Ok(frame) = self.audio_consumer.read_chunk(self.params.frame_size_samples) {
            let (first, second) = frame.as_slices();
            let frame_data = [first, second].concat();
            frame.commit_all();

            // Process through Silero VAD
            let speech_prob = self.silero.calc_level(&frame_data)?;

            // Update state
            self.state.update(&self.params, speech_prob);

            // Emit speech samples if triggered
            if self.state.triggered {
                for &sample in &frame_data {
                    let _ = self.speech_producer.push(sample);
                }
            }
        }

        Ok(())
    }

    fn is_finished(&self) -> bool {
        self.is_finished_flag && !self.has_next_frame()
    }

    fn mark_finished(&mut self) {
        self.is_finished_flag = true;
    }

    async fn finalize(&mut self) -> Result<(), Self::Error> {
        // Process final speech segment
        let total_samples = self.state.current_sample;
        self.state.check_for_last_speech(total_samples, &self.params);
        Ok(())
    }
}
```

### Drop Implementation

Both structs should implement `Drop` to ensure proper finalization. Since `process_loop` is async, we need to handle this carefully:

```rust
impl Drop for StreamingParakeetTDT {
    fn drop(&mut self) {
        if !self.is_finished() {
            self.mark_finished();
            // Note: Cannot call async process_loop in Drop
            // Users should explicitly call finalize or use process_loop before dropping
        }
    }
}

impl Drop for StreamingVad {
    fn drop(&mut self) {
        if !self.is_finished() {
            self.mark_finished();
            // Note: Cannot call async process_loop in Drop
            // Users should explicitly call finalize or use process_loop before dropping
        }
    }
}
```

**Note**: Since `Drop` is synchronous and `process_loop` is async, we cannot automatically process remaining frames in `Drop`. Users should explicitly call `mark_finished()` and `process_loop().await` before dropping, or we can provide a synchronous `finalize()` method that blocks on the async operations.

## Data Models

### State Management

Both implementations maintain internal state:

**StreamingParakeetTDT State:**
- `buffer: StreamingAudioBuffer` - Manages audio context and chunking
- `state: Option<State>` - LSTM decoder state
- `previous_token: i32` - Last emitted token for continuity

**StreamingVad State:**
- `state: State` - VAD state machine (triggered, timestamps, etc.)
- `silero: Silero` - The VAD model with internal state

The trait does not expose these internal states, keeping them as implementation details.

### Ring Buffer Flow

```mermaid
graph LR
    A[Audio Input] -->|push| B[Audio Producer]
    B -->|Ring Buffer| C[Audio Consumer]
    C -->|read_chunk| D[FrameProcessor]
    D -->|process_frame| E[Output Producer]
    E -->|Ring Buffer| F[Output Consumer]
    F -->|pop| G[Results]
```

## Error Handling

### Error Type Hierarchy

```rust
// In parakeet crate
#[derive(Error, Debug)]
pub enum Error {
    #[error("ONNX Runtime error: {0}")]
    OrtError(#[from] ort::Error),

    #[error("Frame processing error: {0}")]
    FrameProcessingError(String),

    #[error("Buffer error: {0}")]
    BufferError(String),

    #[error("Invalid state: {0}")]
    InvalidState(String),
}

// In vad crate - already uses ort::Error directly
// No changes needed
```

### Error Propagation

Errors flow from the model layer up through the trait implementation:

1. Model inference error (ort::Error)
2. Frame processing error (trait method)
3. Process loop error (caller)

The `process_loop` default implementation propagates the first error encountered and stops processing.

## Testing Strategy

### Unit Tests

1. **Trait Method Tests**: Test each trait method in isolation
   - `has_next_frame` with various buffer states
   - `process_frame` with valid and invalid frames
   - `is_finished` state transitions
   - `mark_finished` behavior

2. **Implementation Tests**: Test trait implementations
   - StreamingParakeetTDT trait impl
   - StreamingVad trait impl

3. **Drop Tests**: Verify automatic finalization
   - Drop with unprocessed frames
   - Drop after explicit finalization

### Integration Tests

1. **End-to-End Processing**: Test complete processing pipelines
   - Feed audio → process → collect results
   - Verify output correctness

2. **Ring Buffer Integration**: Test producer-consumer patterns
   - Concurrent audio feeding and processing
   - Buffer overflow/underflow handling

3. **State Continuity**: Verify state is maintained across frames
   - LSTM state in Parakeet
   - VAD state machine transitions

### Example-Based Tests

Update existing examples to use the new trait:

```rust
// In examples
async fn process_with_trait<P: FrameProcessor>(mut processor: P) -> Result<(), P::Error> {
    processor.process_loop().await
}

// Usage
let (mut engine, audio_producer, token_consumer) =
    StreamingParakeetTDT::new(model, context, sample_rate);

// Feed audio in background thread
// ...

// Process using trait
process_with_trait(engine).await?;
```

## Async Design

### Rationale

The trait is designed as an async trait using the `async-trait` crate. This decision is based on:

1. **Parakeet Requirements**: The `StreamingParakeetTDT::process_next_chunk` method is async due to ONNX Runtime operations
2. **Flexibility**: Async methods can be called from both async and sync contexts (using block_on if needed)
3. **Future-Proofing**: Async is the direction Rust is moving for I/O-bound operations
4. **Consistency**: Both implementations use the same async interface

### Implementation Details

All trait methods that perform I/O or model inference are marked as `async`:

```rust
#[async_trait]
pub trait FrameProcessor {
    async fn process_frame(&mut self) -> Result<(), Self::Error>;
    async fn process_loop(&mut self) -> Result<(), Self::Error>;
}
```

### Drop Limitation

Since `Drop::drop` is synchronous, we cannot call async methods in the destructor. The `Drop` implementation will only mark the stream as finished:

```rust
impl Drop for StreamingParakeetTDT {
    fn drop(&mut self) {
        if !self.is_finished() {
            self.mark_finished();
            // Note: Cannot call async process_loop in Drop
            // Users should explicitly finalize before dropping
        }
    }
}
```

Users should explicitly finalize processing before dropping:

```rust
// Correct usage
processor.mark_finished();
processor.process_loop().await?;
drop(processor);

// Or use a scoped pattern
{
    let mut processor = create_processor();
    // ... use processor ...
    processor.mark_finished();
    processor.process_loop().await?;
} // processor dropped here
```

## Migration Path

### Phase 1: Create frame-processor Crate
1. Create new `crates/frame-processor` directory
2. Add `Cargo.toml` with `async-trait` and `thiserror` dependencies
3. Create `src/lib.rs` with trait definition
4. Add comprehensive documentation

### Phase 2: Implement for Parakeet
1. Implement `FrameProcessor` for `StreamingParakeetTDT`
2. Add `Drop` implementation
3. Update tests to verify trait behavior

### Phase 3: Implement for VAD
1. Implement `FrameProcessor` for `StreamingVad`
2. Add `Drop` implementation
3. Update tests to verify trait behavior

### Phase 4: Update Examples
1. Demonstrate trait usage in examples
2. Show generic processing functions
3. Document the new pattern

## Design Decisions and Rationales

### Decision 1: Trait Location
**Decision**: Define trait in a separate `frame-processor` crate
**Rationale**: Keeps the trait independent from both implementations, allowing users to depend on only the crates they need without pulling in unnecessary dependencies. This also makes the trait reusable for future audio processing implementations.

### Decision 2: Async Support
**Decision**: Use `async-trait` for async methods
**Rationale**: Maintains the async nature of Parakeet processing while providing a clean trait interface

### Decision 3: Drop-based Finalization
**Decision**: Implement `Drop` to auto-finalize
**Rationale**: Ensures resources are cleaned up and remaining frames are processed even if user forgets to call finalize

### Decision 4: Associated Error Type
**Decision**: Use associated type for errors
**Rationale**: Allows each implementation to use its natural error type (ort::Error for VAD, custom Error for Parakeet)

### Decision 5: Default process_loop
**Decision**: Provide default implementation of process_loop
**Rationale**: Reduces boilerplate and ensures consistent processing logic across implementations

## Performance Considerations

### Zero-Cost Abstraction
The trait should compile to the same machine code as the current direct method calls due to monomorphization.

### Async Overhead
Using `async-trait` adds a small heap allocation for the returned Future, but this is negligible compared to model inference time.

### Ring Buffer Efficiency
No changes to the underlying ring buffer implementation, so performance characteristics remain the same.

## Future Enhancements

1. **Producer Dropping**: Make the audio producer an `Option` so it can be dropped to signal stream completion. When dropped, `consumer.is_abandoned()` will return true, allowing automatic detection of stream end without explicit `mark_finished()` calls.

2. **Generic Frame Types**: Support different frame representations (f32, i16, etc.)

3. **Batch Processing**: Extend trait to support batch frame processing

4. **Metrics**: Add trait methods for performance monitoring

5. **Cancellation**: Support for graceful cancellation of processing loops

6. **Backpressure**: Better handling of slow consumers
