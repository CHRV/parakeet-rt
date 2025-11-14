# Design Document: Audio Resampling to 16kHz

## Overview

This design implements automatic audio resampling for the parakeet-mic-tauri application using the `dasp` (Digital Audio Signal Processing) library. The solution modifies the audio capture pipeline to detect the microphone's native sample rate and resample to 16kHz when necessary, ensuring compatibility with all microphone devices while maintaining optimal performance for devices that natively support 16kHz.

## Architecture

### High-Level Flow

```
Microphone Device
    ↓ (Native Sample Rate)
Audio Stream (cpal)
    ↓
Sample Format Conversion (to f32)
    ↓
[Conditional Resampling]
    ↓ (16kHz f32 samples)
Ring Buffer
    ↓
Processing Pipeline (Parakeet)
```

### Key Design Decisions

1. **Use dasp for resampling**: The `dasp` crate provides high-quality audio resampling with a simple API and good performance characteristics
2. **Conditional resampling**: Only instantiate the resampler when the native sample rate differs from 16kHz to avoid unnecessary overhead
3. **In-stream resampling**: Perform resampling within the audio callback to maintain real-time processing
4. **Maintain existing architecture**: Minimal changes to the current codebase, focusing modifications on the `build_audio_stream` function

## Components and Interfaces

### 1. Resampler Component

**Library**: `dasp` crate (specifically `dasp_signal` and `dasp_interpolate`)

**Key Types**:
- `dasp_signal::Signal`: Trait for signal processing
- `dasp_signal::interpolate::Converter`: Resampler that converts between sample rates
- `dasp_interpolate::sinc::Sinc`: High-quality sinc interpolation for resampling

**Integration Point**: Within the `build_audio_stream` function's audio callback

### 2. Modified Audio Stream Builder

**Function**: `build_audio_stream<T>`

**Changes**:
- Accept `native_sample_rate` parameter
- Conditionally create resampler based on sample rate comparison
- Apply resampling in the audio callback before pushing to ring buffer

**Signature**:
```rust
fn build_audio_stream<T>(
    device: &Device,
    config: &StreamConfig,
    mut audio_producer: rtrb::Producer<f32>,
    channels: usize,
    audio_level: Arc<Mutex<f32>>,
    native_sample_rate: u32,
) -> Result<Stream>
where
    T: Sample + cpal::SizedSample + Send + 'static,
    f32: cpal::FromSample<T>,
```

### 3. Modified Recording Task

**Function**: `recording_task`

**Changes**:
- Extract native sample rate from device configuration
- Pass native sample rate to `build_audio_stream`
- Add logging for sample rate information

## Data Models

### Resampler State

The resampler will be created per audio stream and maintained within the audio callback closure:

```rust
// Conditional resampler creation
let resampler = if native_sample_rate != SAMPLE_RATE {
    Some(dasp_signal::interpolate::Converter::from_hz_to_hz(
        dasp_signal::from_iter(std::iter::empty::<f32>()),
        dasp_interpolate::sinc::Sinc::new(dasp_ring_buffer::Fixed::from([0.0; 64])),
        native_sample_rate as f64,
        SAMPLE_RATE as f64,
    ))
} else {
    None
};
```

### Audio Processing Flow

**Without Resampling** (native rate = 16kHz):
```
Raw samples → Format conversion → Mono conversion → Ring buffer
```

**With Resampling** (native rate ≠ 16kHz):
```
Raw samples → Format conversion → Mono conversion → Resampling → Ring buffer
```

## Error Handling

### Error Scenarios

1. **Unsupported Sample Rate**: If the native sample rate is extremely unusual (e.g., < 8kHz or > 192kHz)
   - Log warning with sample rate information
   - Attempt resampling anyway (dasp is flexible)
   - If resampling fails, propagate error to stop recording

2. **Resampling Runtime Errors**: If resampling encounters issues during processing
   - Log error in audio callback
   - Continue processing (drop problematic samples)
   - Emit error event to frontend if errors persist

3. **Buffer Overflow**: If resampling produces more samples than ring buffer can handle
   - Existing behavior: silently drop samples
   - Add warning log when this occurs

### Error Propagation

```rust
// In recording_task initialization
let native_sample_rate = default_config.sample_rate().0;
println!("Native sample rate: {} Hz, Target: {} Hz, Resampling: {}",
    native_sample_rate,
    SAMPLE_RATE,
    native_sample_rate != SAMPLE_RATE
);

// In audio callback
if let Err(e) = /* resampling operation */ {
    eprintln!("Resampling error: {}", e);
    // Continue processing
}
```

## Testing Strategy

### Unit Testing

Not applicable for this feature as it involves real-time audio processing with hardware dependencies.

### Integration Testing

**Manual Testing Approach**:

1. **Test with 16kHz microphone**:
   - Verify no resampling occurs (check logs)
   - Verify transcription quality is unchanged
   - Verify latency is minimal

2. **Test with 48kHz microphone** (most common):
   - Verify resampling is active (check logs)
   - Verify transcription quality is acceptable
   - Measure latency increase (should be < 100ms)

3. **Test with 44.1kHz microphone**:
   - Verify resampling handles non-integer ratio
   - Verify audio quality is maintained

4. **Test with various microphone types**:
   - USB microphones
   - Built-in laptop microphones
   - Bluetooth microphones (if supported by cpal)

### Performance Testing

**Metrics to Monitor**:
- CPU usage during resampling
- Memory usage (resampler state)
- End-to-end latency
- Audio quality (subjective listening test)

**Acceptance Criteria**:
- CPU usage increase < 5% for 48kHz → 16kHz resampling
- Latency increase < 100ms
- No audible artifacts in transcription

## Implementation Notes

### dasp Configuration

**Interpolation Method**: Sinc interpolation with 64-sample ring buffer
- Provides high-quality resampling
- Good balance between quality and performance
- Suitable for real-time processing

**Alternative Approaches Considered**:
1. **Linear interpolation**: Faster but lower quality
2. **FFT-based resampling**: Higher quality but more complex and higher latency
3. **libsamplerate (SRC)**: C library, requires FFI bindings

**Rationale for dasp**: Pure Rust, good quality, simple API, actively maintained

### Dependency Addition

Add to `parakeet-mic-tauri/src-tauri/Cargo.toml`:
```toml
dasp_signal = "0.11"
dasp_interpolate = "0.11"
dasp_ring_buffer = "0.11"
```

### Code Organization

All resampling logic will be contained within the existing `lib.rs` file:
- Modify `build_audio_stream` function
- Modify `recording_task` function
- No new files or modules required

This keeps the implementation simple and localized to the audio capture logic.
