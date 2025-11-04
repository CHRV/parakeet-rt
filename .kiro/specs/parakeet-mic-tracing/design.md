# Design Document

## Overview

This design outlines the addition of comprehensive tracing instrumentation to the parakeet-mic application. The tracing will provide visibility into the three-thread architecture (recording, processing, output), health monitoring, and file I/O operations. The implementation will use the `tracing` crate with a configured subscriber to enable runtime observability through the `RUST_LOG` environment variable.

## Architecture

### Tracing Hierarchy

The parakeet-mic application uses a three-thread architecture with ring buffer communication. Tracing will be added at each level:

```
main [SPAN]
├─ initialization [EVENTS: info, debug]
│  ├─ model loading [EVENT: info]
│  ├─ vocabulary loading [EVENT: info]
│  ├─ audio device setup [EVENT: info]
│  └─ health monitor creation [EVENT: debug]
├─ recording_thread [SPAN]
│  ├─ stream start [EVENT: debug]
│  ├─ audio callback [EVENTS: trace, warn]
│  └─ stream stop [EVENT: debug]
├─ processing_thread [SPAN]
│  ├─ process loop start [EVENT: info]
│  ├─ abandonment detection [EVENT: debug]
│  └─ process loop complete [EVENT: info]
├─ output_thread [SPAN]
│  ├─ token consumption [EVENTS: debug, trace]
│  ├─ file writing [EVENTS: trace, warn]
│  └─ completion [EVENT: info]
└─ shutdown [EVENTS: info, debug]
```

### Subscriber Configuration

The application will initialize a tracing subscriber early in the main function:

```rust
use tracing_subscriber::{fmt, EnvFilter};

// Initialize tracing subscriber
tracing_subscriber::fmt()
    .with_env_filter(EnvFilter::from_default_env())
    .with_thread_names(true)
    .with_target(true)
    .init();
```

This configuration:
- Respects `RUST_LOG` environment variable
- Includes thread names for multi-threaded debugging
- Shows the target module for each event
- Uses human-readable formatting

## Components and Interfaces

### 1. Main Function (`main.rs`)

**Tracing Points:**

1. **Subscriber Initialization**: First operation in main
   ```rust
   tracing_subscriber::fmt()
       .with_env_filter(EnvFilter::from_default_env())
       .with_thread_names(true)
       .init();
   ```

2. **Application Span**: Wrap entire main function
   ```rust
   #[tracing::instrument(name = "parakeet_mic")]
   async fn main() -> Result<()> {
       // ... implementation
   }
   ```

3. **Initialization Events**:
   ```rust
   tracing::info!("Starting Parakeet microphone transcription");
   tracing::debug!(models_path = %args.models, "Loading models");
   tracing::info!("Model loaded successfully");
   tracing::info!(vocab_size = vocab.size(), "Vocabulary loaded");
   tracing::debug!(
       device = %device_name,
       sample_rate = config.sample_rate.0,
       channels = config.channels,
       "Audio device configured"
   );
   ```

4. **Thread Spawn Events**:
   ```rust
   tracing::debug!("Spawning recording thread");
   tracing::debug!("Spawning processing thread");
   tracing::debug!("Spawning output thread");
   ```

5. **Shutdown Events**:
   ```rust
   tracing::info!("Shutdown initiated");
   tracing::debug!("Waiting for threads to complete");
   tracing::info!("Application shutdown complete");
   ```

6. **Error Events**:
   ```rust
   if let Err(e) = recording_result {
       tracing::error!(error = %e, "Recording thread error");
   }
   ```

### 2. Recording Thread (`recording_thread`)

**Tracing Points:**

1. **Thread Span**:
   ```rust
   #[tracing::instrument(
       skip(device, config, audio_producer, audio_writer, health_monitor, shutdown),
       fields(device_name, sample_format, channels)
   )]
   fn recording_thread(...) -> Result<()> {
       // ... implementation
   }
   ```

2. **Stream Lifecycle Events**:
   ```rust
   tracing::debug!(
       sample_format = ?sample_format,
       channels = channels,
       "Building audio input stream"
   );

   tracing::debug!("Audio stream started");

   // In shutdown loop
   tracing::debug!("Shutdown signal detected, stopping recording");

   tracing::debug!("Audio stream stopped");
   ```

3. **Error Events**:
   ```rust
   // In stream error callback
   tracing::trace!(error = %err, "Audio stream error");
   ```

### 3. Audio Stream Callback (`build_audio_stream`)

**Tracing Points:**

The audio callback is a hot path, so tracing must be minimal and rate-limited:

1. **Buffer Full Warning** (important, always log):
   ```rust
   if audio_producer.push(mono_sample).is_err() {
       tracing::warn!("Audio buffer full, dropping samples");
   }
   ```

2. **Periodic Statistics** (use a counter to rate-limit):
   ```rust
   // Add a counter in the closure
   let mut sample_counter = 0;
   const TRACE_INTERVAL: usize = 16000; // Every 1 second at 16kHz

   move |data: &[T], _: &cpal::InputCallbackInfo| {
       sample_counter += data.len() / channels;

       if sample_counter >= TRACE_INTERVAL {
           tracing::trace!(
               samples_processed = sample_counter,
               buffer_slots = audio_producer.slots(),
               "Audio callback statistics"
           );
           sample_counter = 0;
       }
       // ... rest of callback
   }
   ```

3. **Audio Write Errors** (trace level, not critical):
   ```rust
   if let Err(e) = audio_writer.write_sample(mono_sample) {
       tracing::trace!(error = %e, "Failed to write audio sample");
   }
   ```

### 4. Health Monitor (`health.rs`)

**Tracing Points:**

1. **Initialization**:
   ```rust
   impl AudioHealthMonitor {
       pub fn new(threshold: f32, timeout_secs: f32, sample_rate: u32) -> Self {
           tracing::debug!(
               threshold = threshold,
               timeout_secs = timeout_secs,
               sample_rate = sample_rate,
               "Health monitor initialized"
           );
           // ... implementation
       }
   }
   ```

2. **Status Changes**:
   ```rust
   pub fn process_sample(&mut self, sample: f32) -> HealthStatus {
       // ... existing logic

       if self.silence_sample_count >= self.timeout_sample_count && !self.warning_issued {
           self.warning_issued = true;
           tracing::warn!(
               silence_duration_secs = self.timeout_secs,
               threshold = self.threshold,
               "Silence detected - no audio input"
           );
           return HealthStatus::SilenceDetected;
       }

       if self.warning_issued {
           self.warning_issued = false;
           tracing::info!("Audio input resumed");
           return HealthStatus::AudioResumed;
       }

       // ... rest of implementation
   }
   ```

3. **Periodic Statistics** (add to health monitor):
   ```rust
   // Add a method for periodic stats logging
   pub fn log_statistics(&self) {
       tracing::trace!(
           silence_samples = self.silence_sample_count,
           threshold = self.threshold,
           warning_issued = self.warning_issued,
           "Health monitor statistics"
       );
   }
   ```

### 5. Processing Thread (`processing_thread`)

**Tracing Points:**

1. **Thread Span**:
   ```rust
   #[tracing::instrument(skip(processor))]
   async fn processing_thread(mut processor: StreamingParakeetTDT) -> Result<()> {
       tracing::info!("Processing thread started");

       processor
           .process_loop()
           .await
           .map_err(|e| {
               tracing::error!(error = %e, "Processing loop error");
               anyhow!("{}", e)
           })?;

       tracing::info!("Processing thread completed");
       Ok(())
   }
   ```

2. **Abandonment Detection** (if we can detect it):
   ```rust
   // This would require changes to the parakeet library to expose abandonment
   // For now, we rely on the library's internal tracing
   ```

### 6. Output Thread (`output_thread`)

**Tracing Points:**

1. **Thread Span**:
   ```rust
   #[tracing::instrument(
       skip(token_consumer, output_writer, shutdown),
       fields(output_format)
   )]
   fn output_thread(...) -> Result<OutputWriter> {
       tracing::info!("Output thread started");

       let mut token_count = 0;
       // ... implementation
   }
   ```

2. **Token Consumption**:
   ```rust
   while let Ok(token_result) = token_consumer.pop() {
       token_count += 1;
       // ... processing
   }

   if !new_tokens.is_empty() {
       tracing::debug!(
           token_count = new_tokens.len(),
           total_tokens = token_count,
           "Consumed tokens"
       );
   }
   ```

3. **File Writing**:
   ```rust
   if let Err(e) = output_writer.write_tokens(&new_tokens) {
       tracing::warn!(error = %e, "Failed to write tokens to output file");
       eprintln!("Error writing to output file: {}", e);
   }
   ```

4. **Completion**:
   ```rust
   tracing::info!(
       total_tokens = token_count,
       buffer_remaining = token_consumer.slots(),
       "Output thread completed"
   );
   ```

### 7. Output Writer (`output.rs`)

**Tracing Points:**

1. **Initialization**:
   ```rust
   impl OutputWriter {
       pub fn new(...) -> Result<Self> {
           if let Some(ref path) = output_path {
               tracing::debug!(
                   path = %path,
                   format = ?format,
                   append = append,
                   "Output writer initialized"
               );
           }
           // ... implementation
       }
   }
   ```

2. **Write Operations**:
   ```rust
   pub fn write_tokens(&self, tokens: &[TokenResult]) -> Result<()> {
       if let Some(writer) = &self.writer {
           tracing::trace!(token_count = tokens.len(), "Writing tokens to file");
           // ... implementation
       }
       Ok(())
   }
   ```

3. **Finalization**:
   ```rust
   pub fn finalize(&self) -> Result<()> {
       if let Some(writer) = &self.writer {
           tracing::debug!("Finalizing output file");
           // ... implementation
           tracing::info!("Output file finalized successfully");
       }
       Ok(())
   }
   ```

### 8. Audio Writer (`output.rs`)

**Tracing Points:**

1. **Initialization**:
   ```rust
   impl AudioWriter {
       pub fn new(audio_path: Option<String>, sample_rate: u32) -> Result<Self> {
           if let Some(ref path) = audio_path {
               tracing::debug!(
                   path = %path,
                   sample_rate = sample_rate,
                   "Audio writer initialized"
               );
           }
           // ... implementation
       }
   }
   ```

2. **Finalization**:
   ```rust
   pub fn finalize(self) -> Result<()> {
       if let Some(writer) = self.writer {
           tracing::debug!("Finalizing audio file");
           // ... implementation
           tracing::info!("Audio file finalized successfully");
       }
       Ok(())
   }
   ```

## Data Models

No changes to existing data models. Tracing is purely additive instrumentation.

## Error Handling

### Error Event Pattern

Before returning errors or when catching errors, emit a tracing error event:

```rust
if let Err(e) = some_operation() {
    tracing::error!(error = %e, "Operation failed");
    return Err(e);
}
```

### Thread Error Reporting

In the main function, report thread errors with context:

```rust
let recording_result = recording_handle.join().unwrap();
if let Err(e) = recording_result {
    tracing::error!(error = %e, thread = "recording", "Thread error");
    eprintln!("Recording thread error: {}", e);
}
```

## Testing Strategy

### Manual Testing

1. **Basic Tracing Output**
   - Run with `RUST_LOG=info parakeet-mic`
   - Verify initialization and shutdown events appear
   - Verify thread lifecycle events appear

2. **Detailed Tracing**
   - Run with `RUST_LOG=debug parakeet-mic`
   - Verify token consumption and file writing events
   - Verify health monitoring events

3. **Verbose Tracing**
   - Run with `RUST_LOG=trace parakeet-mic`
   - Verify audio callback statistics
   - Verify sample-level events (rate-limited)

4. **Module-Specific Tracing**
   - Run with `RUST_LOG=parakeet_mic::health=debug`
   - Verify only health monitor events appear

5. **Error Scenarios**
   - Trigger silence detection
   - Verify warning events appear
   - Verify audio resume events appear

6. **Performance Testing**
   - Run with `RUST_LOG=off` (no tracing)
   - Run with `RUST_LOG=info` (minimal tracing)
   - Run with `RUST_LOG=trace` (full tracing)
   - Verify no significant performance degradation

### Integration Testing

Since parakeet-mic is an application (not a library), integration testing will be manual:

1. Run the application with various `RUST_LOG` levels
2. Verify trace output is readable and useful
3. Verify no excessive logging in hot paths
4. Verify error events provide actionable information

## Implementation Notes

### Dependency Management

**Cargo.toml changes:**
```toml
[dependencies]
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
```

### Performance Considerations

1. **Hot Path Optimization**
   - Audio callback: Rate-limit trace events to once per second
   - Use `tracing::trace!` for high-frequency events
   - Avoid string formatting in hot paths unless tracing is enabled

2. **Span Overhead**
   - Spans are cheap when no subscriber is active
   - Use `skip` attribute to avoid capturing large data structures
   - Use `fields` attribute to capture specific values

3. **Rate Limiting**
   - Implement counters for periodic statistics
   - Avoid per-sample logging in audio callback
   - Use warn/error levels for important events only

### Thread Safety

All tracing operations are thread-safe by design. The `tracing` crate uses thread-local storage and lock-free data structures for minimal overhead.

### Backward Compatibility

Adding tracing is non-breaking:
- No public API changes
- Tracing is opt-in via `RUST_LOG` environment variable
- Default behavior (no `RUST_LOG`) has minimal overhead

## Tracing Levels Guide

- **ERROR**: Critical errors that prevent operation (thread failures, model loading errors)
- **WARN**: Recoverable issues that may affect quality (buffer full, silence detected, file write errors)
- **INFO**: High-level lifecycle events (startup, shutdown, thread completion, model loaded)
- **DEBUG**: Detailed operational info (token consumption, file operations, configuration)
- **TRACE**: Very detailed info (audio statistics, per-operation timing, sample-level events)

## Example Tracing Output

### Startup (RUST_LOG=info)
```
INFO parakeet_mic: Starting Parakeet microphone transcription
INFO parakeet_mic: Model loaded successfully
INFO parakeet_mic: Vocabulary loaded vocab_size=1024
INFO parakeet_mic: Audio device configured device="MacBook Pro Microphone"
INFO parakeet_mic::processing_thread: Processing thread started
INFO parakeet_mic::output_thread: Output thread started
```

### Runtime (RUST_LOG=debug)
```
DEBUG parakeet_mic::recording_thread: Audio stream started
DEBUG parakeet_mic::health: Health monitor initialized threshold=0.001 timeout_secs=10.0
DEBUG parakeet_mic::output_thread: Consumed tokens token_count=5 total_tokens=42
DEBUG parakeet_mic::output: Writing tokens to file token_count=5
```

### Silence Detection (RUST_LOG=info)
```
WARN parakeet_mic::health: Silence detected - no audio input silence_duration_secs=10.0 threshold=0.001
INFO parakeet_mic::health: Audio input resumed
```

### Shutdown (RUST_LOG=info)
```
INFO parakeet_mic: Shutdown initiated
INFO parakeet_mic::processing_thread: Processing thread completed
INFO parakeet_mic::output_thread: Output thread completed total_tokens=156
INFO parakeet_mic::output: Output file finalized successfully
INFO parakeet_mic: Application shutdown complete
```
