# Design Document: Microphone Health Monitor

## Overview

The microphone health monitoring feature adds audio level tracking and silence detection to parakeet-mic. It monitors incoming audio samples in real-time and alerts users when the microphone appears to be non-functional (e.g., due to silent permission blocking on macOS or hardware issues).

The design integrates seamlessly with the existing three-thread architecture without introducing additional threads or complex synchronization. The monitoring logic runs within the existing recording thread's audio callback, making it lightweight and efficient.

## Architecture

### Integration Point

The health monitor integrates into the existing `recording_thread` function by wrapping or augmenting the audio callback that processes incoming samples. The monitor tracks audio levels as samples flow through the system.

### Component Location

```
Recording Thread
  ├── Audio Stream (CPAL)
  ├── Audio Callback
  │   ├── Sample Conversion
  │   ├── Health Monitor (NEW)
  │   │   ├── Level Tracking
  │   │   └── Silence Detection
  │   ├── Ring Buffer Push
  │   └── Audio Writer
  └── Stream Management
```

### Data Flow

```
Microphone → CPAL → Audio Callback → Health Monitor → Ring Buffer → Processing Thread
                                    ↓
                              Warning Messages
```

## Components and Interfaces

### AudioHealthMonitor Struct

```rust
pub struct AudioHealthMonitor {
    /// Threshold below which audio is considered silence
    threshold: f32,

    /// Duration of silence before triggering a warning (in seconds)
    timeout_secs: f32,

    /// Sample rate for time calculations
    sample_rate: u32,

    /// Counter for samples below threshold
    silence_sample_count: usize,

    /// Total samples needed to trigger timeout
    timeout_sample_count: usize,

    /// Whether a warning has been issued
    warning_issued: bool,

    /// Last time a status message was printed
    last_status_time: std::time::Instant,
}
```

### Public API

```rust
impl AudioHealthMonitor {
    /// Create a new health monitor with specified parameters
    pub fn new(threshold: f32, timeout_secs: f32, sample_rate: u32) -> Self;

    /// Process a single audio sample and check health status
    /// Returns true if audio is healthy, false if warning should be issued
    pub fn process_sample(&mut self, sample: f32) -> HealthStatus;

    /// Get default threshold value
    pub const fn default_threshold() -> f32;

    /// Get default timeout value
    pub const fn default_timeout() -> f32;
}

pub enum HealthStatus {
    /// Audio is being received normally
    Healthy,

    /// No audio detected, warning should be issued
    SilenceDetected,

    /// Audio resumed after previous silence warning
    AudioResumed,

    /// No change in status
    NoChange,
}
```

### Command Line Arguments

Add to the existing `Args` struct in `main.rs`:

```rust
/// Threshold for detecting audio (amplitude below this is considered silence)
#[arg(long, default_value = "0.001")]
audio_threshold: f32,

/// Timeout in seconds before warning about no audio input
#[arg(long, default_value = "10.0")]
silence_timeout: f32,
```

## Data Models

### Health Monitor State

The monitor maintains minimal state:
- **silence_sample_count**: Incremented for each sample below threshold, reset when audio detected
- **warning_issued**: Boolean flag to avoid repeated warnings
- **last_status_time**: Used to throttle status messages (avoid spam)

### Sample Processing Logic

```
For each audio sample:
  1. Calculate absolute amplitude
  2. If amplitude > threshold:
     - Reset silence counter
     - If warning was previously issued:
       - Set status to AudioResumed
       - Clear warning flag
  3. If amplitude <= threshold:
     - Increment silence counter
     - If counter >= timeout_sample_count AND not warned:
       - Set status to SilenceDetected
       - Set warning flag
  4. Return status
```

## Error Handling

### Invalid Configuration

- **Negative or zero threshold**: Use default value (0.001) and log warning
- **Negative or zero timeout**: Use default value (10.0) and log warning
- **Extremely low threshold (< 0.00001)**: May cause false positives, log warning but allow
- **Extremely high threshold (> 0.1)**: May never detect audio, log warning but allow

### Edge Cases

- **Very quiet environments**: May trigger false warnings if threshold is too high
- **Noisy environments**: Constant background noise prevents warnings (expected behavior)
- **Intermittent audio**: Short bursts of audio reset the timer (expected behavior)
- **Application startup**: Don't warn during first timeout period (grace period)

### Warning Message Format

```
⚠ WARNING: No audio detected for 10.0 seconds
  Possible causes:
  • Microphone permissions may be blocked by the operating system
  • Wrong audio device selected (use --list-devices to see options)
  • Microphone hardware not connected or muted
  • Ambient noise level below detection threshold (--audio-threshold)

  Tip: On macOS, check System Settings → Privacy & Security → Microphone
```

### Resume Message Format

```
✓ Audio input detected - microphone is working
```

## Testing Strategy

### Unit Tests

1. **Threshold Detection**
   - Test that samples above threshold reset silence counter
   - Test that samples below threshold increment silence counter
   - Test boundary conditions (exactly at threshold)

2. **Timeout Calculation**
   - Verify correct sample count calculation from timeout and sample rate
   - Test various sample rates (8000, 16000, 44100, 48000)

3. **State Transitions**
   - Test transition from Healthy → SilenceDetected
   - Test transition from SilenceDetected → AudioResumed
   - Test that warnings are not repeated

4. **Edge Cases**
   - Test with zero samples
   - Test with alternating loud/quiet samples
   - Test with gradually decreasing amplitude

### Integration Tests

1. **Mock Audio Stream**
   - Create mock audio data with known silence periods
   - Verify warnings are issued at correct times
   - Verify resume messages appear when audio returns

2. **Configuration Validation**
   - Test command-line argument parsing
   - Test default value application
   - Test invalid value handling

### Manual Testing

1. **macOS Permission Blocking**
   - Run application with microphone permissions denied
   - Verify warning appears after timeout
   - Grant permissions and verify resume message

2. **Device Selection**
   - Test with wrong device selected
   - Test with no device connected
   - Test with muted microphone

3. **Real Audio**
   - Test in quiet room (should warn)
   - Test with speech (should not warn)
   - Test with background noise (should not warn)

## Implementation Notes

### Performance Considerations

- **Minimal overhead**: Single comparison and counter increment per sample
- **No allocations**: All state is pre-allocated in the struct
- **No locks**: Runs entirely within audio callback thread
- **Throttled messages**: Status messages limited to avoid console spam

### Thread Safety

The health monitor runs entirely within the audio callback on the recording thread. No synchronization is needed because:
- Only one thread accesses the monitor
- No shared state with other threads
- Warning messages use standard output (already thread-safe)

### Startup Behavior

Display health monitoring configuration at startup:

```
Audio Health Monitoring:
  • Silence threshold: 0.001
  • Silence timeout: 10.0s
  • Status: Active
```

### Graceful Degradation

If health monitoring encounters any issues:
- Log error to stderr
- Disable monitoring for the session
- Continue normal audio capture
- Application remains functional

## Future Enhancements

Potential improvements not included in this initial design:

1. **Audio level visualization**: Display real-time audio levels in console
2. **Periodic health checks**: Report audio levels every N seconds
3. **Automatic device switching**: Try alternative devices if current one fails
4. **Audio level history**: Track and report average levels over time
5. **Platform-specific guidance**: Detect OS and provide targeted troubleshooting steps
