# Implementation Plan

- [ ] 1. Create AudioHealthMonitor module with core monitoring logic
  - Create new file `crates/parakeet-mic/src/health.rs` with the AudioHealthMonitor struct
  - Implement the HealthStatus enum with variants: Healthy, SilenceDetected, AudioResumed, NoChange
  - Implement AudioHealthMonitor::new() constructor with threshold, timeout, and sample_rate parameters
  - Implement AudioHealthMonitor::process_sample() method that tracks silence and returns HealthStatus
  - Implement default_threshold() and default_timeout() as const functions
  - Add module declaration in `crates/parakeet-mic/src/lib.rs`
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [ ] 2. Add command-line arguments for health monitoring configuration
  - Add `audio_threshold` field to Args struct with default value 0.001
  - Add `silence_timeout` field to Args struct with default value 10.0
  - Add validation logic to check for invalid threshold/timeout values and log warnings
  - Update startup output to display configured health monitoring parameters
  - _Requirements: 3.1, 3.2, 3.3, 3.4_

- [ ] 3. Integrate AudioHealthMonitor into the recording thread
  - Create AudioHealthMonitor instance in main() before spawning recording thread
  - Pass the monitor instance to recording_thread() function
  - Modify build_audio_stream() to accept and use the health monitor
  - Update audio callback to call monitor.process_sample() for each mono sample
  - Handle HealthStatus return values and print appropriate warning/resume messages
  - Ensure monitor is shared properly with the audio callback closure
  - _Requirements: 1.1, 1.2, 1.3, 2.1, 2.2, 2.3, 2.4_

- [ ] 4. Implement warning and status message formatting
  - Create helper function to format the silence warning message with troubleshooting tips
  - Create helper function to format the audio resume confirmation message
  - Include platform-specific guidance for macOS in warning messages
  - Add message throttling to prevent console spam
  - _Requirements: 2.1, 2.2, 2.3, 2.4_

- [ ]* 5. Add unit tests for AudioHealthMonitor
  - Test threshold detection with samples above and below threshold
  - Test timeout calculation with various sample rates
  - Test state transitions between Healthy, SilenceDetected, and AudioResumed
  - Test that warnings are not repeated after initial detection
  - Test edge cases: zero samples, alternating levels, boundary conditions
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [ ]* 6. Update documentation
  - Add health monitoring section to README.md explaining the feature
  - Document new command-line arguments (--audio-threshold, --silence-timeout)
  - Add troubleshooting section for common microphone issues
  - Include examples of adjusting sensitivity for different environments
  - _Requirements: 2.1, 2.2, 2.3, 3.1, 3.2, 3.4_
