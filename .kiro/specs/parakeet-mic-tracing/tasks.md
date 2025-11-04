# Implementation Plan

- [x] 1. Add tracing dependencies to Cargo.toml
  - Add `tracing = "0.1"` to dependencies in `crates/parakeet-mic/Cargo.toml`
  - Add `tracing-subscriber = { version = "0.3", features = ["env-filter"] }` to dependencies
  - _Requirements: 7.1_

- [x] 2. Initialize tracing subscriber in main function
  - Add `use tracing_subscriber::{fmt, EnvFilter};` import to `crates/parakeet-mic/src/main.rs`
  - Add tracing subscriber initialization as first operation in main function
  - Configure subscriber with `EnvFilter::from_default_env()` to respect `RUST_LOG`
  - Enable thread names with `.with_thread_names(true)`
  - Enable target module with `.with_target(true)`
  - _Requirements: 7.2, 7.3, 7.4, 7.5_

- [x] 3. Add tracing to main function lifecycle
  - Add `use tracing;` import to `crates/parakeet-mic/src/main.rs`
  - Add `tracing::info!` event at start of main with application name
  - Add `tracing::debug!` event after parsing args with models path
  - Add `tracing::info!` event after model loading success
  - Add `tracing::info!` event after vocabulary loading with vocab size
  - Add `tracing::debug!` event after audio device configuration with device name, sample rate, channels
  - Add `tracing::debug!` events when spawning each thread (recording, processing, output)
  - Add `tracing::info!` event when shutdown is initiated
  - Add `tracing::debug!` event before waiting for threads
  - Add `tracing::error!` events for thread errors with thread name and error
  - Add `tracing::info!` event at application shutdown complete
  - _Requirements: 1.2, 1.3, 1.4, 1.5_

- [x] 4. Add tracing to recording thread
  - Add `#[tracing::instrument(skip(device, config, audio_producer, audio_writer, health_monitor, shutdown))]` to `recording_thread` function
  - Add `tracing::debug!` event with sample format and channels before building stream
  - Add `tracing::debug!` event after stream starts
  - Add `tracing::debug!` event when shutdown signal is detected
  - Add `tracing::debug!` event after stream stops
  - _Requirements: 2.1, 2.2, 2.5_

- [x] 5. Add tracing to audio stream callback
  - Add rate-limiting counter in `build_audio_stream` closure (sample_counter, TRACE_INTERVAL = 16000)
  - Add `tracing::trace!` event with periodic statistics (samples processed, buffer slots) when counter reaches interval
  - Add `tracing::warn!` event when audio_producer.push() fails (buffer full)
  - Add `tracing::trace!` event when audio_writer.write_sample() fails
  - Modify stream error callback to use `tracing::trace!` instead of eprintln
  - _Requirements: 3.1, 3.2, 3.3, 3.5_

- [x] 6. Add tracing to health monitor
  - Add `use tracing;` import to `crates/parakeet-mic/src/health.rs`
  - Add `tracing::debug!` event in `AudioHealthMonitor::new` with threshold, timeout_secs, sample_rate
  - Add `tracing::warn!` event in `process_sample` when silence is detected with silence_duration_secs and threshold
  - Add `tracing::info!` event in `process_sample` when audio resumes
  - _Requirements: 4.1, 4.2, 4.3, 4.4_

- [x] 7. Add tracing to processing thread
  - Add `#[tracing::instrument(skip(processor))]` to `processing_thread` function
  - Add `tracing::info!` event at start of processing thread
  - Add `tracing::error!` event in error mapping with error details
  - Add `tracing::info!` event at completion of processing thread
  - _Requirements: 5.1, 5.2, 5.4_

- [x] 8. Add tracing to output thread
  - Add `#[tracing::instrument(skip(token_consumer, output_writer, shutdown))]` to `output_thread` function
  - Add `tracing::info!` event at start of output thread
  - Add token_count variable to track total tokens consumed
  - Add `tracing::debug!` event after consuming tokens with token count and total
  - Add `tracing::warn!` event when output_writer.write_tokens() fails
  - Add `tracing::info!` event at completion with total_tokens and buffer_remaining
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

- [x] 9. Add tracing to output writer
  - Add `use tracing;` import to `crates/parakeet-mic/src/output.rs`
  - Add `tracing::debug!` event in `OutputWriter::new` when output_path is Some with path, format, append
  - Add `tracing::trace!` event in `write_tokens` with token count
  - Add `tracing::debug!` event at start of `finalize` method
  - Add `tracing::info!` event after successful finalization
  - _Requirements: 6.3_

- [x] 10. Add tracing to audio writer
  - Add `tracing::debug!` event in `AudioWriter::new` when audio_path is Some with path and sample_rate
  - Add `tracing::debug!` event at start of `finalize` method
  - Add `tracing::info!` event after successful finalization
  - _Requirements: 8.3, 8.4_

- [ ]* 11. Test tracing output with different log levels
  - Run `RUST_LOG=info cargo run --bin parakeet-mic -- --models models` and verify info-level events
  - Run `RUST_LOG=debug cargo run --bin parakeet-mic -- --models models` and verify debug-level events
  - Run `RUST_LOG=trace cargo run --bin parakeet-mic -- --models models` and verify trace-level events
  - Run `RUST_LOG=parakeet_mic::health=debug cargo run --bin parakeet-mic -- --models models` and verify module-specific filtering
  - Verify thread names appear in output
  - Verify timestamps appear in output
  - _Requirements: 7.3, 7.4, 7.5_

- [ ]* 12. Test error scenarios and silence detection
  - Trigger silence detection by not speaking for timeout period
  - Verify `WARN` event appears for silence detection
  - Verify `INFO` event appears when audio resumes
  - Test with invalid models path and verify error events
  - _Requirements: 4.2, 4.3_
