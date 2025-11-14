x # Implementation Plan

- [x] 1. Add dasp dependencies to Cargo.toml
  - Add `dasp_signal`, `dasp_interpolate`, and `dasp_ring_buffer` crates with version 0.11
  - _Requirements: 1.2, 1.3, 3.1_

- [x] 2. Modify build_audio_stream function to support resampling
  - [x] 2.1 Add native_sample_rate parameter to function signature
    - Update function signature to accept `native_sample_rate: u32` parameter
    - _Requirements: 1.1_

  - [x] 2.2 Implement conditional resampler creation in audio callback
    - Create resampler using dasp when native_sample_rate differs from SAMPLE_RATE (16kHz)
    - Use sinc interpolation with 64-sample ring buffer for high-quality resampling
    - Store resampler state in callback closure
    - _Requirements: 1.2, 1.3, 3.1, 3.2_

  - [x] 2.3 Implement resampling logic in audio stream callback
    - Apply resampling to mono samples when resampler is active
    - Pass samples directly to ring buffer when no resampling is needed
    - Handle resampling errors with logging and graceful degradation
    - _Requirements: 1.3, 2.1, 4.1, 4.3_

- [x] 3. Update recording_task to extract and pass native sample rate
  - [x] 3.1 Extract native sample rate from device configuration
    - Query native sample rate from `default_config.sample_rate().0`
    - _Requirements: 1.1_

  - [x] 3.2 Add logging for sample rate information
    - Log native sample rate, target sample rate, and whether resampling is active
    - _Requirements: 2.2_

  - [x] 3.3 Pass native sample rate to build_audio_stream calls
    - Update all three `build_audio_stream` calls (F32, I16, U16) to include native_sample_rate parameter
    - _Requirements: 1.1, 1.4_

- [x] 4. Remove hardcoded sample rate configuration
  - [x] 4.1 Modify initialize_audio_device to use device's default sample rate
    - Query device's default input configuration instead of forcing 16kHz
    - Use native sample rate in StreamConfig
    - _Requirements: 1.1, 2.1_

  - [x] 4.2 Update configuration logging and error messages
    - Update log messages to reflect that we're using native sample rate with resampling
    - Add error handling for unsupported sample rates
    - _Requirements: 2.2, 4.2_
