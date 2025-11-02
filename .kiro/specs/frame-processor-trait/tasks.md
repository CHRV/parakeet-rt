# Implementation Plan

- [x] 1. Create frame-processor crate
  - Create new crate directory structure with Cargo.toml and src/lib.rs
  - Add dependencies: async-trait and thiserror
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [x] 1.1 Define FrameProcessor trait
  - Write the trait definition with async-trait annotation
  - Include all required methods: has_next_frame, process_frame, is_finished, mark_finished, finalize
  - Implement default process_loop method
  - Add comprehensive documentation comments
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 4.1, 4.2, 4.3, 4.4, 4.5, 5.1, 5.2, 5.3_

- [x] 1.2 Add FrameProcessorError type
  - Define error enum using thiserror
  - Include variants for common error cases
  - _Requirements: 4.1, 4.2, 4.3, 4.5_

- [x] 2. Update parakeet crate
  - Add frame-processor dependency to Cargo.toml
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

- [x] 2.1 Implement FrameProcessor for StreamingParakeetTDT
  - Add async-trait import
  - Implement all trait methods in the trait impl block
  - Map has_next_frame to buffer.has_next_chunk()
  - Map process_frame to process_next_chunk()
  - Implement is_finished logic
  - Implement mark_finished to call buffer.finish()
  - Implement finalize method (empty for now)
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

- [x] 2.2 Update StreamingAudioBuffer
  - Add is_finished field tracking
  - Ensure finish() method sets the flag correctly
  - _Requirements: 2.4_

- [x] 2.3 Update existing process_audio method
  - Refactor to use the trait methods internally if beneficial
  - Maintain backward compatibility
  - _Requirements: 2.5_

- [x] 3. Update vad crate
  - Add frame-processor dependency to Cargo.toml
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

- [x] 3.1 Add is_finished field to StreamingVad
  - Add is_finished_flag: bool field to struct
  - Initialize to false in new()
  - _Requirements: 3.4_

- [x] 3.2 Implement FrameProcessor for StreamingVad
  - Add async-trait import
  - Implement all trait methods in the trait impl block
  - Map has_next_frame to check audio_consumer.slots()
  - Implement process_frame with frame reading and VAD processing
  - Implement is_finished logic
  - Implement mark_finished to set flag
  - Implement finalize to call check_for_last_speech
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

- [x] 3.3 Update existing process_audio method
  - Refactor to use the trait methods internally if beneficial
  - Maintain backward compatibility
  - _Requirements: 3.5_

- [ ] 4. Update examples
  - Update parakeet streaming examples to demonstrate trait usage
  - Update vad examples to demonstrate trait usage
  - Show generic processing functions using the trait
  - _Requirements: 5.1, 5.2, 5.3_

- [ ] 4.1 Update streaming_transcribe.rs example
  - Demonstrate using process_loop() method
  - Show mark_finished() usage
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

- [ ] 4.2 Update realtime_speech_recorder.rs example
  - Demonstrate using process_loop() method
  - Show mark_finished() usage
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

- [ ] 5. Documentation
  - Add README.md to frame-processor crate
  - Document the trait design and usage patterns
  - Include examples of implementing the trait
  - Document the async nature and Drop limitations
  - _Requirements: 5.1, 5.2, 5.3, 5.4_
