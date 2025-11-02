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

- [x] 3. Update silero crate
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

- [x] 4. Refactor parakeet-mic to use multi-threaded architecture
  - Implement three-thread design: recording, processing, and output
  - Use FrameProcessor trait for audio processing
  - Add shutdown coordination with Arc<AtomicBool>
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7_

- [x] 4.1 Implement recording thread
  - Create recording_thread function that captures audio from microphone
  - Initialize CPAL audio input stream
  - Push audio samples to ring buffer producer
  - Check shutdown signal periodically
  - _Requirements: 6.2, 6.5_

- [x] 4.2 Implement processing thread
  - Create async processing_thread function that uses FrameProcessor trait
  - Process frames in a loop until shutdown signal
  - Call mark_finished and process_loop on shutdown
  - Handle errors appropriately
  - _Requirements: 6.1, 6.3, 6.6_

- [x] 4.3 Implement output thread
  - Create output_thread function that consumes tokens from ring buffer
  - Decode tokens using vocabulary
  - Print transcription results to console
  - Drain remaining tokens on shutdown
  - _Requirements: 6.4, 6.6_

- [x] 4.4 Update main.rs with thread orchestration
  - Create ring buffers for audio and tokens
  - Setup Ctrl+C handler with shutdown signal
  - Spawn all three threads
  - Wait for all threads to complete and collect errors
  - _Requirements: 6.7_

- [ ]* 5. Update realtime_speech_recorder.rs example
  - Refactor to use process_loop() method instead of manual process_audio() calls
  - Add mark_finished() call when Ctrl+C is received
  - Demonstrate the trait-based processing pattern
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 5.1, 5.2, 5.3_

- [ ]* 6. Add README.md to frame-processor crate
  - Document the trait design and usage patterns
  - Include examples of implementing the trait for custom processors
  - Show usage examples with both parakeet and silero implementations
  - Document the async nature and Drop limitations
  - Explain the lifecycle: create → feed audio → process → mark_finished → finalize
  - _Requirements: 5.1, 5.2, 5.3, 5.4_

- [x] 7. Make token_producer optional in StreamingParakeetTDT
  - Change token_producer field to Option<Producer<TokenResult>>
  - Update new() method to wrap token producer in Some()
  - Update process_next_chunk to handle optional producer when pushing tokens
  - Implement close_output() method to drop token producer
  - _Requirements: 8.1, 8.5, 8.7_

- [x] 8. Make speech_producer optional in StreamingVad
  - Change speech_producer field to Option<Producer<f32>>
  - Update new() method to wrap speech producer in Some()
  - Update process_frame_sync to handle optional producer when pushing speech samples
  - Implement close_output() method to drop speech producer
  - _Requirements: 8.2, 8.5, 8.7_

- [x] 9. Implement bidirectional abandonment detection for StreamingParakeetTDT
  - Modify has_next_frame to check audio_consumer.is_abandoned()
  - When audio producer is abandoned, process remaining buffered samples
  - Modify process_frame to check token_producer.is_abandoned() (downstream abandonment)
  - When token consumer is abandoned, drop token_producer and mark stream as finished
  - At end of process_frame, check if audio_consumer.is_abandoned() and no more frames (upstream abandonment)
  - When input is exhausted, drop token_producer to signal downstream
  - Update tests to verify bidirectional abandonment detection behavior
  - _Requirements: 7.1, 7.3, 7.4, 8.3, 8.5, 8.7_

- [x] 10. Implement bidirectional abandonment detection for StreamingVad
  - Modify has_next_frame to check audio_consumer.is_abandoned()
  - When audio producer is abandoned, process remaining buffered frames
  - Modify process_frame to check speech_producer.is_abandoned() (downstream abandonment)
  - When speech consumer is abandoned, drop speech_producer and mark stream as finished
  - At end of process_frame, check if audio_consumer.is_abandoned() and no more frames (upstream abandonment)
  - When input is exhausted, drop speech_producer to signal downstream
  - Update tests to verify bidirectional abandonment detection behavior
  - _Requirements: 7.2, 7.3, 7.5, 8.4, 8.6, 8.7_

- [x] 11. Update parakeet-mic to leverage abandonment detection
  - Remove explicit mark_finished calls where abandonment detection handles it
  - Simplify shutdown coordination by relying on automatic detection
  - Update documentation to explain the automatic cleanup behavior
  - Test graceful shutdown scenarios with thread exits
  - _Requirements: 7.6, 7.7_
