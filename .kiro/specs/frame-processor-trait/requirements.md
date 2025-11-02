# Requirements Document

## Introduction

This document specifies the requirements for refactoring the streaming audio processing components in the Parakeet and VAD crates to use a unified `FrameProcessor` trait. Currently, both `StreamingParakeetTDT` and `StreamingVad` implement similar frame-by-frame processing patterns with ring buffers, but without a common abstraction. The refactoring will introduce a trait that standardizes the frame processing interface, making the code more maintainable, testable, and extensible.

## Glossary

- **FrameProcessor**: A trait that defines the interface for processing audio frames in a streaming manner
- **StreamingParakeetTDT**: The existing streaming transcription engine for Parakeet TDT models
- **StreamingVad**: The existing streaming voice activity detection engine
- **RingBuffer**: A lock-free circular buffer from the `rtrb` crate used for producer-consumer patterns
- **Frame**: A fixed-size chunk of audio samples processed as a unit
- **AudioBuffer**: A component that manages audio context and extracts frames for processing
- **ProcessingState**: Internal state maintained across frame processing iterations

## Requirements

### Requirement 1

**User Story:** As a developer, I want a common trait for frame-based audio processing, so that I can write generic code that works with different streaming processors.

#### Acceptance Criteria

1. THE FrameProcessor trait SHALL provide a method to check whether another frame is available for processing
2. THE FrameProcessor trait SHALL provide a method to process the next available frame
3. THE FrameProcessor trait SHALL provide a method to check whether the stream has been marked as complete
4. THE FrameProcessor trait SHALL provide a method to mark the stream as finished
5. WHEN the processing loop executes, THE FrameProcessor trait SHALL continue processing while frames are available
6. WHEN all frames are processed, THE FrameProcessor trait SHALL invoke finalization logic
7. THE FrameProcessor trait SHALL support both voice activity detection and transcription processing use cases

### Requirement 2

**User Story:** As a developer, I want StreamingParakeetTDT to implement the FrameProcessor trait, so that it follows the standardized processing interface.

#### Acceptance Criteria

1. THE StreamingParakeetTDT struct SHALL implement the FrameProcessor trait
2. WHEN frame availability is checked, THE StreamingParakeetTDT SHALL query its internal buffer state
3. WHEN a frame is processed, THE StreamingParakeetTDT SHALL process one audio chunk and emit tokens to the output buffer
4. WHEN stream completion is checked, THE StreamingParakeetTDT SHALL return the finalization state of its internal buffer
5. THE StreamingParakeetTDT SHALL maintain compatibility with existing audio processing logic

### Requirement 3

**User Story:** As a developer, I want StreamingVad to implement the FrameProcessor trait, so that it follows the standardized processing interface.

#### Acceptance Criteria

1. THE StreamingVad struct SHALL implement the FrameProcessor trait
2. WHEN frame availability is checked, THE StreamingVad SHALL verify sufficient samples are available in its audio consumer
3. WHEN a frame is processed, THE StreamingVad SHALL process one frame through the voice activity detection model
4. WHEN a frame is processed, THE StreamingVad SHALL update its internal state based on detection results
5. WHEN stream completion is checked, THE StreamingVad SHALL return whether the stream has been finalized
6. THE StreamingVad SHALL maintain compatibility with existing audio processing logic

### Requirement 4

**User Story:** As a developer, I want the FrameProcessor trait to support error handling, so that processing errors can be properly propagated and handled.

#### Acceptance Criteria

1. THE FrameProcessor trait SHALL define an associated Error type that implementors can customize
2. WHEN a frame is processed, THE FrameProcessor trait SHALL return a Result type indicating success or failure
3. WHEN an error occurs during frame processing, THE implementation SHALL return error information to the caller
4. WHEN the processing loop encounters an error, THE FrameProcessor trait SHALL propagate the error to the caller
5. THE error types SHALL provide consistent error handling across implementations

### Requirement 5

**User Story:** As a developer, I want a clean and well-defined interface for frame processing, so that the code is readable and maintainable.

#### Acceptance Criteria

1. THE FrameProcessor trait SHALL provide method signatures with descriptive names
2. THE FrameProcessor trait SHALL include documentation comments for all methods
3. THE FrameProcessor trait SHALL follow Rust naming conventions and idioms
4. WHEN a FrameProcessor instance is dropped, THE implementation SHALL mark the stream as finished
5. THE FrameProcessor trait SHALL prioritize simplicity and clarity in its design

### Requirement 6

**User Story:** As a user of the parakeet-mic application, I want the audio recording, processing, and output to run concurrently, so that the system can handle real-time audio efficiently.

#### Acceptance Criteria

1. THE parakeet-mic application SHALL use the FrameProcessor trait for audio processing
2. THE parakeet-mic application SHALL run audio recording in a dedicated thread
3. THE parakeet-mic application SHALL run frame processing in a dedicated thread
4. THE parakeet-mic application SHALL run output printing in a dedicated thread
5. WHEN audio is recorded, THE recording thread SHALL push samples to a ring buffer for the processing thread
6. WHEN frames are processed, THE processing thread SHALL emit results to a ring buffer for the output thread
7. THE parakeet-mic application SHALL coordinate thread lifecycle and graceful shutdown

### Requirement 7

**User Story:** As a developer, I want the streaming processors to automatically detect when ring buffer producers or consumers are dropped, so that the system can gracefully handle disconnections without explicit signaling.

#### Acceptance Criteria

1. WHEN the audio producer is dropped, THE StreamingParakeetTDT SHALL detect abandonment using the consumer's is_abandoned method
2. WHEN the audio producer is dropped, THE StreamingVad SHALL detect abandonment using the consumer's is_abandoned method
3. WHEN abandonment is detected, THE FrameProcessor implementation SHALL automatically mark the stream as finished
4. WHEN the token consumer is abandoned, THE StreamingParakeetTDT SHALL detect abandonment using the producer's is_abandoned method
5. WHEN the speech consumer is abandoned, THE StreamingVad SHALL detect abandonment using the producer's is_abandoned method
6. WHEN output consumer abandonment is detected, THE FrameProcessor implementation SHALL stop emitting results and prepare for shutdown
7. THE abandonment detection SHALL eliminate the need for explicit mark_finished calls in normal shutdown scenarios

### Requirement 8

**User Story:** As a developer, I want output producers to be automatically dropped when input or output streams are abandoned, so that shutdown signals propagate through the entire processing pipeline.

#### Acceptance Criteria

1. THE StreamingParakeetTDT struct SHALL store the token producer as an Option type
2. THE StreamingVad struct SHALL store the speech producer as an Option type
3. WHEN the token consumer is detected as abandoned, THE StreamingParakeetTDT SHALL drop its token producer by setting it to None
4. WHEN the speech consumer is detected as abandoned, THE StreamingVad SHALL drop its speech producer by setting it to None
5. WHEN the audio consumer is detected as abandoned at the end of processing, THE StreamingParakeetTDT SHALL drop its token producer by setting it to None
6. WHEN the audio consumer is detected as abandoned at the end of processing, THE StreamingVad SHALL drop its speech producer by setting it to None
7. THE automatic producer dropping SHALL create bidirectional shutdown signaling through the processing pipeline
