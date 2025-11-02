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

1. THE FrameProcessor trait SHALL define a method `has_next_frame` that returns a boolean indicating whether another frame is available for processing
2. THE FrameProcessor trait SHALL define a method `process_frame` that processes the next available frame and returns a Result type
3. THE FrameProcessor trait SHALL define a method `is_finished` that returns a boolean indicating whether the stream has been marked as complete
4. THE FrameProcessor trait SHALL define a default method `process_loop` that continues processing while `is_finished` returns false, checking `has_next_frame` before each `process_frame` call, and calls `finalize` when the loop completes
5. THE FrameProcessor trait SHALL be generic enough to support both VAD and transcription use cases

### Requirement 2

**User Story:** As a developer, I want StreamingParakeetTDT to implement the FrameProcessor trait, so that it follows the standardized processing interface.

#### Acceptance Criteria

1. THE StreamingParakeetTDT struct SHALL implement the FrameProcessor trait with all methods defined in the trait impl block
2. WHEN `has_next_frame` is called, THE StreamingParakeetTDT SHALL delegate to its internal buffer's `has_next_chunk` method
3. WHEN `process_frame` is called, THE StreamingParakeetTDT SHALL process one audio chunk and emit tokens to the output ring buffer
4. WHEN `is_finished` is called, THE StreamingParakeetTDT SHALL return the finalization state of its internal buffer
5. THE trait implementation SHALL encapsulate the existing `process_audio` and `process_next_chunk` logic

### Requirement 3

**User Story:** As a developer, I want StreamingVad to implement the FrameProcessor trait, so that it follows the standardized processing interface.

#### Acceptance Criteria

1. THE StreamingVad struct SHALL implement the FrameProcessor trait with all methods defined in the trait impl block
2. WHEN `has_next_frame` is called, THE StreamingVad SHALL check if sufficient samples are available in its audio consumer
3. WHEN `process_frame` is called, THE StreamingVad SHALL process one frame through the Silero model and update its internal state
4. WHEN `is_finished` is called, THE StreamingVad SHALL return whether the stream has been finalized
5. THE trait implementation SHALL encapsulate the existing `process_audio` and `process_frame` logic

### Requirement 4

**User Story:** As a developer, I want the FrameProcessor trait to support error handling, so that processing errors can be properly propagated and handled.

#### Acceptance Criteria

1. THE `process_frame` method SHALL return a Result type with an associated Error type
2. THE FrameProcessor trait SHALL define an associated Error type that can be customized by implementors
3. WHEN an error occurs during frame processing, THE implementation SHALL return an Err variant with appropriate error information
4. THE `process_loop` default method SHALL propagate errors from `process_frame` to the caller
5. THE error types SHALL be defined using the `thiserror` crate for consistent error handling

### Requirement 5

**User Story:** As a developer, I want a clean and well-defined interface for frame processing, so that the code is readable and maintainable.

#### Acceptance Criteria

1. THE FrameProcessor trait SHALL have clear method signatures with descriptive names
2. THE FrameProcessor trait SHALL include comprehensive documentation comments for all methods
3. THE trait methods SHALL follow Rust naming conventions and idioms
4. WHEN a FrameProcessor instance is dropped, THE implementation SHALL automatically finalize any remaining processing
5. THE trait design SHALL prioritize simplicity and clarity over backward compatibility
