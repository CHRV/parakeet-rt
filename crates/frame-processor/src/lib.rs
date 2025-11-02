//! Frame-based audio processing trait
//!
//! This crate provides a unified trait for frame-by-frame audio processing with streaming semantics.
//! It is designed to work with ring buffer-based producer-consumer patterns and supports both
//! voice activity detection (VAD) and transcription use cases.
//!
//! # Example
//!
//! ```rust,ignore
//! use frame_processor::FrameProcessor;
//!
//! async fn process_audio<P: FrameProcessor>(mut processor: P) -> Result<(), P::Error> {
//!     // Mark the stream as finished when no more input will be provided
//!     processor.mark_finished();
//!
//!     // Process all frames until completion
//!     processor.process_loop().await?;
//!
//!     Ok(())
//! }
//! ```

use async_trait::async_trait;
use thiserror::Error;

/// Error type for frame processing operations
#[derive(Error, Debug)]
pub enum FrameProcessorError {
    /// A generic processing error occurred
    #[error("Processing error: {0}")]
    ProcessingError(String),

    /// An error occurred with the buffer operations
    #[error("Buffer error: {0}")]
    BufferError(String),

    /// An error occurred in the underlying model
    #[error("Model error: {0}")]
    ModelError(#[from] Box<dyn std::error::Error + Send + Sync>),
}

/// Trait for frame-by-frame audio processing with streaming semantics
///
/// This trait defines a common interface for processing audio frames in a streaming manner.
/// Implementations maintain internal state and process frames one at a time, typically
/// reading from a ring buffer and emitting results to another ring buffer.
///
/// # Lifecycle
///
/// 1. Create the processor instance
/// 2. Feed audio data to the input buffer (typically in a separate thread)
/// 3. Call `process_frame()` repeatedly or use `process_loop()` to process all frames
/// 4. Call `mark_finished()` when no more input will be provided
/// 5. Continue processing until `is_finished()` returns true
/// 6. Call `finalize()` to perform any cleanup operations
///
/// # Example
///
/// ```rust,ignore
/// // Create processor
/// let mut processor = MyProcessor::new();
///
/// // Feed audio in background
/// tokio::spawn(async move {
///     // ... feed audio to input buffer ...
/// });
///
/// // Process frames
/// while !processor.is_finished() {
///     if processor.has_next_frame() {
///         processor.process_frame().await?;
///     } else {
///         tokio::task::yield_now().await;
///     }
/// }
///
/// // Or use the convenience method
/// processor.process_loop().await?;
/// ```
#[async_trait]
pub trait FrameProcessor {
    /// The error type returned by processing operations
    type Error: std::error::Error + Send + Sync + 'static;

    /// Check if another frame is available for processing
    ///
    /// Returns `true` if `process_frame()` can be called to process the next frame.
    /// This should check the internal buffer state to determine if sufficient samples
    /// are available to form a complete frame.
    ///
    /// # Returns
    ///
    /// - `true` if a frame is ready to be processed
    /// - `false` if more input is needed before the next frame can be processed
    fn has_next_frame(&self) -> bool;

    /// Process the next available frame
    ///
    /// This method should:
    /// 1. Extract the next frame from the input buffer
    /// 2. Perform the necessary processing (VAD, transcription, etc.)
    /// 3. Emit results to the output buffer
    /// 4. Update internal state
    ///
    /// This method should only be called when `has_next_frame()` returns `true`.
    ///
    /// # Errors
    ///
    /// Returns an error if frame processing fails for any reason, such as:
    /// - Model inference errors
    /// - Buffer read/write errors
    /// - Invalid state transitions
    async fn process_frame(&mut self) -> Result<(), Self::Error>;

    /// Check if the stream has been marked as finished
    ///
    /// Returns `true` if no more input will be provided and all frames
    /// have been processed. This typically means both:
    /// - `mark_finished()` has been called
    /// - No more frames are available (`has_next_frame()` returns `false`)
    ///
    /// # Returns
    ///
    /// - `true` if the stream is complete and no more processing is needed
    /// - `false` if more frames may become available or need to be processed
    fn is_finished(&self) -> bool;

    /// Mark the stream as finished
    ///
    /// This signals that no more input will be provided. The processor
    /// should prepare to process any remaining buffered frames and then
    /// finalize processing.
    ///
    /// After calling this method, `is_finished()` will return `true` once
    /// all buffered frames have been processed.
    fn mark_finished(&mut self);

    /// Finalize processing after the stream is finished
    ///
    /// This method is called after all frames have been processed to perform
    /// any cleanup or final operations. Implementations can override this
    /// to add custom finalization logic, such as:
    /// - Flushing remaining output
    /// - Processing final state transitions
    /// - Releasing resources
    ///
    /// The default implementation does nothing.
    ///
    /// # Errors
    ///
    /// Returns an error if finalization fails.
    async fn finalize(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }

    /// Process all available frames in a loop
    ///
    /// This default implementation continues processing while the stream
    /// is not finished, checking for available frames before each call
    /// to `process_frame()`. Once the stream is marked as finished and all
    /// frames are processed, it calls `finalize()` and returns.
    ///
    /// This is a convenience method that implements the typical processing loop:
    /// 1. While not finished, process frames as they become available
    /// 2. Yield to other tasks when no frames are available
    /// 3. After the stream is marked finished, process remaining frames
    /// 4. Call `finalize()` to complete processing
    ///
    /// # Errors
    ///
    /// Returns the first error encountered during frame processing or finalization.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Simple usage
    /// processor.mark_finished();
    /// processor.process_loop().await?;
    ///
    /// // Or with background audio feeding
    /// tokio::spawn(async move {
    ///     // ... feed audio ...
    ///     processor.mark_finished();
    /// });
    /// processor.process_loop().await?;
    /// ```
    async fn process_loop(&mut self) -> Result<(), Self::Error> {
        while !self.is_finished() {
            if self.has_next_frame() {
                self.process_frame().await?;
            } else {
                // No frames available yet, yield to allow other tasks to run
                tokio::task::yield_now().await;
            }
        }

        // Process any remaining frames after stream is finished
        while self.has_next_frame() {
            self.process_frame().await?;
        }

        // Finalize processing
        self.finalize().await?;

        Ok(())
    }
}
