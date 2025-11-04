# Implementation Plan

- [x] 1. Update dependencies and migrate error handling to thiserror
  - Add `thiserror = "2.0"` to `crates/parakeet/Cargo.toml` dependencies
  - Verify `tracing = "0.1"` is present in dependencies
  - _Requirements: 4.1, 4.2_

- [x] 1.1 Refactor error.rs to use thiserror
  - Replace `#[derive(Debug)]` with `#[derive(Debug, Error)]` on Error enum
  - Add `use thiserror::Error;` import
  - Add `#[error("IO error: {0}")]` attribute to `Io` variant
  - Add `#[from]` attribute to `Io(#[from] std::io::Error)` variant
  - Add `#[error("ONNX Runtime error: {0}")]` attribute to `Ort` variant
  - Add `#[from]` attribute to `Ort(#[from] ort::Error)` variant
  - Add `#[error("Audio processing error: {0}")]` attribute to `Audio` variant
  - Add `#[error("Model error: {0}")]` attribute to `Model` variant
  - Add `#[error("Tokenizer error: {0}")]` attribute to `Tokenizer` variant
  - Add `#[error("Config error: {0}")]` attribute to `Config` variant
  - Remove manual `impl fmt::Display for Error` block
  - Remove manual `impl std::error::Error for Error` block
  - Remove manual `impl From<std::io::Error> for Error` block
  - Remove manual `impl From<ort::Error> for Error` block
  - Keep manual `impl From<serde_json::Error> for Error` (cannot use #[from] due to String conversion)
  - Change `#[cfg(test)]` to `#[cfg(feature = "audio")]` for `impl From<hound::Error>`
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 4.3, 4.4, 4.5_

- [x] 2. Add tracing instrumentation to vocab module
  - Add `use tracing;` import to `crates/parakeet/src/vocab.rs`
  - Add `#[tracing::instrument]` to `from_file` function
  - Add `tracing::debug!` event after opening file with path display
  - Add `tracing::info!` event after successful vocabulary loading with vocab size
  - Add `tracing::error!` event before returning Config errors in `from_file`
  - _Requirements: 2.4, 2.5, 3.3, 3.4_

- [x] 3. Add tracing instrumentation to decoder module
  - Add `use tracing;` import to `crates/parakeet/src/decoder.rs`
  - Add `#[tracing::instrument(skip(self, tokens, frame_indices, _durations))]` to `decode_with_timestamps`
  - Add `tracing::debug!` event at start of `decode_with_timestamps` with token count
  - Add `tracing::debug!` event after decoding with final transcription length
  - _Requirements: 2.3, 2.5, 3.4_

- [x] 4. Add tracing instrumentation to model module
  - Add `use tracing;` import to `crates/parakeet/src/model.rs`
  - _Requirements: 2.1, 2.5_

- [x] 4.1 Instrument model loading functions
  - Add `#[tracing::instrument]` to `from_pretrained` function
  - Add `tracing::debug!` event in `find_preprocessor` when file is found
  - Add `tracing::debug!` event in `find_encoder` when file is found
  - Add `tracing::debug!` event in `find_decoder_joint` when file is found
  - Add `tracing::error!` events before returning Config errors in find functions
  - Add `tracing::info!` event after successful model loading in `from_pretrained`
  - _Requirements: 2.1, 2.5, 3.3, 3.4_

- [x] 4.2 Instrument inference pipeline functions
  - Add `#[tracing::instrument(skip(self, waves, waves_lens))]` to `forward` function
  - Add `tracing::debug!` event at start of `forward` with input shape information
  - Add `#[tracing::instrument(skip(self, wave, lens))]` to `preprocess` function
  - Add `tracing::trace!` event before preprocessor inference in `preprocess`
  - Add `tracing::trace!` event after preprocessor inference in `preprocess`
  - Add `tracing::error!` events before returning Model errors in `preprocess`
  - Add `#[tracing::instrument(skip(self, features, features_lens))]` to `encode` function
  - Add `tracing::trace!` event before encoder inference in `encode`
  - Add `tracing::trace!` event after encoder inference in `encode`
  - Add `tracing::debug!` event with output shape information in `encode`
  - Add `tracing::error!` events before returning Model errors in `encode`
  - _Requirements: 2.1, 2.5, 3.2, 3.3, 3.4, 3.5_

- [x] 4.3 Instrument decoding functions
  - Add `#[tracing::instrument(skip(self, tokens, prev_state, encoding))]` to `decode` function
  - Add `tracing::trace!` event before decoder_joint inference in `decode`
  - Add `tracing::trace!` event after decoder_joint inference in `decode`
  - Add `tracing::error!` events before returning Model errors in `decode`
  - Add `#[tracing::instrument(skip(self, encoder_out, encoder_out_lens))]` to `decoding` function
  - Add `tracing::debug!` event when token is emitted in `decoding` loop
  - Add `tracing::info!` event at end of `decoding` with final token count
  - Add `tracing::error!` events before returning Model errors in `decoding`
  - _Requirements: 2.1, 2.5, 3.1, 3.3, 3.4, 3.5_

- [x] 5. Add tracing instrumentation to streaming module
  - Add `use tracing;` import to `crates/parakeet/src/streaming.rs`
  - _Requirements: 2.2, 2.5_

- [x] 5.1 Instrument streaming initialization and lifecycle
  - Add `tracing::debug!` event in `new_with_vocab` with buffer sizes and context configuration
  - Add `tracing::debug!` event in `reset` function
  - Add `tracing::info!` event in `finalize` function
  - _Requirements: 2.2, 2.5, 3.4_

- [x] 5.2 Instrument streaming audio processing
  - Add `#[tracing::instrument(skip(self))]` to `process_audio` function
  - Add `tracing::debug!` event with chunk count after processing in `process_audio`
  - Add `#[tracing::instrument(skip(self))]` to `process_next_chunk` function
  - Add `tracing::trace!` event at start of `process_next_chunk`
  - Add `tracing::trace!` event before preprocessing in `process_next_chunk`
  - Add `tracing::trace!` event before encoding in `process_next_chunk`
  - Add `tracing::debug!` event with frame processing progress in `process_next_chunk` loop
  - Add `tracing::debug!` event with emitted token count at end of `process_next_chunk`
  - Add `#[tracing::instrument(skip(self, encoding, tokens))]` to `decode_frame` function
  - Add `tracing::trace!` event in `decode_frame` before calling model decode
  - _Requirements: 2.2, 2.5, 3.1, 3.2, 3.4, 3.5_

- [x] 6. Verify implementation and test
  - Run `cargo build` in `crates/parakeet` to verify compilation
  - Run `cargo test` in `crates/parakeet` to verify existing tests pass
  - Check that error messages display correctly with thiserror formatting
  - _Requirements: 1.5, 4.1, 4.2_
