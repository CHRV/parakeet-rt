//
// Copyright (c) 2024–2025, Daily
//
// SPDX-License-Identifier: BSD 2-Clause License
//

//! Silero Voice Activity Detection (VAD) implementation.
//!
//! This module provides a VAD analyzer based on the Silero VAD ONNX model,
//! which can detect voice activity in audio streams with high accuracy.
//! Supports 8kHz and 16kHz sample rates.
//!
//! The Silero VAD model is a pre-trained neural network that can efficiently
//! detect speech segments in audio data. It maintains internal state to provide
//! context-aware detection across audio chunks.

use crate::utils;
use ndarray::{Array, Array2, ArrayBase, ArrayD, Dim, IxDynImpl, OwnedRepr, s};
use ort::{session::Session, value::Tensor};
use std::path::Path;

/// ONNX runtime wrapper for the Silero VAD model.
///
/// Provides voice activity detection using the pre-trained Silero VAD model
/// with ONNX runtime for efficient inference. Handles model state management
/// and input validation for audio processing.
///
/// The model maintains internal state between calls to provide context-aware
/// detection. State should be reset periodically to prevent memory growth.
#[derive(Debug)]
pub struct Silero {
    /// ONNX runtime session for model inference
    session: Session,
    /// Sample rate configuration for the model
    sample_rate: ArrayBase<OwnedRepr<i64>, Dim<[usize; 1]>>,
    /// Internal model state for context-aware detection
    state: ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>,
    /// Context buffer maintaining previous audio samples for continuity
    /// Size: 64 samples for 16kHz, 32 samples for 8kHz
    context: ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>,
    /// Sample rate value for determining context size
    context_size: usize,

    expected_samples: usize,
}

impl Silero {
    /// Initialize the Silero VAD model.
    ///
    /// Creates a new Silero VAD instance with the specified sample rate and model file.
    /// The model is configured for single-threaded execution for optimal performance
    /// in real-time audio processing scenarios.
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Audio sample rate (8kHz or 16kHz)
    /// * `model_path` - Path to the ONNX model file
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the initialized Silero instance or an ONNX runtime error.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use vad::{silero::Silero, utils::SampleRate};
    ///
    /// let vad = Silero::new(SampleRate::SixteenkHz, "models/silero_vad.onnx")?;
    /// # Ok::<(), ort::Error>(())
    /// ```
    pub fn new(
        sample_rate: utils::SampleRate,
        model_path: impl AsRef<Path>,
    ) -> Result<Self, ort::Error> {
        // Configure ONNX session for single-threaded execution
        // This provides optimal performance for real-time audio processing
        let session = Session::builder()?
            .with_inter_threads(1)?
            .with_intra_threads(1)?
            .commit_from_file(model_path)?;

        // Initialize model state: [2, batch_size=1, hidden_size=128]
        let state = ArrayD::<f32>::zeros([2, 1, 128].as_slice());

        let sample_rate_value: i64 = sample_rate.into();

        // Initialize context buffer based on sample rate
        // 64 samples for 16kHz, 32 samples for 8kHz
        let context_size = if sample_rate_value == 16000 { 64 } else { 32 };
        let context = ArrayD::<f32>::zeros([1, context_size].as_slice());

        // Convert sample rate to the format expected by the model
        let sample_rate = Array::from_shape_vec([1], vec![sample_rate_value]).unwrap();

        let expected_samples = if sample_rate_value == 16000 { 512 } else { 256 };

        Ok(Self {
            session,
            sample_rate,
            state,
            context,
            context_size,
            expected_samples, // Expected frame size
        })
    }

    /// Reset the internal model state.
    ///
    /// Clears the model's internal state, which is useful for processing
    /// new audio streams or preventing memory growth during long-running
    /// detection sessions. Should be called periodically in streaming
    /// applications.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use vad::{silero::Silero, utils::SampleRate};
    /// # let mut vad = Silero::new(SampleRate::SixteenkHz, "models/silero_vad.onnx")?;
    /// // Reset state when starting a new audio stream
    /// vad.reset();
    /// # Ok::<(), ort::Error>(())
    /// ```
    pub fn reset(&mut self) {
        // Reset to initial state: [2, batch_size=1, hidden_size=128]
        self.state = ArrayD::<f32>::zeros([2, 1, 128].as_slice());

        // Reset context buffer based on sample rate
        self.context = ArrayD::<f32>::zeros([1, self.context_size].as_slice());
    }

    /// Calculate voice activity confidence for the given audio frame.
    ///
    /// Processes an audio frame through the Silero VAD model to determine
    /// the likelihood of voice activity. Handles variable input sizes by
    /// processing them in chunks of the expected frame size:
    /// - 16kHz: 512 samples (32ms)
    /// - 8kHz: 256 samples (32ms)
    ///
    /// For inputs larger than the expected frame size, processes multiple
    /// chunks and returns the maximum confidence score.
    ///
    /// # Arguments
    ///
    /// * `audio_frame` - Audio samples as 16-bit signed integers
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the voice confidence score (0.0 to 1.0)
    /// or an ONNX runtime error. Higher values indicate higher likelihood
    /// of voice activity.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use vad::{silero::Silero, utils::SampleRate};
    /// # let mut vad = Silero::new(SampleRate::SixteenkHz, "models/silero_vad.onnx")?;
    /// let audio_samples: Vec<i16> = vec![0; 1024]; // Can handle various sizes
    /// let confidence = vad.calc_level(&audio_samples)?;
    ///
    /// if confidence > 0.5 {
    ///     println!("Voice detected with confidence: {:.2}", confidence);
    /// }
    /// # Ok::<(), ort::Error>(())
    /// ```
    pub fn calc_level(&mut self, audio_frame: &[i16]) -> Result<f32, ort::Error> {
        // Determine expected frame size based on sample rate

        // Handle variable input sizes by processing in chunks
        if audio_frame.len() < self.expected_samples {
            // Input too small - pad with zeros
            let mut padded_frame = vec![0i16; self.expected_samples];
            padded_frame[..audio_frame.len()].copy_from_slice(audio_frame);
            return self.process_single_frame(&padded_frame);
        } else if audio_frame.len() == self.expected_samples {
            // Perfect size - process directly
            return self.process_single_frame(audio_frame);
        } else {
            // Input larger than expected - process in chunks and return max confidence
            let mut max_confidence = 0.0f32;
            let chunks = audio_frame.chunks(self.expected_samples);

            for chunk in chunks {
                let confidence = if chunk.len() == self.expected_samples {
                    self.process_single_frame(chunk)?
                } else {
                    // Last chunk might be smaller - pad it
                    let mut padded_chunk = vec![0i16; self.expected_samples];
                    padded_chunk[..chunk.len()].copy_from_slice(chunk);
                    self.process_single_frame(&padded_chunk)?
                };
                max_confidence = max_confidence.max(confidence);
            }

            return Ok(max_confidence);
        }
    }

    /// Process a single audio frame of the expected size.
    ///
    /// Internal method that handles the core VAD processing for frames
    /// of exactly the right size (512 samples for 16kHz, 256 for 8kHz).
    fn process_single_frame(&mut self, audio_frame: &[i16]) -> Result<f32, ort::Error> {
        // Convert 16-bit signed integers to float32 in range [-1.0, 1.0]
        // This normalization is required by the Silero model
        let data = audio_frame
            .iter()
            .map(|x| (*x as f32) / (i16::MAX as f32))
            .collect::<Vec<_>>();

        // Create new audio frame array
        let new_frame = Array2::<f32>::from_shape_vec([1, data.len()], data).unwrap();

        // Concatenate context with new frame (similar to Python implementation)
        // This provides continuity between audio chunks for better detection
        let input_frame = if self.context.shape()[1] == 0 {
            // First frame - no context available yet
            new_frame
        } else {
            // Concatenate context + new frame
            let context_2d = self
                .context
                .clone()
                .into_dimensionality::<ndarray::Ix2>()
                .unwrap();
            ndarray::concatenate![ndarray::Axis(1), context_2d, new_frame]
        };

        // Prepare model inputs
        let inps = ort::inputs![
            "input" => Tensor::from_array(input_frame.clone())?,
            "state" => Tensor::from_array(std::mem::take(&mut self.state))?,
            "sr" => Tensor::from_array(self.sample_rate.clone().into_dyn())?,
        ];

        // Run inference
        let res = self.session.run(inps)?;

        // Update internal state for next inference
        self.state = res["stateN"].try_extract_array().unwrap().to_owned();

        // Update context buffer with the last context_size samples from input
        // This maintains continuity for the next frame
        let input_shape = input_frame.shape();
        let total_samples = input_shape[1];
        if total_samples >= self.context_size {
            let start_idx = total_samples - self.context_size;
            self.context = input_frame.slice(s![.., start_idx..]).to_owned().into_dyn();
        }

        // Extract and return confidence score
        Ok(*res["output"]
            .try_extract_tensor::<f32>()
            .unwrap()
            .1
            .first()
            .unwrap())
    }
}
