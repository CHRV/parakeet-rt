use crate::error::Result;
use crate::model::{ParakeetTDTModel, State};
use crate::vocab::Vocabulary;
use async_trait::async_trait;
use frame_processor::FrameProcessor;
use ndarray::{Array1, Array2, Array3};
use rtrb::{Consumer, Producer, RingBuffer};
use tracing;

/// Context configuration for streaming inference
#[derive(Debug, Clone)]
pub struct ContextConfig {
    /// Left context in samples (improves quality, doesn't affect latency)
    pub left_samples: usize,
    /// Chunk size in samples (affects latency)
    pub chunk_samples: usize,
    /// Right context in samples (affects latency)
    pub right_samples: usize,
}

impl ContextConfig {
    /// Create new context config with time-based parameters
    pub fn new(left_secs: f32, chunk_secs: f32, right_secs: f32, sample_rate: usize) -> Self {
        Self {
            left_samples: (left_secs * sample_rate as f32) as usize,
            chunk_samples: (chunk_secs * sample_rate as f32) as usize,
            right_samples: (right_secs * sample_rate as f32) as usize,
        }
    }

    /// Total context size in samples
    pub fn total_samples(&self) -> usize {
        self.left_samples + self.chunk_samples + self.right_samples
    }

    /// Theoretical latency in seconds
    pub fn latency_secs(&self, sample_rate: usize) -> f32 {
        (self.chunk_samples + self.right_samples) as f32 / sample_rate as f32
    }
}

/// Audio buffer for streaming inference with context management
pub struct StreamingAudioBuffer {
    /// Ring buffer consumer for reading audio samples
    pub(crate) audio_consumer: Consumer<f32>,
    /// Context configuration
    context: ContextConfig,
    /// Whether we've reached the end of the stream
    pub(crate) is_finished: bool,
    /// Internal buffer to maintain context across chunks
    context_buffer: Vec<f32>,
}

impl StreamingAudioBuffer {
    /// Create new streaming buffer with audio consumer
    pub fn new(context: ContextConfig, audio_consumer: Consumer<f32>) -> Self {
        let context_len = context.left_samples + context.right_samples;
        Self {
            audio_consumer,
            context,
            is_finished: false,
            context_buffer: vec![0.0f32; context_len],
        }
    }

    /// Mark the stream as finished (no more samples will be added)
    pub fn finish(&mut self) {
        self.is_finished = true;
    }

    /// Check if there's enough data for the next chunk
    pub fn has_next_chunk(&self) -> bool {
        let available = self.audio_consumer.slots();

        if self.is_finished {
            // If finished, we can process any remaining samples
            available > 0 || !self.context_buffer.is_empty()
        } else {
            // Need at least chunk_samples + right_samples available
            // The left context comes from our internal buffer
            available >= self.context.chunk_samples + self.context.right_samples
        }
    }

    /// Get the next chunk with context for processing
    pub fn get_next_chunk(&mut self) -> Option<(Array2<f32>, usize)> {
        if !self.has_next_chunk() {
            return None;
        }

        let available_samples = self.audio_consumer.slots();

        // Determine how many new samples to read
        let new_samples_needed = if self.is_finished {
            available_samples // Read all remaining
        } else {
            self.context.chunk_samples + self.context.right_samples
        };

        // Read new samples from the ring buffer
        let mut new_samples = Vec::with_capacity(new_samples_needed);
        for _ in 0..new_samples_needed {
            if let Ok(sample) = self.audio_consumer.pop() {
                new_samples.push(sample);
            } else {
                break;
            }
        }

        // Build the complete frame with context
        let mut frame_data = Vec::new();

        // Add left context from our buffer
        let left_context_start = self
            .context_buffer
            .len()
            .saturating_sub(self.context.left_samples);
        frame_data.extend_from_slice(&self.context_buffer[left_context_start..]);

        // Add new samples
        frame_data.extend_from_slice(&new_samples);

        if frame_data.is_empty() {
            return None;
        }

        // Determine the actual chunk size (excluding context)
        let chunk_length = if self.is_finished {
            // When finished, process all remaining samples
            new_samples.len()
        } else {
            // Normal operation: chunk_samples
            self.context.chunk_samples
        };

        // Update context buffer for next iteration
        // Keep the last (left_samples + chunk_samples) for the next chunk's left context
        let keep_size = self.context.left_samples + self.context.chunk_samples;
        if frame_data.len() > keep_size {
            self.context_buffer = frame_data[frame_data.len() - keep_size..].to_vec();
        } else {
            self.context_buffer = frame_data.clone();
        }

        // Create batch array [1, samples] with full context
        let audio_array = Array2::from_shape_vec((1, frame_data.len()), frame_data)
            .map_err(|_| ())
            .ok()?;

        Some((audio_array, chunk_length))
    }

    /// Reset buffer for new stream
    pub fn reset(&mut self) {
        // Clear ring buffer by consuming all available samples

        let context_len = self.context.left_samples + self.context.right_samples;

        while self.audio_consumer.pop().is_ok() {}
        self.is_finished = false;
        self.context_buffer.clear();
        self.context_buffer.resize(context_len, 0.0);
    }
}

/// Token with timestamp information
#[derive(Debug, Clone)]
pub struct TokenResult {
    pub token_id: i32,
    pub timestamp: usize,
    pub confidence: f32,
    pub text: Option<String>,
}

/// Streaming inference engine for Parakeet TDT
pub struct StreamingParakeetTDT {
    /// The underlying TDT model
    model: ParakeetTDTModel,
    /// Context configuration
    context: ContextConfig,
    /// Ring buffer producer for outputting transcription tokens (optional for bidirectional shutdown)
    token_producer: Option<Producer<TokenResult>>,
    /// Audio buffer for context management
    buffer: StreamingAudioBuffer,
    /// Current decoder state
    state: Option<State>,
    /// Vocabulary for token decoding
    vocab: Option<Vocabulary>,
    /// Input buffer size
    rx_buffer_size: usize,
    /// Output buffer size
    tx_buffer_size: usize,

    previous_token: i32,
}

impl StreamingParakeetTDT {
    /// Create new streaming inference engine
    /// Returns (engine, audio_producer, token_consumer)
    pub fn new(
        model: ParakeetTDTModel,
        context: ContextConfig,
    ) -> (Self, Producer<f32>, Consumer<TokenResult>) {
        Self::new_with_vocab(model, context, None)
    }

    /// Create new streaming inference engine with vocabulary
    /// Returns (engine, audio_producer, token_consumer)
    pub fn new_with_vocab(
        model: ParakeetTDTModel,
        context: ContextConfig,
        vocab: Option<Vocabulary>,
    ) -> (Self, Producer<f32>, Consumer<TokenResult>) {
        // Buffer sizes optimized for real-time processing
        let rx_buffer_size = context.total_samples() * 2; // 2 seconds of input audio buffer
        let tx_buffer_size = 1024; // Buffer for 1000 tokens

        tracing::debug!(
            left_samples = context.left_samples,
            chunk_samples = context.chunk_samples,
            right_samples = context.right_samples,
            rx_buffer_size = rx_buffer_size,
            tx_buffer_size = tx_buffer_size,
            "Initializing streaming engine with buffer sizes and context configuration"
        );

        // Create ring buffers for audio input and token output
        let (audio_producer, audio_consumer) = RingBuffer::new(rx_buffer_size);
        let (token_producer, token_consumer) = RingBuffer::new(tx_buffer_size);
        let previous_token = model.config.blank_idx as i32;

        let engine = Self {
            model,
            context: context.clone(),
            token_producer: Some(token_producer),
            buffer: StreamingAudioBuffer::new(context, audio_consumer),
            state: None,
            vocab,
            rx_buffer_size,
            tx_buffer_size,
            previous_token,
        };

        (engine, audio_producer, token_consumer)
    }

    /// Process audio from the input ring buffer
    /// This should be called regularly to process incoming audio
    #[tracing::instrument(skip(self))]
    pub async fn process_audio(&mut self) -> Result<()> {
        // Process available chunks and emit tokens
        // Uses trait methods internally for consistency
        let mut chunk_count = 0;
        while self.has_next_frame() {
            self.process_frame().await?;
            chunk_count += 1;
        }

        tracing::debug!(chunk_count = chunk_count, "Processed audio chunks");
        Ok(())
    }

    /// Mark the audio stream as finished
    pub async fn finalize(&mut self) {
        tracing::info!("Finalizing streaming engine");
        self.buffer.finish();

        // Process any remaining audio
        let _ = self.process_audio().await;
    }

    /// Process a single chunk and return new tokens with timestamps
    #[tracing::instrument(skip(self))]
    async fn process_next_chunk(&mut self) -> Result<Vec<(i32, usize)>> {
        tracing::trace!("Processing next audio chunk");

        let (audio_chunk, chunk_length) = match self.buffer.get_next_chunk() {
            Some(chunk) => chunk,
            None => return Ok(Vec::new()),
        };

        // Store values we need later before moving them
        let audio_chunk_shape = audio_chunk.shape()[1];
        let audio_lengths = Array1::from_vec(vec![audio_chunk.shape()[1] as i64]);
        let audio_lengths_val = audio_lengths[0];

        // Run preprocessing
        tracing::trace!("Running preprocessing");
        let (features, features_len) = self.model.preprocess(audio_chunk, audio_lengths).await?;

        // Run encoder
        tracing::trace!("Running encoder");
        let (encoder_out, encoder_len) = self.model.encode(features, features_len).await?;

        // Process only the chunk portion (excluding context)
        let mut new_tokens = Vec::new();
        //let mut chunk_tokens = Vec::new(); // Local token history for this chunk

        // Calculate the actual left context size in the current audio chunk
        let actual_left_samples = if audio_chunk_shape >= self.context.left_samples + chunk_length {
            self.context.left_samples
        } else {
            // For early chunks, we might have less left context
            audio_chunk_shape.saturating_sub(chunk_length)
        };

        let start_frame =
            (actual_left_samples * encoder_len[0] as usize) / audio_lengths_val as usize;
        let chunk_frames = (chunk_length * encoder_len[0] as usize) / audio_chunk_shape;
        let end_frame = std::cmp::min(start_frame + chunk_frames, encoder_len[0] as usize);

        // Process frames for this chunk only
        for frame_idx in start_frame.saturating_sub(2)..end_frame {
            if frame_idx >= encoder_out.shape()[1] {
                break;
            }

            tracing::debug!(
                frame_idx = frame_idx,
                start_frame = start_frame,
                end_frame = end_frame,
                "Processing frame"
            );

            // Get encoding at this frame - encoder_out shape is (batch, time, features)
            let encoding = encoder_out.slice(ndarray::s![0, frame_idx, ..]).to_owned();

            // Decode this frame
            let (probs, _step, new_state) =
                self.decode_frame(encoding, &[self.previous_token]).await?;

            // Get token with highest probability
            let token = probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(self.model.config.blank_idx) as i32;

            // Calculate frame index for timestamp (decoder expects frame indices)
            let timestamp = frame_idx;

            if token != self.model.config.blank_idx as i32 {
                self.state = Some(new_state);
                // chunk_tokens.push(token);
                new_tokens.push((token, timestamp));
                self.previous_token = token;

                // Send tokens to output ring buffer
                let text = self
                    .vocab
                    .as_ref()
                    .and_then(|v| v.decode_token(token))
                    .map(|s| s.to_string());

                let token_result = TokenResult {
                    token_id: token,
                    timestamp: frame_idx,
                    confidence: 1.0, // TODO: Add confidence calculation
                    text,
                };

                // Push token to output if producer is available
                if let Some(producer) = &mut self.token_producer {
                    if producer.push(token_result).is_err() {
                        // Output buffer full, could log warning
                        break;
                    }
                }
            }
        }

        tracing::debug!(
            emitted_tokens = new_tokens.len(),
            "Emitted tokens from chunk"
        );
        Ok(new_tokens)
    }

    /// Decode a single frame
    #[tracing::instrument(skip(self, encoding, tokens))]
    async fn decode_frame(
        &mut self,
        encoding: Array1<f32>,
        tokens: &[i32],
    ) -> Result<(Array1<f32>, usize, State)> {
        let prev_state = self.state.clone().unwrap_or_else(|| State {
            h: Array3::<f32>::zeros((2, 1, 640)),
            c: Array3::<f32>::zeros((2, 1, 640)),
        });

        tracing::trace!("Calling model decode");
        self.model.decode(tokens, prev_state, encoding).await
    }

    /// Reset the streaming state
    pub fn reset(&mut self) {
        tracing::debug!("Resetting streaming state");
        self.buffer.reset();
        // Note: We can't clear the output buffer from here since we don't have access to the consumer
        self.state = None;
    }

    /// Get buffer sizes for monitoring
    pub fn rx_buffer_size(&self) -> usize {
        self.rx_buffer_size
    }

    pub fn tx_buffer_size(&self) -> usize {
        self.tx_buffer_size
    }

    /// Drop the token producer to signal end of output to downstream
    /// This enables bidirectional shutdown signaling through the processing pipeline
    pub fn close_output(&mut self) {
        self.token_producer = None;
    }
}

/// Implementation of FrameProcessor trait for StreamingParakeetTDT
#[async_trait]
impl FrameProcessor for StreamingParakeetTDT {
    type Error = crate::error::Error;

    fn has_next_frame(&self) -> bool {
        // Check for upstream abandonment (audio producer dropped)
        if self.buffer.audio_consumer.is_abandoned() {
            // Process remaining buffered samples
            return self.buffer.audio_consumer.slots() > 0;
        }

        self.buffer.has_next_chunk()
    }

    async fn process_frame(&mut self) -> std::result::Result<(), Self::Error> {
        // Scenario 1: Check if output consumer is abandoned (downstream abandonment)
        if let Some(producer) = &self.token_producer {
            if producer.is_abandoned() {
                // Drop our token producer to signal downstream
                self.token_producer = None;
                self.mark_finished();
                return Ok(());
            }
        }

        // Process the next chunk
        self.process_next_chunk().await?;

        // Scenario 2: After processing, check if input is abandoned and no more frames (upstream abandonment)
        if self.buffer.audio_consumer.is_abandoned() && !self.has_next_frame() {
            // All input processed, drop output producer to signal downstream
            self.token_producer = None;
        }

        Ok(())
    }

    fn is_finished(&self) -> bool {
        self.buffer.is_finished && !self.has_next_frame()
    }

    fn mark_finished(&mut self) {
        self.buffer.finish();
    }

    async fn finalize(&mut self) -> std::result::Result<(), Self::Error> {
        // No additional finalization needed for Parakeet
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_config() {
        let context = ContextConfig::new(2.0, 0.5, 1.0, 16000);
        assert_eq!(context.left_samples, 32000);
        assert_eq!(context.chunk_samples, 8000);
        assert_eq!(context.right_samples, 16000);
        assert_eq!(context.total_samples(), 56000);
        assert!((context.latency_secs(16000) - 1.5).abs() < 0.001);
    }

    #[test]
    fn test_streaming_buffer() {
        let context = ContextConfig::new(0.1, 0.1, 0.1, 1000); // 100ms each
        // left_samples = 100, chunk_samples = 100, right_samples = 100

        // Create ring buffer for testing
        let (mut producer, consumer) = RingBuffer::new(1000);
        let mut buffer = StreamingAudioBuffer::new(context.clone(), consumer);

        // Add some samples (less than chunk + right context needed)
        let samples: Vec<f32> = (0..150).map(|i| i as f32).collect();
        for sample in &samples {
            let _ = producer.push(*sample);
        }

        // Should not have chunk yet (need chunk_samples + right_samples = 200 samples)
        assert!(!buffer.has_next_chunk());

        // Add more samples to reach the required size
        let more_samples: Vec<f32> = (150..250).map(|i| i as f32).collect();
        for sample in &more_samples {
            let _ = producer.push(*sample);
        }

        // Now should have a chunk (250 samples >= 200 needed)
        assert!(buffer.has_next_chunk());

        let (chunk, length) = buffer.get_next_chunk().unwrap();
        assert_eq!(chunk.shape()[0], 1); // batch size
        assert_eq!(length, 100); // chunk length

        // After processing first chunk:
        // - We consumed chunk_samples (100) + right_samples (100) = 200 from ring buffer
        // - Ring buffer now has 50 samples left
        // - Context buffer has 200 samples (left_samples + chunk_samples)
        // - Need 200 more samples in ring buffer for next chunk
        assert!(!buffer.has_next_chunk());

        // Add enough samples for second chunk
        let more_samples2: Vec<f32> = (250..450).map(|i| i as f32).collect();
        for sample in &more_samples2 {
            let _ = producer.push(*sample);
        }

        // Now should have second chunk (50 + 200 = 250 samples >= 200 needed)
        assert!(buffer.has_next_chunk());
        let (chunk2, length2) = buffer.get_next_chunk().unwrap();
        assert_eq!(chunk2.shape()[0], 1);
        assert_eq!(length2, 100);
    }

    #[test]
    fn test_upstream_abandonment_detection() {
        let context = ContextConfig::new(0.1, 0.1, 0.1, 1000); // 100ms each
        let (mut producer, consumer) = RingBuffer::new(1000);
        let buffer = StreamingAudioBuffer::new(context.clone(), consumer);

        // Add some samples
        let samples: Vec<f32> = (0..250).map(|i| i as f32).collect();
        for sample in &samples {
            let _ = producer.push(*sample);
        }

        // Should have a chunk available
        assert!(buffer.has_next_chunk());

        // Drop the producer to simulate upstream abandonment
        drop(producer);

        // Should still detect remaining samples
        assert!(buffer.audio_consumer.is_abandoned());
        assert!(buffer.audio_consumer.slots() > 0);
    }

    #[test]
    fn test_downstream_abandonment_detection() {
        let (audio_producer, audio_consumer) = RingBuffer::<f32>::new(1000);
        let (token_producer, token_consumer) = RingBuffer::<TokenResult>::new(100);

        // Create a minimal StreamingParakeetTDT for testing
        // We can't fully test without a model, but we can test the producer state
        assert!(!token_producer.is_abandoned());

        // Drop the consumer to simulate downstream abandonment
        drop(token_consumer);

        // Producer should detect abandonment
        assert!(token_producer.is_abandoned());

        // Clean up
        drop(audio_producer);
        drop(audio_consumer);
        drop(token_producer);
    }

    #[test]
    fn test_has_next_frame_with_abandonment() {
        let context = ContextConfig::new(0.1, 0.1, 0.1, 1000);
        let (mut producer, consumer) = RingBuffer::new(1000);
        let buffer = StreamingAudioBuffer::new(context.clone(), consumer);

        // Add insufficient samples for normal operation
        let samples: Vec<f32> = (0..150).map(|i| i as f32).collect();
        for sample in &samples {
            let _ = producer.push(*sample);
        }

        // Should not have chunk in normal operation (need 200 samples)
        assert!(!buffer.has_next_chunk());

        // Drop producer to simulate abandonment
        drop(producer);

        // Now should detect remaining samples even though insufficient for normal chunk
        assert!(buffer.audio_consumer.is_abandoned());
        assert!(buffer.audio_consumer.slots() > 0);
    }

    #[test]
    fn test_bidirectional_shutdown_signaling() {
        let (audio_producer, audio_consumer) = RingBuffer::<f32>::new(1000);
        let (token_producer, token_consumer) = RingBuffer::<TokenResult>::new(100);

        // Simulate upstream abandonment
        drop(audio_producer);
        assert!(audio_consumer.is_abandoned());

        // Simulate downstream abandonment
        drop(token_consumer);
        assert!(token_producer.is_abandoned());

        // Clean up
        drop(audio_consumer);
        drop(token_producer);
    }
}
