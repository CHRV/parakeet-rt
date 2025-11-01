use crate::error::Result;
use crate::parakeet_tdt::{ParakeetTDTModel, State};
use crate::vocab::Vocabulary;
use ndarray::{Array1, Array2, Array3};
use rtrb::{Consumer, Producer, RingBuffer};

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
    pub fn new(
        left_secs: f32,
        chunk_secs: f32,
        right_secs: f32,
        sample_rate: usize,
    ) -> Self {
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
    /// Internal buffer storing all audio samples for context access
    buffer: Vec<f32>,
    /// Context configuration
    context: ContextConfig,
    /// Current position in the stream
    position: usize,
    /// Whether we've reached the end of the stream
    is_finished: bool,
}

impl StreamingAudioBuffer {
    /// Create new streaming buffer
    pub fn new(context: ContextConfig) -> Self {
        Self {
            buffer: Vec::new(),
            context,
            position: 0,
            is_finished: false,
        }
    }

    /// Add audio samples to the buffer
    pub fn add_samples(&mut self, samples: &[f32]) {
        self.buffer.extend_from_slice(samples);
    }

    /// Mark the stream as finished (no more samples will be added)
    pub fn finish(&mut self) {
        self.is_finished = true;
    }

    /// Check if there's enough data for the next chunk
    pub fn has_next_chunk(&self) -> bool {
        if self.is_finished {
            // If finished, we can process remaining samples
            self.buffer.len() > self.position
        } else {
            // Need full context for processing
            self.buffer.len() >= self.position + self.context.total_samples()
        }
    }

    /// Get the next chunk with context for processing
    pub fn get_next_chunk(&mut self) -> Option<(Array2<f32>, usize)> {
        if !self.has_next_chunk() {
            return None;
        }

        let start_pos = self.position.saturating_sub(self.context.left_samples);

        let end_pos = if self.is_finished {
            // For the last chunk, take all remaining samples
            self.buffer.len()
        } else {
            std::cmp::min(
                self.position + self.context.chunk_samples + self.context.right_samples,
                self.buffer.len(),
            )
        };

        if start_pos >= end_pos {
            return None;
        }

        // Extract samples for this chunk
        let chunk_samples = self.buffer[start_pos..end_pos].to_vec();

        // Calculate actual chunk length (excluding context)
        let actual_chunk_start = if self.position >= self.context.left_samples {
            self.context.left_samples
        } else {
            self.position
        };

        let actual_chunk_end = if self.is_finished {
            chunk_samples.len()
        } else {
            actual_chunk_start + self.context.chunk_samples
        };

        let chunk_length = actual_chunk_end - actual_chunk_start;

        // Advance position
        self.position += chunk_length;

        // Create batch array [1, samples]
        let audio_array = Array2::from_shape_vec((1, chunk_samples.len()), chunk_samples)
            .map_err(|_| ())
            .ok()?;

        Some((audio_array, chunk_length))
    }

    /// Reset buffer for new stream
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.position = 0;
        self.is_finished = false;
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
    /// Ring buffer consumer for reading input audio
    audio_consumer: Consumer<f32>,
    /// Ring buffer producer for outputting transcription tokens
    token_producer: Producer<TokenResult>,
    /// Audio buffer for context management
    buffer: StreamingAudioBuffer,
    /// Current decoder state
    state: Option<State>,
    /// Vocabulary for token decoding
    vocab: Option<Vocabulary>,
    /// Sample rate
    _sample_rate: usize,
    /// Input buffer size
    rx_buffer_size: usize,
    /// Output buffer size
    tx_buffer_size: usize,
}

impl StreamingParakeetTDT {
    /// Create new streaming inference engine
    /// Returns (engine, audio_producer, token_consumer)
    pub fn new(
        model: ParakeetTDTModel,
        context: ContextConfig,
        sample_rate: usize
    ) -> (Self, Producer<f32>, Consumer<TokenResult>) {
        Self::new_with_vocab(model, context, sample_rate, None)
    }

    /// Create new streaming inference engine with vocabulary
    /// Returns (engine, audio_producer, token_consumer)
    pub fn new_with_vocab(
        model: ParakeetTDTModel,
        context: ContextConfig,
        sample_rate: usize,
        vocab: Option<Vocabulary>
    ) -> (Self, Producer<f32>, Consumer<TokenResult>) {
        // Buffer sizes optimized for real-time processing
        let rx_buffer_size = sample_rate * 2; // 2 seconds of input audio buffer
        let tx_buffer_size = 1000; // Buffer for 1000 tokens

        // Create ring buffers for audio input and token output
        let (audio_producer, audio_consumer) = RingBuffer::new(rx_buffer_size);
        let (token_producer, token_consumer) = RingBuffer::new(tx_buffer_size);

        let engine = Self {
            model,
            context: context.clone(),
            audio_consumer,
            token_producer,
            buffer: StreamingAudioBuffer::new(context),
            state: None,
            vocab,
            _sample_rate: sample_rate,
            rx_buffer_size,
            tx_buffer_size,
        };

        (engine, audio_producer, token_consumer)
    }

    /// Process audio from the input ring buffer
    /// This should be called regularly to process incoming audio
    pub fn process_audio(&mut self) -> Result<()> {
        // Read all available audio from ring buffer
        self.update_from_ring_buffer();

        // Process available chunks and emit tokens
        while self.buffer.has_next_chunk() {
            let tokens = self.process_next_chunk()?;

            // Send tokens to output ring buffer
            for token in tokens {
                let text = self.vocab.as_ref()
                    .and_then(|v| v.decode_token(token.0))
                    .map(|s| s.to_string());

                let token_result = TokenResult {
                    token_id: token.0,
                    timestamp: token.1,
                    confidence: 1.0, // TODO: Add confidence calculation
                    text,
                };

                if self.token_producer.push(token_result).is_err() {
                    // Output buffer full, could log warning
                    break;
                }
            }
        }

        Ok(())
    }

    /// Update buffer from ring buffer
    fn update_from_ring_buffer(&mut self) {
        let mut samples = Vec::new();
        while let Ok(sample) = self.audio_consumer.pop() {
            samples.push(sample);
        }
        if !samples.is_empty() {
            self.buffer.add_samples(&samples);
        }
    }

    /// Mark the audio stream as finished
    pub fn finalize(&mut self) {
        self.update_from_ring_buffer();
        self.buffer.finish();

        // Process any remaining audio
        let _ = self.process_audio();
    }

    /// Process a single chunk and return new tokens with timestamps
    fn process_next_chunk(&mut self) -> Result<Vec<(i32, usize)>> {
        let (audio_chunk, chunk_length) = match self.buffer.get_next_chunk() {
            Some(chunk) => chunk,
            None => return Ok(Vec::new()),
        };

        // Store values we need later before moving them
        let audio_chunk_shape = audio_chunk.shape()[1];
        let audio_lengths = Array1::from_vec(vec![audio_chunk.shape()[1] as i64]);
        let audio_lengths_val = audio_lengths[0];

        // Run preprocessing
        let (features, features_len) = self.model.preprocess(audio_chunk, audio_lengths)?;

        // Run encoder
        let (encoder_out, encoder_len) = self.model.encode(features, features_len)?;

        // Process only the chunk portion (excluding context)
        let mut new_tokens = Vec::new();
        let mut chunk_tokens = Vec::new(); // Local token history for this chunk

        let start_frame = if self.buffer.position >= self.context.left_samples {
            // Calculate frame offset for left context
            (self.context.left_samples * encoder_len[0] as usize) / audio_lengths_val as usize
        } else {
            0
        };

        let chunk_frames = (chunk_length * encoder_len[0] as usize) / audio_chunk_shape;
        let end_frame = std::cmp::min(start_frame + chunk_frames, encoder_len[0] as usize);

        // Process frames for this chunk only
        for frame_idx in start_frame..end_frame {
            if frame_idx >= encoder_out.shape()[1] {
                break;
            }

            // Get encoding at this frame - encoder_out shape is (batch, time, features)
            let encoding = encoder_out.slice(ndarray::s![0, frame_idx, ..]).to_owned();

            // Decode this frame
            let (probs, step, new_state) = self.decode_frame(encoding, &chunk_tokens)?;

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
                chunk_tokens.push(token);
                new_tokens.push((token, timestamp));
            }

            // Handle time advancement (simplified for streaming)
            if step == 0 && token == self.model.config.blank_idx as i32 {
                // Continue to next frame
                continue;
            }
        }

        Ok(new_tokens)
    }

    /// Decode a single frame
    fn decode_frame(&mut self, encoding: Array1<f32>, tokens: &[i32]) -> Result<(Array1<f32>, usize, State)> {
        let prev_state = self.state.clone().unwrap_or_else(|| State {
            h: Array3::<f32>::zeros((2, 1, 640)),
            c: Array3::<f32>::zeros((2, 1, 640)),
        });

        self.model.decode(tokens, prev_state, encoding)
    }

    /// Reset the streaming state
    pub fn reset(&mut self) {
        self.buffer.reset();
        // Clear input ring buffer
        while self.audio_consumer.pop().is_ok() {}
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
        let mut buffer = StreamingAudioBuffer::new(context);

        // Add some samples
        let samples: Vec<f32> = (0..500).map(|i| i as f32).collect();
        buffer.add_samples(&samples);

        // Should not have chunk yet (need 300 samples total for full context)
        assert!(!buffer.has_next_chunk());

        // Add more samples to reach the required context size
        let more_samples: Vec<f32> = (500..800).map(|i| i as f32).collect();
        buffer.add_samples(&more_samples);

        // Now should have a chunk
        assert!(buffer.has_next_chunk());

        let (chunk, length) = buffer.get_next_chunk().unwrap();
        assert_eq!(chunk.shape()[0], 1); // batch size
        assert_eq!(length, 100); // chunk length
    }
}