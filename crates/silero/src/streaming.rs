use crate::{model, utils};
use async_trait::async_trait;
use frame_processor::FrameProcessor;
use rtrb::{Consumer, Producer, RingBuffer};

#[derive(Debug, Clone)]
pub struct SpeechSegment {
    pub start_sample: usize,
    pub end_sample: usize,
    pub audio_data: Vec<i16>,
    pub duration_ms: f32,
    pub sample_rate: usize,
}

impl SpeechSegment {
    pub fn duration_seconds(&self) -> f32 {
        self.duration_ms / 1000.0
    }
}

#[derive(Debug, Clone)]
pub enum VadEvent {
    SpeechStarted {
        start_sample: usize,
    },
    SpeechEnded {
        segment: SpeechSegment,
    },
    SpeechOngoing {
        current_sample: usize,
        duration_ms: f32,
    },
}

pub struct StreamingVad {
    silero: model::Silero,
    params: Params,
    state: State,
    audio_consumer: Consumer<f32>,
    speech_producer: Option<Producer<f32>>,
    rx_buffer_size: usize,
    tx_buffer_size: usize,
    is_finished_flag: bool,
}

// Exact same Params struct as vad.rs
#[allow(unused)]
#[derive(Debug)]
struct Params {
    frame_size: usize,
    threshold: f32,
    min_silence_duration_ms: usize,
    speech_pad_ms: usize,
    min_speech_duration_ms: usize,
    max_speech_duration_s: f32,
    sample_rate: usize,
    sr_per_ms: usize,
    frame_size_samples: usize,
    min_speech_samples: usize,
    speech_pad_samples: usize,
    max_speech_samples: f32,
    min_silence_samples: usize,
    min_silence_samples_at_max_speech: usize,
}

impl From<utils::VadParams> for Params {
    fn from(value: utils::VadParams) -> Self {
        let frame_size = value.frame_size;
        let threshold = value.threshold;
        let min_silence_duration_ms = value.min_silence_duration_ms;
        let speech_pad_ms = value.speech_pad_ms;
        let min_speech_duration_ms = value.min_speech_duration_ms;
        let max_speech_duration_s = value.max_speech_duration_s;
        let sample_rate = value.sample_rate;
        let sr_per_ms = sample_rate / 1000;
        let frame_size_samples = frame_size * sr_per_ms;
        let min_speech_samples = sr_per_ms * min_speech_duration_ms;
        let speech_pad_samples = sr_per_ms * speech_pad_ms;
        let max_speech_samples = sample_rate as f32 * max_speech_duration_s
            - frame_size_samples as f32
            - 2.0 * speech_pad_samples as f32;
        let min_silence_samples = sr_per_ms * min_silence_duration_ms;
        let min_silence_samples_at_max_speech = sr_per_ms * 98;
        Self {
            frame_size,
            threshold,
            min_silence_duration_ms,
            speech_pad_ms,
            min_speech_duration_ms,
            max_speech_duration_s,
            sample_rate,
            sr_per_ms,
            frame_size_samples,
            min_speech_samples,
            speech_pad_samples,
            max_speech_samples,
            min_silence_samples,
            min_silence_samples_at_max_speech,
        }
    }
}

// Exact same State struct as vad.rs but with event emission
#[derive(Debug, Default)]
struct State {
    current_sample: usize,
    temp_end: usize,
    next_start: usize,
    prev_end: usize,
    triggered: bool,
    current_speech: utils::TimeStamp,
    speeches: Vec<utils::TimeStamp>,
}

impl State {
    fn new() -> Self {
        Default::default()
    }

    fn update(&mut self, params: &Params, speech_prob: f32) {
        self.current_sample += params.frame_size_samples;

        if speech_prob > params.threshold {
            if self.temp_end != 0 {
                self.temp_end = 0;
                if self.next_start < self.prev_end {
                    self.next_start = self
                        .current_sample
                        .saturating_sub(params.frame_size_samples)
                }
            }
            if !self.triggered {
                self.triggered = true;
                self.current_speech.start =
                    self.current_sample as i64 - params.frame_size_samples as i64;
            }
            return;
        }

        if self.triggered
            && (self.current_sample as i64 - self.current_speech.start) as f32
                > params.max_speech_samples
        {
            if self.prev_end > 0 {
                self.current_speech.end = self.prev_end as _;
                self.take_speech();
                if self.next_start < self.prev_end {
                    self.triggered = false
                } else {
                    self.current_speech.start = self.next_start as _;
                }
                self.prev_end = 0;
                self.next_start = 0;
                self.temp_end = 0;
            } else {
                self.current_speech.end = self.current_sample as _;
                self.take_speech();
                self.prev_end = 0;
                self.next_start = 0;
                self.temp_end = 0;
                self.triggered = false;
            }
            return;
        }

        if self.triggered && speech_prob < (params.threshold - 0.15) {
            if self.temp_end == 0 {
                self.temp_end = self.current_sample;
            }

            if self.current_sample.saturating_sub(self.temp_end)
                > params.min_silence_samples_at_max_speech
            {
                self.prev_end = self.temp_end;
            }
            if self.current_sample.saturating_sub(self.temp_end) >= params.min_silence_samples {
                self.current_speech.end = self.temp_end as _;
                if self.current_speech.end - self.current_speech.start
                    > params.min_speech_samples as _
                {
                    self.take_speech();
                    self.prev_end = 0;
                    self.next_start = 0;
                    self.temp_end = 0;
                    self.triggered = false;
                }
            }
        }
    }

    fn take_speech(&mut self) {
        let speech = std::mem::take(&mut self.current_speech);
        self.speeches.push(speech);
    }

    fn check_for_last_speech(&mut self, last_sample: usize, _params: &Params) {
        if self.current_speech.start > 0 {
            self.current_speech.end = last_sample as _;
            self.take_speech();
            self.prev_end = 0;
            self.next_start = 0;
            self.temp_end = 0;
            self.triggered = false;
        }
    }
}

impl StreamingVad {
    pub fn new(
        silero: model::Silero,
        params: utils::VadParams,
    ) -> (Self, Producer<f32>, Consumer<f32>) {
        let params = Params::from(params);

        // Buffer sizes optimized for real-time processing
        let rx_buffer_size = params.sample_rate * 2; // 2 seconds of input audio buffer
        let tx_buffer_size = params.sample_rate * 10; // 10 seconds of speech output buffer

        // Create ring buffers for audio input and speech output
        let (audio_producer, audio_consumer) = RingBuffer::new(rx_buffer_size);
        let (speech_producer, speech_consumer) = RingBuffer::new(tx_buffer_size);

        let vad = Self {
            silero,
            params,
            state: State::new(),
            audio_consumer,
            speech_producer: Some(speech_producer),
            rx_buffer_size,
            tx_buffer_size,
            is_finished_flag: false,
        };

        (vad, audio_producer, speech_consumer)
    }

    pub fn process_audio(&mut self) -> Result<(), ort::Error> {
        // Process all available complete frames from the ring buffer
        // This method maintains backward compatibility by using a synchronous interface
        // It uses the same logic as the trait implementation but in a synchronous manner
        while self.has_next_frame() {
            self.process_frame_sync()?;
        }
        Ok(())
    }

    fn process_frame_sync(&mut self) -> Result<(), ort::Error> {
        // Read one frame from the ring buffer
        if let Ok(frame) = self
            .audio_consumer
            .read_chunk(self.params.frame_size_samples)
        {
            let (first, second) = frame.as_slices();
            let frame_data = [first, second].concat();
            frame.commit_all();

            // Process through Silero VAD
            let speech_prob = self.silero.calc_level(&frame_data)?;

            // Update state with proper VAD logic
            self.state.update(&self.params, speech_prob);

            // Only send frame to speech producer if we're in triggered state
            if self.state.triggered {
                if let Some(producer) = &mut self.speech_producer {
                    // Use the efficient chunk writing method if available
                    if let Ok(chunk) = producer.write_chunk_uninit(frame_data.len()) {
                        chunk.fill_from_iter(frame_data.iter().copied());
                    } else {
                        // Fallback to individual pushes if chunk writing fails
                        for &sample in &frame_data {
                            if producer.push(sample).is_err() {
                                break; // Buffer full, skip remaining samples
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    pub fn reset(&mut self) {
        self.silero.reset();
        self.state = State::new();
        // Clear ring buffers by consuming all available samples
        while self.audio_consumer.pop().is_ok() {}
    }

    pub fn finalize(&mut self) {
        // Check for any ongoing speech at the end
        let total_samples = self.state.current_sample;
        self.state
            .check_for_last_speech(total_samples, &self.params);
    }

    pub fn speeches(&self) -> &[utils::TimeStamp] {
        &self.state.speeches
    }

    pub fn rx_buffer_size(&self) -> usize {
        self.rx_buffer_size
    }

    pub fn tx_buffer_size(&self) -> usize {
        self.tx_buffer_size
    }

    pub fn is_triggered(&self) -> bool {
        self.state.triggered
    }

    /// Drop the speech producer to signal end of output to downstream
    pub fn close_output(&mut self) {
        self.speech_producer = None;
    }

    /// Check if the audio consumer has been abandoned (audio producer dropped)
    pub fn is_audio_abandoned(&self) -> bool {
        self.audio_consumer.is_abandoned()
    }

    /// Check if the speech producer has been dropped
    pub fn is_speech_producer_closed(&self) -> bool {
        self.speech_producer.is_none()
    }

    /// Check if the speech consumer has been abandoned (speech consumer dropped)
    pub fn is_speech_abandoned(&self) -> bool {
        self.speech_producer
            .as_ref()
            .map(|p| p.is_abandoned())
            .unwrap_or(true)
    }
}

#[async_trait]
impl FrameProcessor for StreamingVad {
    type Error = ort::Error;

    fn has_next_frame(&self) -> bool {
        // Check if audio producer was abandoned (upstream disconnection)
        if self.audio_consumer.is_abandoned() {
            // Process remaining buffered frames
            return self.audio_consumer.slots() >= self.params.frame_size_samples;
        }

        // Normal operation: check for sufficient samples
        self.audio_consumer.slots() >= self.params.frame_size_samples
    }

    async fn process_frame(&mut self) -> Result<(), Self::Error> {
        // Scenario 1: Check if speech consumer is abandoned (downstream disconnection)
        if let Some(producer) = &self.speech_producer {
            if producer.is_abandoned() {
                // Drop our speech producer to signal downstream
                self.speech_producer = None;
                self.mark_finished();
                return Ok(());
            }
        }

        // Delegate to the synchronous implementation
        // The Silero model operations are CPU-bound, not I/O-bound,
        // so there's no benefit to making them truly async
        self.process_frame_sync()?;

        // Scenario 2: After processing, check if input is abandoned and no more frames (upstream exhaustion)
        if self.audio_consumer.is_abandoned() && !self.has_next_frame() {
            // All input processed, drop output producer to signal downstream
            self.speech_producer = None;
        }

        Ok(())
    }

    fn is_finished(&self) -> bool {
        self.is_finished_flag && !self.has_next_frame()
    }

    fn mark_finished(&mut self) {
        self.is_finished_flag = true;
    }

    async fn finalize(&mut self) -> Result<(), Self::Error> {
        // Check for any ongoing speech at the end
        let total_samples = self.state.current_sample;
        self.state
            .check_for_last_speech(total_samples, &self.params);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::SampleRate;

    #[tokio::test]
    async fn test_streaming_vad_basic() {
        let silero = model::Silero::new(SampleRate::SixteenkHz, "../../models/silero_vad.onnx")
            .expect("Failed to create Silero model");

        let params = utils::VadParams::default();
        let (mut vad, mut audio_producer, mut speech_consumer) = StreamingVad::new(silero, params);

        // Test with silence
        let silence = vec![0f32; 1600]; // 100ms of silence

        // Push silence to the audio producer
        for sample in silence {
            let _ = audio_producer.push(sample);
        }

        // Process the audio
        vad.process_audio().expect("Failed to process silence");

        // Should not be triggered for silence
        assert!(
            !vad.is_triggered(),
            "VAD should not be triggered for silence"
        );

        // Should not receive any speech samples for silence
        assert!(
            speech_consumer.pop().is_err(),
            "Should not receive speech samples for silence"
        );
    }
}
