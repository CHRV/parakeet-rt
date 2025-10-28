use crate::{silero, utils};
use rtrb::{Consumer, Producer, RingBuffer};
use tracing::debug;

const DEBUG_SPEECH_PROB: bool = true;

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
    silero: silero::Silero,
    params: Params,
    state: State,
    audio_consumer: Consumer<i16>,
    audio_producer: Producer<i16>,
    speech_consumer: Consumer<i16>,
    speech_producer: Producer<i16>,
    event_producer: Producer<VadEvent>,
    rx_buffer_size: usize,
    tx_buffer_size: usize,
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

    fn update(
        &mut self,
        params: &Params,
        speech_prob: f32,
        event_producer: &mut Producer<VadEvent>,
        speech_consumer: &Consumer<i16>,
    ) {
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
                self.debug(speech_prob, params, "start");
                self.triggered = true;
                self.current_speech.start =
                    self.current_sample as i64 - params.frame_size_samples as i64;

                // Emit speech started event
                let _ = event_producer.push(VadEvent::SpeechStarted {
                    start_sample: self.current_speech.start as usize,
                });
            } else {
                // Emit ongoing speech event
                let duration_ms = ((self.current_sample as i64 - self.current_speech.start) as f32
                    / params.sample_rate as f32)
                    * 1000.0;

                let _ = event_producer.push(VadEvent::SpeechOngoing {
                    current_sample: self.current_sample,
                    duration_ms,
                });
            }
            return;
        }

        if self.triggered
            && (self.current_sample as i64 - self.current_speech.start) as f32
                > params.max_speech_samples
        {
            if self.prev_end > 0 {
                self.current_speech.end = self.prev_end as _;
                self.take_speech(params, event_producer, speech_consumer);
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
                self.take_speech(params, event_producer, speech_consumer);
                self.prev_end = 0;
                self.next_start = 0;
                self.temp_end = 0;
                self.triggered = false;
            }
            return;
        }

        if speech_prob >= (params.threshold - 0.15) && (speech_prob < params.threshold) {
            if self.triggered {
                self.debug(speech_prob, params, "speaking")
            } else {
                self.debug(speech_prob, params, "silence")
            }
        }

        if self.triggered && speech_prob < (params.threshold - 0.15) {
            if self.temp_end == 0 {
                self.temp_end = self.current_sample;
                self.debug(speech_prob, params, "end"); // Only debug on first "end" detection
            } else {
                self.debug(speech_prob, params, "silence"); // Debug as silence for subsequent frames
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
                    self.take_speech(params, event_producer, speech_buffer);
                    self.prev_end = 0;
                    self.next_start = 0;
                    self.temp_end = 0;
                    self.triggered = false;
                }
            }
        }
    }

    fn take_speech(
        &mut self,
        params: &Params,
        event_producer: &mut Producer<VadEvent>,
        speech_consumer: &Consumer<i16>,
    ) {
        let speech = std::mem::take(&mut self.current_speech);

        // Extract audio data for this speech segment
        let audio_data = extract_speech_segment(
            speech_consumer,
            speech.start as usize,
            speech.end as usize,
            self.current_sample,
        );

        // Emit speech ended event
        let duration_ms = ((speech.end - speech.start) as f32 / params.sample_rate as f32) * 1000.0;
        let segment = SpeechSegment {
            start_sample: speech.start as usize,
            end_sample: speech.end as usize,
            audio_data,
            duration_ms,
            sample_rate: params.sample_rate,
        };

        let _ = event_producer.push(VadEvent::SpeechEnded { segment });

        self.speeches.push(speech);
    }

    fn check_for_last_speech(
        &mut self,
        last_sample: usize,
        params: &Params,
        event_producer: &mut Producer<VadEvent>,
        speech_consumer: &Consumer<i16>,
    ) {
        if self.current_speech.start > 0 {
            self.current_speech.end = last_sample as _;
            self.take_speech(params, event_producer, speech_consumer);
            self.prev_end = 0;
            self.next_start = 0;
            self.temp_end = 0;
            self.triggered = false;
        }
    }

    fn debug(&self, speech_prob: f32, params: &Params, title: &str) {
        if DEBUG_SPEECH_PROB {
            let speech = self.current_sample as f32
                - params.frame_size_samples as f32
                - if title == "end" {
                    params.speech_pad_samples
                } else {
                    0
                } as f32; // minus window_size_samples to get precise start time point.
            debug!(
                "[{:10}: {:.3} s ({:.3}) {:8}]",
                title,
                speech / params.sample_rate as f32,
                speech_prob,
                self.current_sample - params.frame_size_samples,
            );
        }
    }
}

fn extract_speech_segment(
    speech_buffer: &VecDeque<i16>,
    start_sample: usize,
    end_sample: usize,
    current_sample: usize,
) -> Vec<i16> {
    let buffer_len = speech_buffer.len();

    if buffer_len == 0 || start_sample >= current_sample {
        return Vec::new();
    }

    let buffer_start = current_sample.saturating_sub(buffer_len);

    if start_sample < buffer_start {
        // Speech started before our buffer
        let relative_end = end_sample.saturating_sub(buffer_start).min(buffer_len);
        speech_buffer.range(0..relative_end).copied().collect()
    } else {
        let relative_start = start_sample - buffer_start;
        let relative_end = (end_sample - buffer_start).min(buffer_len);

        if relative_start < buffer_len {
            speech_buffer
                .range(relative_start..relative_end)
                .copied()
                .collect()
        } else {
            Vec::new()
        }
    }
}

impl StreamingVad {
    pub fn new(silero: silero::Silero, params: utils::VadParams) -> (Self, Consumer<VadEvent>) {
        let (event_producer, event_consumer) = RingBuffer::new(1024);
        let params = Params::from(params);
        let max_buffer_size = params.sample_rate * 30; // Keep 30 seconds of audio max

        let vad = Self {
            silero,
            params,
            state: State::new(),
            audio_buffer: VecDeque::new(),
            speech_audio_buffer: VecDeque::new(),
            event_producer,
            max_buffer_size,
        };

        (vad, event_consumer)
    }

    pub fn process_audio(&mut self, audio_chunk: &[i16]) -> Result<(), ort::Error> {
        // Add to audio buffers
        self.audio_buffer.extend(audio_chunk);
        self.speech_audio_buffer.extend(audio_chunk);

        // Keep speech buffer size manageable
        while self.speech_audio_buffer.len() > self.max_buffer_size {
            self.speech_audio_buffer.pop_front();
        }

        // Process complete frames
        while self.audio_buffer.len() >= self.params.frame_size_samples {
            let frame: Vec<i16> = self
                .audio_buffer
                .drain(..self.params.frame_size_samples)
                .collect();
            self.process_frame(&frame)?;
        }

        Ok(())
    }

    fn process_frame(&mut self, frame: &[i16]) -> Result<(), ort::Error> {
        let speech_prob = self.silero.calc_level(frame)?;
        self.state.update(
            &self.params,
            speech_prob,
            &mut self.event_producer,
            &self.speech_audio_buffer,
        );
        Ok(())
    }

    pub fn reset(&mut self) {
        self.silero.reset();
        self.state = State::new();
        self.audio_buffer.clear();
        self.speech_audio_buffer.clear();
    }

    pub fn finalize(&mut self) {
        // Check for any ongoing speech at the end
        let total_samples = self.state.current_sample;
        self.state.check_for_last_speech(
            total_samples,
            &self.params,
            &mut self.event_producer,
            &self.speech_audio_buffer,
        );
    }

    pub fn speeches(&self) -> &[utils::TimeStamp] {
        &self.state.speeches
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::SampleRate;

    #[test]
    fn test_streaming_vad_basic() {
        let silero = silero::Silero::new(SampleRate::SixteenkHz, "../../models/silero_vad.onnx")
            .expect("Failed to create Silero model");

        let params = utils::VadParams::default();
        let (mut vad, mut event_consumer) = StreamingVad::new(silero, params);

        // Test with silence
        let silence = vec![0i16; 1600]; // 100ms of silence
        vad.process_audio(&silence)
            .expect("Failed to process silence");

        // Should not receive any events for silence
        assert!(
            event_consumer.pop().is_err(),
            "Should not receive events for silence"
        );
    }
}
