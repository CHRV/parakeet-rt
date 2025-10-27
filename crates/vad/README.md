# VAD (Voice Activity Detection) Crate

A Rust implementation of real-time Voice Activity Detection using the Silero VAD model with Tokio async support.

## Features

- **Real-time streaming VAD** with immediate speech detection
- **Tokio async/await support** for non-blocking processing
- **Event-driven architecture** using channels for speech notifications
- **Configurable parameters** for different use cases
- **Audio segment extraction** with captured speech data
- **Exact Silero VAD algorithm** implementation for accurate speech detection

## Quick Start

### Basic Usage

```rust
use vad::{
    silero::Silero,
    utils::{SampleRate, VadParams},
    StreamingVad, VadEvent,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize the Silero VAD model
    let silero = Silero::new(SampleRate::SixteenkHz, "path/to/silero_vad.onnx")?;

    // Configure VAD parameters
    let params = VadParams {
        frame_size: 64,           // 64ms frames
        threshold: 0.5,           // Speech detection threshold
        min_silence_duration_ms: 300, // 300ms silence to end speech
        speech_pad_ms: 64,        // Padding around speech
        min_speech_duration_ms: 200, // Minimum speech duration
        max_speech_duration_s: 30.0, // Maximum speech duration
        sample_rate: 16000,       // 16kHz sample rate
    };

    // Create streaming VAD
    let (mut vad, mut events) = StreamingVad::new(silero, params);

    // Process audio chunks
    let audio_chunk: Vec<i16> = get_audio_from_microphone();
    vad.process_audio(&audio_chunk).await?;

    // Handle events
    while let Some(event) = events.recv().await {
        match event {
            VadEvent::SpeechStarted { start_sample } => {
                println!("Speech started at sample {}", start_sample);
            }
            VadEvent::SpeechEnded { segment } => {
                println!("Speech ended: {:.2}s duration", segment.duration_seconds());
                // Process the captured audio segment
                process_speech_segment(&segment.audio_data);
            }
            VadEvent::SpeechOngoing { duration_ms, .. } => {
                println!("Speech ongoing: {:.1}s", duration_ms / 1000.0);
            }
        }
    }

    Ok(())
}
```

### Real-time Microphone Processing

```rust
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use tokio::sync::mpsc;

async fn setup_realtime_vad() -> Result<(), Box<dyn std::error::Error>> {
    let (audio_sender, mut audio_receiver) = mpsc::unbounded_channel();

    // Setup audio input (using cpal)
    let host = cpal::default_host();
    let device = host.default_input_device().unwrap();
    let config = device.default_input_config()?;

    let stream = device.build_input_stream(
        &config.into(),
        move |data: &[i16], _: &cpal::InputCallbackInfo| {
            let _ = audio_sender.send(data.to_vec());
        },
        |err| eprintln!("Audio error: {}", err),
        None,
    )?;

    // Setup VAD
    let silero = Silero::new(SampleRate::SixteenkHz, "silero_vad.onnx")?;
    let (mut vad, mut events) = StreamingVad::new(silero, VadParams::default());

    // Start audio stream
    stream.play()?;

    // Process audio in real-time
    tokio::spawn(async move {
        while let Some(audio_chunk) = audio_receiver.recv().await {
            if let Err(e) = vad.process_audio(&audio_chunk).await {
                eprintln!("VAD error: {}", e);
            }
        }
    });

    // Handle speech events
    while let Some(event) = events.recv().await {
        handle_speech_event(event).await;
    }

    Ok(())
}
```

## API Reference

### StreamingVad

The main real-time VAD processor.

#### Methods

- `new(silero: Silero, params: VadParams) -> (StreamingVad, Receiver<VadEvent>)`
  - Creates a new streaming VAD instance with event channel
- `process_audio(&mut self, audio: &[i16]) -> Result<(), ort::Error>`
  - Process an audio chunk and emit events for speech detection
- `finalize(&mut self) -> ()`
  - Finalize processing and emit any pending speech segments
- `reset(&mut self)`
  - Reset the VAD state

### VadEvent

Events emitted by the streaming VAD:

- `SpeechStarted { start_sample: usize }`
  - Emitted when speech begins
- `SpeechOngoing { current_sample: usize, duration_ms: f32 }`
  - Emitted periodically during ongoing speech
- `SpeechEnded { segment: SpeechSegment }`
  - Emitted when speech ends with the complete segment

### SpeechSegment

Contains information about a detected speech segment:

```rust
pub struct SpeechSegment {
    pub start_sample: usize,    // Start position in samples
    pub end_sample: usize,      // End position in samples
    pub audio_data: Vec<i16>,   // Captured audio data
    pub duration_ms: f32,       // Duration in milliseconds
    pub sample_rate: usize,     // Sample rate of the audio
}
```

### VadParams

Configuration parameters for the VAD:

```rust
pub struct VadParams {
    pub frame_size: usize,              // Frame size in milliseconds (default: 64)
    pub threshold: f32,                 // Speech detection threshold (default: 0.5)
    pub min_silence_duration_ms: usize, // Minimum silence to end speech (default: 0)
    pub speech_pad_ms: usize,           // Padding around speech (default: 64)
    pub min_speech_duration_ms: usize,  // Minimum speech duration (default: 64)
    pub max_speech_duration_s: f32,     // Maximum speech duration (default: ∞)
    pub sample_rate: usize,             // Audio sample rate (default: 16000)
}
```

## Configuration Guidelines

### For Real-time Applications

```rust
VadParams {
    frame_size: 32,           // Smaller frames for lower latency
    threshold: 0.4,           // Lower threshold for sensitivity
    min_silence_duration_ms: 200, // Quick speech end detection
    min_speech_duration_ms: 150,  // Filter out very short sounds
    ..Default::default()
}
```

### For Batch Processing

```rust
VadParams {
    frame_size: 64,           // Standard frame size
    threshold: 0.5,           // Balanced threshold
    min_silence_duration_ms: 500, // Longer silence for stability
    min_speech_duration_ms: 250,  // Filter short utterances
    ..Default::default()
}
```

### For Noisy Environments

```rust
VadParams {
    threshold: 0.6,           // Higher threshold to reduce false positives
    min_speech_duration_ms: 300, // Longer minimum to filter noise
    ..Default::default()
}
```

## Performance Considerations

- **Frame Size**: Smaller frames (32ms) provide lower latency but higher CPU usage
- **Threshold**: Lower values are more sensitive but may cause false positives
- **Buffer Management**: The streaming VAD maintains an internal audio buffer for segment extraction
- **Event Handling**: Process events asynchronously to avoid blocking audio processing

## Examples

Run the included examples:

```bash
# Simple real-time usage example (simulated audio)
cargo run --example simple_realtime_usage

# Full real-time VAD example with simulation
cargo run --example realtime_vad_example

# Real microphone VAD (captures from your microphone)
cargo run --example microphone_vad

# Simple microphone VAD (basic version)
cargo run --example simple_microphone_vad
```

### Microphone Examples

The microphone examples capture real audio from your default input device and perform real-time speech detection:

- **`microphone_vad`**: Full-featured example with detailed logging, automatic resampling, and comprehensive speech segment saving
- **`simple_microphone_vad`**: Minimal example showing the basic microphone integration

Both examples will:
1. 🎤 Capture audio from your microphone
2. 🔍 Detect speech in real-time using the Silero VAD
3. 📢 Print "Speech started" and "Speech ended" messages
4. 💾 Save each speech segment as a separate WAV file

**Requirements:**
- A working microphone
- The Silero VAD model file at `../../models/silero_vad.onnx`

**Output:**
- Speech segments are saved to `speech_recordings/` (full example) or `recordings/` (simple example)
- Each file is named with a counter and timestamp: `speech_001_1234567890.wav`

## Testing

Run the test suite:

```bash
# Run all tests
cargo test

# Run integration tests
cargo test --test integration_test
cargo test --test streaming_integration_test
```

## Requirements

- Silero VAD ONNX model file
- Audio sample rate: 8kHz or 16kHz
- Audio format: 16-bit signed integers (i16)

## License

This crate is part of the Parakeet RT project.
#
# Quick Start with Microphone

To quickly get started with real-time microphone VAD:

1. **Run the simple microphone example:**
   ```bash
   cargo run --example simple_microphone_vad
   ```

2. **Speak into your microphone** - you'll see:
   ```
   🟢 Speech started!
   🔴 Speech ended! Duration: 2.34s
   💾 Saved: recordings/speech_001.wav
   ```

3. **Check the recordings folder** for your saved speech segments

4. **Press Ctrl+C** to stop recording

The example will automatically:
- 🎤 Capture audio from your default microphone
- 🔍 Detect when you start and stop speaking
- 💾 Save each speech segment as a separate WAV file
- 📊 Show the duration of each speech segment

Perfect for building voice assistants, transcription systems, or any application that needs to know when someone is speaking!