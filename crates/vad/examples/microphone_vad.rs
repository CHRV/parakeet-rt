use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, SampleRate as CpalSampleRate, StreamConfig};
use hound::{WavSpec, WavWriter};
use std::fs;
use std::io::BufWriter;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;
use tokio::time::{sleep, timeout};
use vad::{
    SpeechSegment, StreamingVad, VadEvent,
    silero::Silero,
    utils::{SampleRate, VadParams},
};

const MODEL_PATH: &str = "../../models/silero_vad.onnx";
const OUTPUT_DIR: &str = "speech_recordings";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("🎤 Real-time Microphone VAD Example");
    println!("This example captures audio from your microphone and detects speech in real-time");
    println!("Press Ctrl+C to stop recording\n");

    // Create output directory
    fs::create_dir_all(OUTPUT_DIR)?;
    println!("📁 Speech recordings will be saved to: {}", OUTPUT_DIR);

    // Initialize audio system
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or("No input device available")?;

    println!("🔊 Using audio device: {}", device.name()?);

    // Get device configuration
    let config = get_audio_config(&device)?;
    println!(
        "📊 Audio config: {} Hz, {} channels",
        config.sample_rate.0, config.channels
    );

    // Verify we're using 16kHz (required for Silero VAD)
    if config.sample_rate.0 != 16000 {
        println!(
            "⚠️  Warning: Device sample rate is {} Hz, but VAD expects 16kHz",
            config.sample_rate.0
        );
        println!("   Audio will be resampled, which may affect quality");
    }

    // Initialize VAD
    let silero = Silero::new(SampleRate::SixteenkHz, MODEL_PATH)?;
    let vad_params = VadParams {
        frame_size: 64,               // 64ms frames for responsiveness
        threshold: 0.4,               // Speech detection threshold
        min_silence_duration_ms: 100, // 200ms silence to end speech
        speech_pad_ms: 30,            // 100ms padding around speech
        min_speech_duration_ms: 300,  // Minimum 300ms for valid speech
        max_speech_duration_s: 2.0,   // Maximum 2 seconds per segment
        sample_rate: 16000,           // 16kHz sample rate
    };

    let (mut vad, mut vad_events) = StreamingVad::new(silero, vad_params);
    println!("✅ VAD initialized with Silero model");

    // Create channels for audio data
    let (audio_sender, mut audio_receiver) = mpsc::unbounded_channel::<Vec<i16>>();

    // Shared flag for graceful shutdown
    let running = Arc::new(AtomicBool::new(true));
    let running_clone = running.clone();

    // Setup Ctrl+C handler
    tokio::spawn(async move {
        tokio::signal::ctrl_c()
            .await
            .expect("Failed to listen for Ctrl+C");
        println!("\n🛑 Received Ctrl+C, shutting down gracefully...");
        running_clone.store(false, Ordering::Relaxed);
    });

    // Build audio input stream
    let stream = build_audio_stream(&device, &config, audio_sender, running.clone())?;

    // Start audio capture
    stream.play()?;
    println!("🎙️  Recording started! Speak into your microphone...\n");

    // Spawn VAD processing task
    let vad_running = running.clone();
    let vad_task = tokio::spawn(async move {
        while vad_running.load(Ordering::Relaxed) {
            tokio::select! {
                // Process incoming audio
                audio_chunk = audio_receiver.recv() => {
                    match audio_chunk {
                        Some(chunk) => {
                            if let Err(e) = vad.process_audio(&chunk) {
                                eprintln!("❌ VAD processing error: {}", e);
                                break;
                            }
                        }
                        None => {
                            println!("📡 Audio channel closed");
                            break;
                        }
                    }
                }

                // Check if we should stop
                _ = sleep(Duration::from_millis(100)) => {
                    if !vad_running.load(Ordering::Relaxed) {
                        break;
                    }
                }
            }
        }

        // Finalize any ongoing speech
        vad.finalize();
        println!("🔄 VAD processing finalized");
    });

    // Spawn event handling task
    let event_running = running.clone();
    let event_task = tokio::spawn(async move {
        let mut speech_counter = 0u32;

        while event_running.load(Ordering::Relaxed) {
            tokio::select! {
                // Handle VAD events (poll for events)
                _ = sleep(Duration::from_millis(10)) => {
                    while let Ok(event) = vad_events.pop() {
                        match event {
                            VadEvent::SpeechStarted { start_sample } => {
                                let timestamp = get_timestamp();
                                println!("🟢 [{}] Speech STARTED at sample {}", timestamp, start_sample);
                            }

                            VadEvent::SpeechOngoing { duration_ms, .. } => {
                                // Print ongoing speech every 500ms to avoid spam
                                if (duration_ms as u32) % 500 < 100 { // Approximate every 500ms
                                    print!("\r🗣️  Speaking... {:.1}s", duration_ms / 1000.0);
                                    std::io::Write::flush(&mut std::io::stdout()).unwrap();
                                }
                            }

                            VadEvent::SpeechEnded { segment } => {
                                speech_counter += 1;
                                let timestamp = get_timestamp();
                                println!("\n🔴 [{}] Speech ENDED: {:.2}s duration",
                                       timestamp, segment.duration_seconds());

                                // Save speech segment to file
                                match save_speech_segment(&segment, speech_counter).await {
                                    Ok(filename) => {
                                        println!("💾 Saved speech segment to: {}", filename);
                                    }
                                    Err(e) => {
                                        eprintln!("❌ Failed to save speech segment: {}", e);
                                    }
                                }

                                println!("   📊 Segment info:");
                                println!("      - Duration: {:.2}s", segment.duration_seconds());
                                println!("      - Samples: {} -> {}", segment.start_sample, segment.end_sample);
                                println!("      - Audio data: {} samples", segment.audio_data.len());
                                println!();
                            }
                        }
                    }
                }

                // Check if we should stop
                _ = sleep(Duration::from_millis(100)) => {
                    if !event_running.load(Ordering::Relaxed) {
                        break;
                    }
                }
            }
        }

        println!(
            "🎯 Event handling completed. Total speech segments: {}",
            speech_counter
        );
    });

    // Wait for shutdown signal
    while running.load(Ordering::Relaxed) {
        sleep(Duration::from_millis(100)).await;
    }

    // Stop audio stream
    drop(stream);

    // Wait for tasks to complete with timeout
    println!("⏳ Waiting for tasks to complete...");

    let _ = timeout(Duration::from_secs(5), vad_task).await;
    let _ = timeout(Duration::from_secs(5), event_task).await;

    println!("✅ Microphone VAD example completed!");
    println!(
        "📁 Check the '{}' directory for saved speech recordings",
        OUTPUT_DIR
    );

    Ok(())
}

fn get_audio_config(device: &Device) -> Result<StreamConfig, Box<dyn std::error::Error>> {
    let supported_configs = device.supported_input_configs()?;

    // Try to find a 16kHz mono configuration
    for supported_config in supported_configs {
        if supported_config.min_sample_rate() <= CpalSampleRate(16000)
            && supported_config.max_sample_rate() >= CpalSampleRate(16000)
        {
            let config = StreamConfig {
                channels: 1, // Mono
                sample_rate: CpalSampleRate(16000),
                buffer_size: cpal::BufferSize::Default,
            };

            return Ok(config);
        }
    }

    // Fallback to default config
    let default_config = device.default_input_config()?;
    Ok(StreamConfig {
        channels: 1, // Force mono
        sample_rate: default_config.sample_rate(),
        buffer_size: cpal::BufferSize::Default,
    })
}

fn build_audio_stream(
    device: &Device,
    config: &StreamConfig,
    audio_sender: mpsc::UnboundedSender<Vec<i16>>,
    running: Arc<AtomicBool>,
) -> Result<cpal::Stream, Box<dyn std::error::Error>> {
    let sample_rate = config.sample_rate.0;
    let channels = config.channels;

    let stream = device.build_input_stream(
        config,
        move |data: &[f32], _: &cpal::InputCallbackInfo| {
            if !running.load(Ordering::Relaxed) {
                return;
            }

            // Convert f32 samples to i16 and handle multi-channel audio
            let mut samples = Vec::with_capacity(data.len() / channels as usize);

            if channels == 1 {
                // Mono audio - direct conversion
                for &sample in data {
                    let sample_i16 = (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
                    samples.push(sample_i16);
                }
            } else {
                // Multi-channel audio - take only the first channel (left channel)
                for chunk in data.chunks(channels as usize) {
                    if let Some(&sample) = chunk.first() {
                        let sample_i16 = (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
                        samples.push(sample_i16);
                    }
                }
            }

            // Resample if necessary (simple decimation/interpolation)
            let resampled = if sample_rate != 16000 {
                resample_to_16khz(&samples, sample_rate)
            } else {
                samples
            };

            // Send to VAD processing
            if !resampled.is_empty() {
                let _ = audio_sender.send(resampled);
            }
        },
        |err| {
            eprintln!("❌ Audio stream error: {}", err);
        },
        None,
    )?;

    Ok(stream)
}

fn resample_to_16khz(samples: &[i16], original_rate: u32) -> Vec<i16> {
    if original_rate == 16000 {
        return samples.to_vec();
    }

    let ratio = original_rate as f64 / 16000.0;
    let output_len = (samples.len() as f64 / ratio) as usize;
    let mut output = Vec::with_capacity(output_len);

    for i in 0..output_len {
        let src_index = (i as f64 * ratio) as usize;
        if src_index < samples.len() {
            output.push(samples[src_index]);
        }
    }

    output
}

async fn save_speech_segment(
    segment: &SpeechSegment,
    counter: u32,
) -> Result<String, Box<dyn std::error::Error>> {
    let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();

    let filename = format!("{}/speech_{:03}_{}.wav", OUTPUT_DIR, counter, timestamp);

    // Create WAV file specification
    let spec = WavSpec {
        channels: 1,
        sample_rate: segment.sample_rate as u32,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    // Write audio data to WAV file
    let path = Path::new(&filename);
    let file = std::fs::File::create(path)?;
    let mut writer = WavWriter::new(BufWriter::new(file), spec)?;

    for &sample in &segment.audio_data {
        writer.write_sample(sample)?;
    }

    writer.finalize()?;

    Ok(filename)
}

fn get_timestamp() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();

    let secs = now.as_secs();
    let millis = now.subsec_millis();

    // Simple timestamp format: HH:MM:SS.mmm
    let hours = (secs / 3600) % 24;
    let minutes = (secs / 60) % 60;
    let seconds = secs % 60;

    format!("{:02}:{:02}:{:02}.{:03}", hours, minutes, seconds, millis)
}
