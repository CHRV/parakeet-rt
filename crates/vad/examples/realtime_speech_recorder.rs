use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleRate as CpalSampleRate, StreamConfig};
use hound::{WavSpec, WavWriter};
use std::fs;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tokio::time::sleep;
use vad::{
    StreamingVad,
    silero::Silero,
    utils::{SampleRate, VadParams},
};

const MODEL_PATH: &str = "../../models/silero_vad.onnx";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎤 Real-time Speech Recorder (No Silence)");
    println!(
        "This example captures speech in real-time and stores only speech audio without silence."
    );
    println!("Press Ctrl+C to stop recording.\n");

    // Create output directory
    fs::create_dir_all("audio/recordings")?;

    // Setup audio capture
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or("No input device available")?;

    println!("🎙️  Using input device: {}", device.name()?);

    let config = StreamConfig {
        channels: 1,
        sample_rate: CpalSampleRate(16000),
        buffer_size: cpal::BufferSize::Default,
    };

    // Initialize VAD
    let silero = Silero::new(SampleRate::SixteenkHz, MODEL_PATH)?;
    let params = VadParams {
        frame_size: 32,               // 32ms frames for responsiveness
        threshold: 0.4,               // Speech detection threshold
        min_silence_duration_ms: 300, // 300ms of silence to end speech
        speech_pad_ms: 200,           // 100ms padding around speech
        min_speech_duration_ms: 250,  // Minimum 250ms for valid speech
        max_speech_duration_s: 30.0,  // Maximum 30s per speech segment
        sample_rate: 16000,
    };

    let (mut vad, mut audio_producer, mut speech_consumer) = StreamingVad::new(silero, params);
    println!("✅ VAD initialized with optimized parameters");
    // Audio processing channel
    let (audio_tx, mut audio_rx) = mpsc::unbounded_channel::<Vec<f32>>();

    // Shutdown flag
    let running = Arc::new(AtomicBool::new(true));
    let running_clone = running.clone();

    // Ctrl+C handler
    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.unwrap();
        println!("\n🛑 Received Ctrl+C, stopping recording...");
        running_clone.store(false, Ordering::Relaxed);
    });

    // Audio stream - captures microphone input
    let stream = device.build_input_stream(
        &config,
        move |data: &[f32], _| {
            let samples: Vec<f32> = data.iter().copied().collect();
            let _ = audio_tx.send(samples);
        },
        |err| eprintln!("❌ Audio error: {}", err),
        None,
    )?;

    stream.play()?;
    println!("🎙️  Recording started! Speak now...\n");

    // Speech recording state
    let mut continuous_speech_buffer = Vec::new();
    let mut last_speech_time = Instant::now();
    let mut recording_session = 1;
    let mut is_currently_speaking = false;
    let silence_timeout = Duration::from_secs(2); // Save after 2 seconds of no speech

    // Main processing loop
    while running.load(Ordering::Relaxed) {
        tokio::select! {
            // Process incoming audio from microphone
            Some(audio_chunk) = audio_rx.recv() => {
                // Debug: Show audio input activity
                let max_amplitude = audio_chunk.iter().map(|&x| (x.abs() * (i16::MAX as f32)) as i16).max().unwrap_or(0);
                if max_amplitude > 1000 {
                    print!("\r🎵 Audio input: max amplitude {}", max_amplitude);
                    std::io::Write::flush(&mut std::io::stdout()).unwrap();
                }

                // Push audio samples to VAD input ring buffer
                for sample in audio_chunk {
                    let _ = audio_producer.push(sample);
                }

                // Process available frames through VAD
                if let Err(e) = vad.process_audio() {
                    eprintln!("❌ VAD processing error: {}", e);
                    break;
                }

                // Check if VAD is currently detecting speech
                let currently_triggered = vad.is_triggered();

                // Handle speech state changes
                if currently_triggered && !is_currently_speaking {
                    println!("\n🟢 Speech detected - starting recording");
                    is_currently_speaking = true;
                } else if !currently_triggered && is_currently_speaking {
                    println!("\n🔴 Speech ended");
                    is_currently_speaking = false;
                }

                // Collect speech samples from VAD output
                let mut new_speech_samples = Vec::new();
                while let Ok(sample) = speech_consumer.pop() {
                    new_speech_samples.push(sample);
                }

                // Add speech samples to continuous buffer
                if !new_speech_samples.is_empty() {
                    continuous_speech_buffer.extend(new_speech_samples);
                    last_speech_time = Instant::now();

                    // Show progress
                    let duration = continuous_speech_buffer.len() as f32 / 16000.0;
                    print!("\r🗣️  Recording speech: {:.1}s", duration);
                    std::io::Write::flush(&mut std::io::stdout()).unwrap();
                }

                // Debug: Show buffer status periodically
                use std::sync::atomic::{AtomicU32, Ordering};
                static DEBUG_COUNTER: AtomicU32 = AtomicU32::new(0);
                let count = DEBUG_COUNTER.fetch_add(1, Ordering::Relaxed);
                if count % 100 == 0 {
                    println!("\n🔍 Debug: triggered: {}, speech buffer: {} samples",
                           currently_triggered, continuous_speech_buffer.len());
                }
            }

            // Check for silence timeout to save recording
            _ = sleep(Duration::from_millis(100)) => {
                if !continuous_speech_buffer.is_empty() &&
                   last_speech_time.elapsed() > silence_timeout {

                    println!("\n💾 Saving speech recording...");

                    // Save the continuous speech buffer
                    let filename = format!("audio/recordings/speech_session_{:03}.wav", recording_session);
                    let duration = continuous_speech_buffer.len() as f32 / 16000.0;

                    match save_audio(&continuous_speech_buffer, &filename, 16000) {
                        Ok(_) => {
                            println!("✅ Saved: {} ({:.1}s, {} samples)",
                                   filename, duration, continuous_speech_buffer.len());
                        }
                        Err(e) => {
                            eprintln!("❌ Failed to save {}: {}", filename, e);
                        }
                    }

                    // Reset for next recording session
                    continuous_speech_buffer.clear();
                    recording_session += 1;
                    println!("🎙️  Ready for next speech...\n");
                }
            }
        }
    }

    // Finalize VAD and save any remaining speech
    vad.finalize();

    if !continuous_speech_buffer.is_empty() {
        println!("💾 Saving final speech recording...");
        let filename = format!(
            "audio/recordings/speech_session_{:03}.wav",
            recording_session
        );
        let duration = continuous_speech_buffer.len() as f32 / 16000.0;

        match save_audio(&continuous_speech_buffer, &filename, 16000) {
            Ok(_) => {
                println!("✅ Saved final recording: {} ({:.1}s)", filename, duration);
            }
            Err(e) => {
                eprintln!("❌ Failed to save final recording: {}", e);
            }
        }
    }

    println!(
        "\n✅ Recording session completed. Saved {} speech segments.",
        if continuous_speech_buffer.is_empty() {
            recording_session - 1
        } else {
            recording_session
        }
    );

    Ok(())
}

fn save_audio(
    samples: &[f32],
    filename: &str,
    sample_rate: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = WavWriter::create(filename, spec)?;

    for &sample in samples {
        writer.write_sample((sample * i16::MAX as f32) as i16)?;
    }
    writer.finalize()?;

    Ok(())
}
