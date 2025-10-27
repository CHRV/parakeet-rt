use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleRate as CpalSampleRate, StreamConfig};
use hound::{WavSpec, WavWriter};
use std::fs;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::time::sleep;
use vad::{
    StreamingVad, VadEvent,
    silero::Silero,
    utils::{SampleRate, VadParams},
};

const MODEL_PATH: &str = "../../models/silero_vad.onnx";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎤 Simple Microphone VAD");
    println!("Speak into your microphone. Press Ctrl+C to stop.\n");

    // Create output directory
    fs::create_dir_all("recordings")?;

    // Setup audio capture
    let host = cpal::default_host();
    let device = host.default_input_device().unwrap();

    let config = StreamConfig {
        channels: 1,
        sample_rate: CpalSampleRate(16000),
        buffer_size: cpal::BufferSize::Default,
    };

    // Initialize VAD
    let silero = Silero::new(SampleRate::SixteenkHz, MODEL_PATH)?;
    let params = VadParams::default();
    let (mut vad, mut events) = StreamingVad::new(silero, params);

    // Audio processing channel
    let (audio_tx, mut audio_rx) = mpsc::unbounded_channel();

    // Shutdown flag
    let running = Arc::new(AtomicBool::new(true));
    let running_clone = running.clone();

    // Ctrl+C handler
    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.unwrap();
        running_clone.store(false, Ordering::Relaxed);
    });

    // Audio stream
    let stream = device.build_input_stream(
        &config,
        move |data: &[f32], _| {
            let samples: Vec<i16> = data
                .iter()
                .map(|&x| (x.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)
                .collect();
            let _ = audio_tx.send(samples);
        },
        |err| eprintln!("Audio error: {}", err),
        None,
    )?;

    stream.play()?;
    println!("🎙️  Recording... Speak now!");

    // Process audio and handle events
    let mut speech_count = 0;

    while running.load(Ordering::Relaxed) {
        tokio::select! {
            // Process audio
            Some(audio) = audio_rx.recv() => {
                vad.process_audio(&audio).await?;
            }

            // Handle VAD events
            Some(event) = events.recv() => {
                match event {
                    VadEvent::SpeechStarted { .. } => {
                        println!("🟢 Speech started!");
                    }
                    VadEvent::SpeechEnded { segment } => {
                        speech_count += 1;
                        println!("🔴 Speech ended! Duration: {:.2}s", segment.duration_seconds());

                        // Save to file
                        let filename = format!("recordings/speech_{:03}.wav", speech_count);
                        save_audio(&segment.audio_data, &filename, 16000)?;
                        println!("💾 Saved: {}", filename);
                    }
                    _ => {}
                }
            }

            // Check for shutdown
            _ = sleep(Duration::from_millis(100)) => {}
        }
    }

    println!("\n✅ Stopped. Saved {} speech segments.", speech_count);
    Ok(())
}

fn save_audio(
    samples: &[i16],
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
        writer.write_sample(sample)?;
    }
    writer.finalize()?;

    Ok(())
}
