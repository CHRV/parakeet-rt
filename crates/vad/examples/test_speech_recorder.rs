use hound::{WavSpec, WavWriter};
use std::fs;
use std::time::Duration;
use tokio::time::sleep;
use vad::{
    StreamingVad,
    silero::Silero,
    utils::{SampleRate, VadParams},
};

const MODEL_PATH: &str = "models/silero_vad.onnx";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 Test Speech Recorder (Simulated Audio)");
    println!("This example tests the speech recording functionality with simulated audio.\n");

    // Create output directory
    fs::create_dir_all("test_recordings")?;

    // Initialize VAD with optimized parameters
    let silero = Silero::new(SampleRate::SixteenkHz, MODEL_PATH)?;
    let params = VadParams {
        frame_size: 32,               // 32ms frames
        threshold: 0.3,               // Lower threshold for simulated audio
        min_silence_duration_ms: 200, // 200ms silence to end speech
        speech_pad_ms: 50,            // 50ms padding
        min_speech_duration_ms: 100,  // Minimum 100ms for valid speech
        max_speech_duration_s: 10.0,  // Maximum 10s per segment
        sample_rate: 16000,
    };

    let (mut vad, mut audio_producer, mut speech_consumer) = StreamingVad::new(silero, params);
    println!("✅ VAD initialized for testing");

    // Test parameters
    let chunk_size = 512; // 32ms chunks at 16kHz
    let mut continuous_speech_buffer = Vec::new();
    let mut recording_session = 1;

    println!("🎵 Generating test audio sequence...\n");

    // Phase 1: Silence (should not be recorded)
    println!("Phase 1: Silence (2 seconds)");
    for _ in 0..125 {
        // 2 seconds of silence
        let silence = vec![0f32; chunk_size];

        // Push to VAD
        for sample in silence {
            let _ = audio_producer.push(sample);
        }
        vad.process_audio()?;

        // Collect any speech (should be none)
        while let Ok(sample) = speech_consumer.pop() {
            continuous_speech_buffer.push(sample);
        }

        sleep(Duration::from_millis(16)).await; // Simulate real-time
    }
    println!(
        "✓ Silence phase complete - collected {} samples",
        continuous_speech_buffer.len()
    );

    // Phase 2: Simulated speech (sine wave)
    println!("Phase 2: Simulated speech (3 seconds)");
    for i in 0..188 {
        // 3 seconds of speech
        let speech_chunk: Vec<f32> = (0..chunk_size)
            .map(|j| {
                let t = (i * chunk_size + j) as f32 / 16000.0;
                let amplitude = 12000.0; // Strong signal
                let frequency = 440.0 + (t * 50.0).sin() * 100.0; // Varying frequency
                (amplitude * (2.0 * std::f32::consts::PI * frequency * t).sin()) / (i16::MAX as f32)
            })
            .collect();

        // Push to VAD
        for sample in speech_chunk {
            let _ = audio_producer.push(sample);
        }
        vad.process_audio()?;

        // Collect speech samples
        while let Ok(sample) = speech_consumer.pop() {
            continuous_speech_buffer.push(sample);
        }

        // Show progress
        if i % 30 == 0 {
            let duration = continuous_speech_buffer.len() as f32 / 16000.0;
            print!("\r🗣️  Recording: {:.1}s", duration);
            std::io::Write::flush(&mut std::io::stdout()).unwrap();
        }

        sleep(Duration::from_millis(16)).await; // Simulate real-time
    }
    println!(
        "\n✓ Speech phase complete - collected {} samples",
        continuous_speech_buffer.len()
    );
    // Phase 3: More silence (should trigger save)
    println!("Phase 3: Silence to trigger save (1 second)");
    for _ in 0..63 {
        // 1 second of silence
        let silence = vec![0f32; chunk_size];

        // Push to VAD
        for sample in silence {
            let _ = audio_producer.push(sample);
        }
        vad.process_audio()?;

        // Collect any remaining speech
        while let Ok(sample) = speech_consumer.pop() {
            continuous_speech_buffer.push(sample);
        }

        sleep(Duration::from_millis(16)).await;
    }

    // Save the recorded speech
    if !continuous_speech_buffer.is_empty() {
        let filename = format!("test_recordings/test_speech_{:03}.wav", recording_session);
        let duration = continuous_speech_buffer.len() as f32 / 16000.0;

        println!("💾 Saving test recording...");
        match save_audio(&continuous_speech_buffer, &filename, 16000) {
            Ok(_) => {
                println!(
                    "✅ Saved: {} ({:.1}s, {} samples)",
                    filename,
                    duration,
                    continuous_speech_buffer.len()
                );
            }
            Err(e) => {
                eprintln!("❌ Failed to save {}: {}", filename, e);
            }
        }

        continuous_speech_buffer.clear();
        recording_session += 1;
    }

    // Phase 4: Another speech segment
    println!("Phase 4: Second speech segment (2 seconds)");
    for i in 0..125 {
        // 2 seconds of different speech
        let speech_chunk: Vec<_> = (0..chunk_size)
            .map(|j| {
                let t = (i * chunk_size + j) as f32 / 16000.0;
                let amplitude = 10000.0;
                let frequency = 220.0 + (t * 2.0).cos() * 80.0; // Different pattern
                (amplitude * (2.0 * std::f32::consts::PI * frequency * t).sin()) / (i16::MAX as f32)
            })
            .collect();

        // Push to VAD
        for sample in speech_chunk {
            let _ = audio_producer.push(sample);
        }
        vad.process_audio()?;

        // Collect speech samples
        while let Ok(sample) = speech_consumer.pop() {
            continuous_speech_buffer.push(sample);
        }

        if i % 20 == 0 {
            let duration = continuous_speech_buffer.len() as f32 / 16000.0;
            print!("\r🗣️  Recording segment 2: {:.1}s", duration);
            std::io::Write::flush(&mut std::io::stdout()).unwrap();
        }

        sleep(Duration::from_millis(16)).await;
    }
    println!("\n✓ Second speech segment complete");

    // Finalize and save remaining speech
    vad.finalize();

    if !continuous_speech_buffer.is_empty() {
        let filename = format!("test_recordings/test_speech_{:03}.wav", recording_session);
        let duration = continuous_speech_buffer.len() as f32 / 16000.0;

        println!("💾 Saving final test recording...");
        match save_audio(&continuous_speech_buffer, &filename, 16000) {
            Ok(_) => {
                println!(
                    "✅ Saved final: {} ({:.1}s, {} samples)",
                    filename,
                    duration,
                    continuous_speech_buffer.len()
                );
            }
            Err(e) => {
                eprintln!("❌ Failed to save final recording: {}", e);
            }
        }
    }

    println!("\n🎉 Test completed successfully!");
    println!("📁 Check the 'test_recordings' directory for generated WAV files");
    println!("🔍 The recordings should contain only speech audio without silence");

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
        sample_format: hound::SampleFormat::Float,
    };

    let mut writer = WavWriter::create(filename, spec)?;
    for &sample in samples {
        writer.write_sample(sample)?;
    }
    writer.finalize()?;

    Ok(())
}
