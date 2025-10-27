use std::time::Duration;
use tokio::time::sleep;
use vad::{
    StreamingVad, VadEvent,
    silero::Silero,
    utils::{SampleRate, VadParams},
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎤 Real-time VAD Example");
    println!("This example shows how to use the streaming VAD for real-time speech detection");

    // Initialize the Silero VAD model
    let silero = Silero::new(SampleRate::SixteenkHz, "../../models/silero_vad.onnx")?;

    // Configure parameters for real-time responsiveness
    let params = VadParams {
        frame_size: 64,               // 64ms frames for good responsiveness
        threshold: 0.5,               // Speech detection threshold
        min_silence_duration_ms: 300, // 300ms silence to end speech
        speech_pad_ms: 64,            // Padding around speech segments
        min_speech_duration_ms: 200,  // Minimum 200ms for valid speech
        max_speech_duration_s: 30.0,  // Maximum 30 seconds per segment
        sample_rate: 16000,           // 16kHz sample rate
    };

    // Create the streaming VAD
    let (mut streaming_vad, mut event_receiver) = StreamingVad::new(silero, params);

    // Spawn a task to handle VAD events
    let event_handler = tokio::spawn(async move {
        println!("📡 Listening for speech events...\n");

        while let Some(event) = event_receiver.recv().await {
            match event {
                VadEvent::SpeechStarted { start_sample } => {
                    println!("🟢 Speech STARTED at sample {}", start_sample);
                }
                VadEvent::SpeechOngoing { duration_ms, .. } => {
                    // Print ongoing speech updates (you might want to throttle this in real apps)
                    if (duration_ms as u32) % 500 == 0 {
                        // Every 500ms
                        print!("\r🗣️  Speaking... {:.1}s", duration_ms / 1000.0);
                        std::io::Write::flush(&mut std::io::stdout()).unwrap();
                    }
                }
                VadEvent::SpeechEnded { segment } => {
                    println!(
                        "\n🔴 Speech ENDED: {:.2}s duration",
                        segment.duration_seconds()
                    );
                    println!("   📊 Audio samples captured: {}", segment.audio_data.len());

                    // Here you would typically:
                    // - Save the audio segment to a file
                    // - Send it to a speech recognition service
                    // - Process it further

                    if segment.duration_seconds() > 1.0 {
                        println!("   ✨ Long enough for transcription!");
                    }
                    println!();
                }
            }
        }
    });

    // Simulate real-time audio processing
    println!("🎵 Simulating audio stream...");

    // In a real application, you would:
    // 1. Get audio from a microphone using cpal or similar
    // 2. Process it in real-time chunks
    // 3. Feed chunks to the streaming VAD

    simulate_realtime_audio(&mut streaming_vad).await?;

    // Finalize any ongoing speech
    streaming_vad.finalize().await;

    // Give the event handler time to process final events
    sleep(Duration::from_millis(100)).await;

    println!("✅ Real-time VAD example completed!");

    // The event handler will finish when the channel closes
    event_handler.abort();

    Ok(())
}

async fn simulate_realtime_audio(vad: &mut StreamingVad) -> Result<(), ort::Error> {
    let chunk_size = 1600; // 100ms chunks at 16kHz

    println!("   Phase 1: Silence (should not trigger speech detection)");
    for i in 0..10 {
        let silence = vec![0i16; chunk_size];
        vad.process_audio(&silence).await?;

        if i % 3 == 0 {
            print!(".");
            std::io::Write::flush(&mut std::io::stdout()).unwrap();
        }
        sleep(Duration::from_millis(100)).await;
    }
    println!(" ✓");

    println!("   Phase 2: Simulated speech (using sine wave)");
    for i in 0..15 {
        // Generate a sine wave as simulated speech
        let speech_chunk: Vec<i16> = (0..chunk_size)
            .map(|j| {
                let t = (i * chunk_size + j) as f32 / 16000.0;
                let amplitude = 8000.0;
                let frequency = 440.0; // A4 note
                (amplitude * (2.0 * std::f32::consts::PI * frequency * t).sin()) as i16
            })
            .collect();

        vad.process_audio(&speech_chunk).await?;

        if i % 3 == 0 {
            print!("♪");
            std::io::Write::flush(&mut std::io::stdout()).unwrap();
        }
        sleep(Duration::from_millis(100)).await;
    }
    println!(" ✓");

    println!("   Phase 3: Silence again (should end speech detection)");
    for i in 0..5 {
        let silence = vec![0i16; chunk_size];
        vad.process_audio(&silence).await?;

        if i % 2 == 0 {
            print!(".");
            std::io::Write::flush(&mut std::io::stdout()).unwrap();
        }
        sleep(Duration::from_millis(100)).await;
    }
    println!(" ✓");

    Ok(())
}
