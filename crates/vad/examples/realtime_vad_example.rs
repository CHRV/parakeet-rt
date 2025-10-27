use std::time::Duration;
use tokio::sync::{mpsc, watch};
use tokio::time::sleep;
use vad::{
    StreamingVad, VadEvent,
    silero::Silero,
    utils::{SampleRate, VadParams},
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing for logging
    tracing_subscriber::fmt::init();

    // Create the Silero VAD model
    let silero = Silero::new(SampleRate::SixteenkHz, "../../models/silero_vad.onnx")?;

    // Configure VAD parameters for real-time use
    let params = VadParams {
        frame_size: 64,               // 64ms frames for responsiveness
        threshold: 0.5,               // Speech detection threshold
        min_silence_duration_ms: 300, // 300ms of silence to end speech
        speech_pad_ms: 64,            // Padding around speech
        min_speech_duration_ms: 250,  // Minimum 250ms for valid speech
        max_speech_duration_s: 30.0,  // Max 30 seconds per segment
        sample_rate: 16000,           // 16kHz sample rate
    };

    // Create the streaming VAD
    let (mut streaming_vad, mut event_receiver) = StreamingVad::new(silero, params);

    // Create channels for audio input and shutdown signaling
    let (audio_sender, mut audio_receiver) = mpsc::unbounded_channel::<Vec<i16>>();
    let (shutdown_sender, mut shutdown_receiver) = watch::channel(false);

    // Spawn the VAD processing task
    let vad_handle = tokio::spawn(async move {
        loop {
            tokio::select! {
                // Process incoming audio
                audio_chunk = audio_receiver.recv() => {
                    match audio_chunk {
                        Some(chunk) => {
                            if let Err(e) = streaming_vad.process_audio(&chunk).await {
                                eprintln!("VAD processing error: {}", e);
                                break;
                            }
                        }
                        None => {
                            println!("Audio channel closed");
                            break;
                        }
                    }
                }

                // Check for shutdown
                _ = shutdown_receiver.changed() => {
                    if *shutdown_receiver.borrow() {
                        println!("Shutting down VAD processor");
                        streaming_vad.finalize().await;
                        break;
                    }
                }
            }
        }
    });

    // Spawn the event handling task
    let event_handle = tokio::spawn(async move {
        while let Some(event) = event_receiver.recv().await {
            match event {
                VadEvent::SpeechStarted { start_sample } => {
                    println!("🎤 Speech started at sample {}", start_sample);
                }
                VadEvent::SpeechOngoing {
                    current_sample,
                    duration_ms,
                } => {
                    print!("\r🗣️  Speech ongoing: {:.1}s", duration_ms / 1000.0);
                    std::io::Write::flush(&mut std::io::stdout()).unwrap();
                }
                VadEvent::SpeechEnded { segment } => {
                    println!(
                        "\n✅ Speech ended: {:.2}s duration, {} audio samples",
                        segment.duration_seconds(),
                        segment.audio_data.len()
                    );

                    // Here you could save the audio segment, send it for transcription, etc.
                    if !segment.audio_data.is_empty() {
                        println!("   Audio data available for processing");
                    }
                }
            }
        }
    });

    // Simulate real-time audio input
    println!("Starting real-time VAD simulation...");
    println!("This example simulates audio chunks being processed in real-time");

    // Simulate different types of audio
    simulate_audio_stream(&audio_sender).await;

    // Let the system process for a bit
    sleep(Duration::from_secs(2)).await;

    // Signal shutdown
    println!("\nShutting down...");
    shutdown_sender.send(true)?;

    // Wait for tasks to complete
    let _ = tokio::join!(vad_handle, event_handle);

    println!("Real-time VAD example completed");
    Ok(())
}

async fn simulate_audio_stream(audio_sender: &mpsc::UnboundedSender<Vec<i16>>) {
    let chunk_size = 1600; // 100ms at 16kHz

    println!("Simulating silence...");
    // Send silence
    for _ in 0..10 {
        let silence_chunk = vec![0i16; chunk_size];
        let _ = audio_sender.send(silence_chunk);
        sleep(Duration::from_millis(100)).await;
    }

    println!("Simulating speech...");
    // Send simulated speech (random noise above threshold)
    for _i in 0..20 {
        let speech_chunk: Vec<i16> = (0..chunk_size)
            .map(|_| (rand::random::<f32>() * 2000.0 - 1000.0) as i16)
            .collect();
        let _ = audio_sender.send(speech_chunk);
        sleep(Duration::from_millis(100)).await;
    }

    println!("Simulating silence again...");
    // Send silence again
    for _ in 0..5 {
        let silence_chunk = vec![0i16; chunk_size];
        let _ = audio_sender.send(silence_chunk);
        sleep(Duration::from_millis(100)).await;
    }

    println!("Simulating short speech burst...");
    // Send short speech burst
    for _ in 0..3 {
        let speech_chunk: Vec<i16> = (0..chunk_size)
            .map(|_| (rand::random::<f32>() * 1500.0 - 750.0) as i16)
            .collect();
        let _ = audio_sender.send(speech_chunk);
        sleep(Duration::from_millis(100)).await;
    }
}
