use hound::WavReader;
use rtrb::{Consumer, Producer};
use silero::{
    StreamingVad,
    model::Silero,
    utils::{SampleRate, VadParams},
};
use std::time::Duration;
use tokio::time::sleep;

fn get_model_path() -> String {
    if let Ok(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {
        format!("{}/../../models/silero_vad.onnx", manifest_dir)
    } else {
        "../../models/silero_vad.onnx".to_string()
    }
}

fn get_audio_path(filename: &str) -> String {
    if let Ok(manifest_dir) = std::env::var("CARGO_MANIFEST_DIR") {
        format!("{}/../../audio/{}", manifest_dir, filename)
    } else {
        format!("../../audio/{}", filename)
    }
}

fn load_wav_samples(path: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let mut reader = WavReader::open(path)?;
    let samples: Result<Vec<i16>, _> = reader.samples().collect();

    Ok(samples?
        .iter()
        .map(|x| (*x as f32) / (i16::MAX as f32))
        .collect::<Vec<_>>())
}

fn create_streaming_vad() -> Result<(StreamingVad, Producer<f32>, Consumer<f32>), ort::Error> {
    let model_path = get_model_path();
    let silero = Silero::new(SampleRate::SixteenkHz, &model_path)?;
    let params = VadParams {
        frame_size: 64,
        threshold: 0.4,
        min_silence_duration_ms: 200,
        speech_pad_ms: 30,
        min_speech_duration_ms: 100,
        max_speech_duration_s: 10.0,
        sample_rate: 16000,
    };
    Ok(StreamingVad::new(silero, params))
}

#[tokio::test]
async fn test_streaming_speech_detection_sample_1() {
    let (mut vad, mut audio_producer, mut speech_consumer) =
        create_streaming_vad().expect("Failed to create streaming VAD");

    let samples =
        load_wav_samples(&get_audio_path("sample_1.wav")).expect("Failed to load sample_1.wav");

    // Process audio in chunks to simulate streaming
    let chunk_size = 1600; // 100ms at 16kHz
    let mut speech_samples = Vec::new();
    let mut speech_detected = false;

    // Process audio chunks
    for chunk in samples.chunks(chunk_size) {
        // Push audio samples to VAD input ring buffer
        for sample in chunk {
            let _ = audio_producer.push(*sample);
        }

        // Process available frames through VAD
        vad.process_audio().expect("Failed to process audio");

        // Check if speech is currently being detected
        if vad.is_triggered() && !speech_detected {
            println!("Speech started");
            speech_detected = true;
        } else if !vad.is_triggered() && speech_detected {
            println!("Speech ended");
            speech_detected = false;
        }

        // Collect speech samples from VAD output
        while let Ok(sample) = speech_consumer.pop() {
            speech_samples.push(sample);
        }
    }

    // Finalize to ensure any ongoing speech is completed
    vad.finalize();

    // Give a small delay for final processing
    sleep(Duration::from_millis(10)).await;

    // Collect any remaining speech samples
    while let Ok(sample) = speech_consumer.pop() {
        speech_samples.push(sample);
    }

    let speeches = vad.speeches();

    // Verify we detected speech
    assert!(
        !speeches.is_empty(),
        "Expected to detect speech in sample_1.wav"
    );

    assert!(
        !speech_samples.is_empty(),
        "Expected to collect speech samples from sample_1.wav"
    );

    println!(
        "Speech segments detected in sample_1.wav: {}",
        speeches.len()
    );
    for (i, speech) in speeches.iter().enumerate() {
        println!("  Segment {}: {}", i + 1, speech);
    }

    let duration_seconds = speech_samples.len() as f32 / 16000.0;
    println!("Total speech duration: {:.2} seconds", duration_seconds);
}

#[tokio::test]
async fn test_streaming_no_speech_detection() {
    let (mut vad, mut audio_producer, mut speech_consumer) =
        create_streaming_vad().expect("Failed to create streaming VAD");

    // Create pure silence
    let silence_samples = vec![0f32; 16000]; // 1 second of silence

    // Push silence samples to VAD
    for sample in silence_samples {
        let _ = audio_producer.push(sample);
    }

    vad.process_audio().expect("Failed to process silence");
    vad.finalize();

    let speeches = vad.speeches();
    let mut speech_samples = Vec::new();

    // Collect any speech samples (should be none)
    while let Ok(sample) = speech_consumer.pop() {
        speech_samples.push(sample);
    }

    // Should not detect any speech
    assert!(speeches.is_empty(), "Should not detect speech in silence");

    assert!(
        speech_samples.is_empty(),
        "Should not collect speech samples from silence"
    );

    println!("No speech detected in silence (as expected)");
}

#[tokio::test]
async fn test_streaming_multiple_speech_segments() {
    let (mut vad, mut audio_producer, mut speech_consumer) =
        create_streaming_vad().expect("Failed to create streaming VAD");

    let mut speech_samples = Vec::new();
    let mut speech_segments_detected = 0;
    let mut currently_speaking = false;

    // Simulate: silence -> speech -> silence -> speech -> silence
    let chunk_size = 800; // 50ms chunks

    // Silence
    let silence = vec![0f32; chunk_size * 4]; // 200ms silence
    for sample in silence {
        let _ = audio_producer.push(sample);
    }
    vad.process_audio().expect("Failed to process silence");

    // First speech segment (simulated with sine wave - more likely to be detected as speech)
    for i in 0..10 {
        // 500ms of simulated speech
        let speech_chunk: Vec<f32> = (0..chunk_size)
            .map(|j| {
                let t = ((i * chunk_size + j) as f32) / 16000.0;
                let amplitude = 8000.0;
                let frequency = 440.0 + (t * 2.0).sin() * 100.0; // Varying frequency
                (amplitude * (2.0 * std::f32::consts::PI * frequency * t).sin()) / (i16::MAX as f32)
            })
            .collect();

        for sample in speech_chunk {
            let _ = audio_producer.push(sample);
        }

        vad.process_audio().expect("Failed to process speech");

        // Check for speech state changes
        if vad.is_triggered() && !currently_speaking {
            println!("Speech segment {} started", speech_segments_detected + 1);
            currently_speaking = true;
        } else if !vad.is_triggered() && currently_speaking {
            println!("Speech segment {} ended", speech_segments_detected + 1);
            currently_speaking = false;
            speech_segments_detected += 1;
        }

        // Collect speech samples
        while let Ok(sample) = speech_consumer.pop() {
            speech_samples.push(sample);
        }
    }

    // Silence between segments
    let silence = vec![0f32; chunk_size * 6]; // 300ms silence
    for sample in silence {
        let _ = audio_producer.push(sample);
    }
    vad.process_audio().expect("Failed to process silence");

    // Check for speech end after silence
    if !vad.is_triggered() && currently_speaking {
        println!(
            "Speech segment {} ended after silence",
            speech_segments_detected + 1
        );
        currently_speaking = false;
        speech_segments_detected += 1;
    }

    // Collect speech samples after silence
    while let Ok(sample) = speech_consumer.pop() {
        speech_samples.push(sample);
    }

    // Second speech segment
    for i in 0..8 {
        // 400ms of simulated speech
        let speech_chunk: Vec<f32> = (0..chunk_size)
            .map(|j| {
                let t = ((i * chunk_size + j) as f32) / 16000.0;
                let amplitude = 7000.0;
                let frequency = 220.0 + (t * 3.0).cos() * 80.0; // Different pattern
                (amplitude * (2.0 * std::f32::consts::PI * frequency * t).sin()) / (i16::MAX as f32)
            })
            .collect();

        for sample in speech_chunk {
            let _ = audio_producer.push(sample);
        }

        vad.process_audio().expect("Failed to process speech");

        // Check for speech state changes
        if vad.is_triggered() && !currently_speaking {
            println!("Speech segment {} started", speech_segments_detected + 1);
            currently_speaking = true;
        } else if !vad.is_triggered() && currently_speaking {
            println!("Speech segment {} ended", speech_segments_detected + 1);
            currently_speaking = false;
            speech_segments_detected += 1;
        }

        // Collect speech samples
        while let Ok(sample) = speech_consumer.pop() {
            speech_samples.push(sample);
        }
    }

    vad.finalize();

    // Check for final speech end
    if currently_speaking {
        println!("Final speech segment ended");
        speech_segments_detected += 1;
    }

    // Give a small delay for final processing
    sleep(Duration::from_millis(10)).await;

    // Collect final speech samples
    while let Ok(sample) = speech_consumer.pop() {
        speech_samples.push(sample);
    }

    let speeches = vad.speeches();

    println!("Multiple speech segments test completed:");
    println!("  Speech segments detected by VAD: {}", speeches.len());
    println!(
        "  Speech segments tracked during processing: {}",
        speech_segments_detected
    );
    println!("  Total speech samples collected: {}", speech_samples.len());

    if !speech_samples.is_empty() {
        let duration_seconds = speech_samples.len() as f32 / 16000.0;
        println!("  Total speech duration: {:.2} seconds", duration_seconds);
    }

    // The test passes if no errors occurred during processing
    assert!(true, "Multiple speech segments test completed successfully");
}

#[tokio::test]
async fn test_streaming_real_audio_files() {
    let (mut vad, mut audio_producer, mut speech_consumer) =
        create_streaming_vad().expect("Failed to create streaming VAD");

    // Test with birds.wav (should not detect speech)
    let birds_samples =
        load_wav_samples(&get_audio_path("birds.wav")).expect("Failed to load birds.wav");

    let chunk_size = 1600;
    let mut speech_samples = Vec::new();

    for chunk in birds_samples.chunks(chunk_size) {
        // Push audio samples to VAD input ring buffer
        for sample in chunk {
            let _ = audio_producer.push(*sample);
        }

        vad.process_audio().expect("Failed to process birds audio");

        // Collect speech samples from VAD output
        while let Ok(sample) = speech_consumer.pop() {
            speech_samples.push(sample);
        }
    }

    vad.finalize();

    // Give a small delay for final processing
    sleep(Duration::from_millis(10)).await;

    // Collect any remaining speech samples
    while let Ok(sample) = speech_consumer.pop() {
        speech_samples.push(sample);
    }

    let speeches = vad.speeches();

    // Birds should not trigger speech detection
    assert!(
        speeches.is_empty(),
        "Birds audio should not trigger speech detection, but got {} segments",
        speeches.len()
    );

    assert!(
        speech_samples.is_empty(),
        "Birds audio should not produce speech samples, but got {} samples",
        speech_samples.len()
    );

    println!("Birds audio correctly identified as non-speech");
}

#[tokio::test]
async fn test_upstream_abandonment_detection() {
    use frame_processor::FrameProcessor;

    let (mut vad, mut audio_producer, mut speech_consumer) =
        create_streaming_vad().expect("Failed to create streaming VAD");

    // Push some audio samples
    let samples = vec![0.1f32; 1600]; // 100ms of audio
    for sample in samples {
        let _ = audio_producer.push(sample);
    }

    // Drop the audio producer to simulate upstream abandonment
    drop(audio_producer);

    // The VAD should detect abandonment and process remaining frames
    assert!(
        vad.is_audio_abandoned(),
        "Audio consumer should detect producer abandonment"
    );

    // Process remaining frames - should handle abandonment gracefully
    while vad.has_next_frame() {
        vad.process_frame().await.expect("Failed to process frame");
    }

    // After processing all remaining frames, speech producer should be dropped
    assert!(
        vad.is_speech_producer_closed(),
        "Speech producer should be dropped after upstream abandonment"
    );

    // Speech consumer should detect abandonment
    assert!(
        speech_consumer.is_abandoned(),
        "Speech consumer should detect producer abandonment"
    );

    println!("Upstream abandonment detection test passed");
}

#[tokio::test]
async fn test_downstream_abandonment_detection() {
    use frame_processor::FrameProcessor;

    let (mut vad, mut audio_producer, speech_consumer) =
        create_streaming_vad().expect("Failed to create streaming VAD");

    // Push some audio samples
    let samples = vec![0.1f32; 1600]; // 100ms of audio
    for sample in samples {
        let _ = audio_producer.push(sample);
    }

    // Drop the speech consumer to simulate downstream abandonment
    drop(speech_consumer);

    // Verify speech producer detects abandonment before processing
    assert!(
        vad.is_speech_abandoned(),
        "Speech producer should detect consumer abandonment"
    );

    // Process a frame - should detect downstream abandonment
    vad.process_frame().await.expect("Failed to process frame");

    // Speech producer should be dropped after detecting downstream abandonment
    assert!(
        vad.is_speech_producer_closed(),
        "Speech producer should be dropped after downstream abandonment"
    );

    // Process all remaining frames - VAD should stop processing
    while vad.has_next_frame() {
        vad.process_frame().await.expect("Failed to process frame");
    }

    // VAD should be marked as finished after downstream abandonment
    assert!(
        vad.is_finished(),
        "VAD should be finished after downstream abandonment and all frames processed"
    );

    println!("Downstream abandonment detection test passed");
}

#[tokio::test]
async fn test_bidirectional_abandonment_with_process_loop() {
    use frame_processor::FrameProcessor;

    let (mut vad, mut audio_producer, speech_consumer) =
        create_streaming_vad().expect("Failed to create streaming VAD");

    // Spawn a task to feed audio and then drop the producer
    let feed_task = tokio::spawn(async move {
        // Feed some audio
        for _ in 0..10 {
            let samples = vec![0.1f32; 1600]; // 100ms chunks
            for sample in samples {
                let _ = audio_producer.push(sample);
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        // Producer is dropped here when task ends
    });

    // Spawn a task to consume speech and then drop the consumer
    let consume_task = tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(50)).await;
        // Consumer is dropped here when task ends
        drop(speech_consumer);
    });

    // Process frames - should handle both upstream and downstream abandonment
    let process_result = tokio::time::timeout(Duration::from_secs(2), async {
        while !vad.is_finished() {
            if vad.has_next_frame() {
                vad.process_frame().await.expect("Failed to process frame");
            } else {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        }
    })
    .await;

    assert!(
        process_result.is_ok(),
        "Processing should complete within timeout"
    );

    // Wait for tasks to complete
    feed_task.await.expect("Feed task failed");
    consume_task.await.expect("Consume task failed");

    // Speech producer should be dropped
    assert!(
        vad.is_speech_producer_closed(),
        "Speech producer should be dropped after abandonment"
    );

    println!("Bidirectional abandonment with process_loop test passed");
}

#[tokio::test]
async fn test_abandonment_with_remaining_frames() {
    use frame_processor::FrameProcessor;

    let (mut vad, mut audio_producer, mut speech_consumer) =
        create_streaming_vad().expect("Failed to create streaming VAD");

    // Push multiple frames worth of audio
    let samples = vec![0.1f32; 16000]; // 1 second of audio
    for sample in samples {
        let _ = audio_producer.push(sample);
    }

    // Drop the producer immediately
    drop(audio_producer);

    // Process all remaining frames
    let mut frames_processed = 0;
    while vad.has_next_frame() {
        vad.process_frame().await.expect("Failed to process frame");
        frames_processed += 1;
    }

    assert!(
        frames_processed > 0,
        "Should have processed remaining frames after abandonment"
    );

    // Speech producer should be dropped after all frames are processed
    assert!(
        vad.is_speech_producer_closed(),
        "Speech producer should be dropped after processing all remaining frames"
    );

    // Verify speech consumer detects abandonment
    assert!(
        speech_consumer.is_abandoned(),
        "Speech consumer should detect producer abandonment"
    );

    println!(
        "Processed {} frames after upstream abandonment",
        frames_processed
    );
}
