use hound::WavReader;
use rtrb::{Consumer, Producer};
use std::time::Duration;
use tokio::time::sleep;
use vad::{
    StreamingVad,
    silero::Silero,
    utils::{SampleRate, VadParams},
};

const MODEL_PATH: &str = "../../models/silero_vad.onnx";

fn load_wav_samples(path: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let mut reader = WavReader::open(path)?;
    let samples: Result<Vec<i16>, _> = reader.samples().collect();

    Ok(samples?
        .iter()
        .map(|x| (*x as f32) / (i16::MAX as f32))
        .collect::<Vec<_>>())
}

fn create_streaming_vad() -> Result<(StreamingVad, Producer<f32>, Consumer<f32>), ort::Error> {
    let silero = Silero::new(SampleRate::SixteenkHz, MODEL_PATH)?;
    let params = VadParams::default();
    Ok(StreamingVad::new(silero, params))
}

async fn process_audio_and_collect_speech(
    vad: &mut StreamingVad,
    samples: &[f32],
    audio_producer: &mut Producer<f32>,
    speech_consumer: &mut Consumer<f32>,
) -> Vec<f32> {
    let chunk_size = 1600; // 100ms at 16kHz
    let mut speech_samples = Vec::new();

    // Process audio in chunks to simulate streaming
    for chunk in samples.chunks(chunk_size) {
        // Push audio samples to VAD input ring buffer
        for sample in chunk {
            let _ = audio_producer.push(*sample);
        }

        // Process available frames through VAD
        vad.process_audio().expect("Failed to process audio");

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

    speech_samples
}

#[tokio::test]
async fn test_speech_detection_sample_1() {
    let (mut vad, mut audio_producer, mut speech_consumer) =
        create_streaming_vad().expect("Failed to create streaming VAD");

    let samples =
        load_wav_samples("tests/audio/sample_1.wav").expect("Failed to load sample_1.wav");

    let speech_samples = process_audio_and_collect_speech(
        &mut vad,
        &samples,
        &mut audio_producer,
        &mut speech_consumer,
    )
    .await;
    let speeches = vad.speeches();

    // sample_1.wav should contain speech
    assert!(
        !speeches.is_empty(),
        "Expected to detect speech in sample_1.wav"
    );

    // Should have collected speech samples
    assert!(
        !speech_samples.is_empty(),
        "Expected to collect speech samples from sample_1.wav"
    );

    // Verify that detected speech segments have valid timestamps
    for speech in speeches {
        assert!(
            speech.start >= 0,
            "Speech start time should be non-negative"
        );
        assert!(
            speech.end > speech.start,
            "Speech end time should be after start time"
        );
        println!("Detected speech: {}", speech);
    }

    println!(
        "Total speech segments detected in sample_1.wav: {}",
        speeches.len()
    );
    println!("Total speech samples collected: {}", speech_samples.len());

    let duration_seconds = speech_samples.len() as f32 / 16000.0;
    println!("Speech duration: {:.2} seconds", duration_seconds);
}

#[tokio::test]
async fn test_no_speech_detection_birds() {
    let (mut vad, mut audio_producer, mut speech_consumer) =
        create_streaming_vad().expect("Failed to create streaming VAD");

    let samples = load_wav_samples("tests/audio/birds.wav").expect("Failed to load birds.wav");

    let speech_samples = process_audio_and_collect_speech(
        &mut vad,
        &samples,
        &mut audio_producer,
        &mut speech_consumer,
    )
    .await;
    let speeches = vad.speeches();

    // birds.wav should not contain speech
    assert!(
        speeches.is_empty(),
        "Expected no speech detection in birds.wav, but found {} segments",
        speeches.len()
    );

    // Should not have collected any speech samples
    assert!(
        speech_samples.is_empty(),
        "Should not have collected speech samples for birds.wav, but got {} samples",
        speech_samples.len()
    );

    println!("No speech detected in birds.wav (as expected)");
}

#[tokio::test]
async fn test_no_speech_detection_rooster() {
    let (mut vad, mut audio_producer, mut speech_consumer) =
        create_streaming_vad().expect("Failed to create streaming VAD");

    let samples = load_wav_samples("tests/audio/rooster.wav").expect("Failed to load rooster.wav");

    let speech_samples = process_audio_and_collect_speech(
        &mut vad,
        &samples,
        &mut audio_producer,
        &mut speech_consumer,
    )
    .await;
    let speeches = vad.speeches();

    // rooster.wav should not contain speech
    assert!(
        speeches.is_empty(),
        "Expected no speech detection in rooster.wav, but found {} segments",
        speeches.len()
    );

    // Should not have collected any speech samples
    assert!(
        speech_samples.is_empty(),
        "Should not have collected speech samples for rooster.wav, but got {} samples",
        speech_samples.len()
    );

    println!("No speech detected in rooster.wav (as expected)");
}

#[tokio::test]
async fn test_vad_with_custom_params() {
    let silero =
        Silero::new(SampleRate::SixteenkHz, MODEL_PATH).expect("Failed to create Silero model");

    let custom_params = VadParams {
        frame_size: 32,
        threshold: 0.3,
        min_silence_duration_ms: 100,
        speech_pad_ms: 32,
        min_speech_duration_ms: 100,
        max_speech_duration_s: 10.0,
        sample_rate: 16000,
    };

    let (mut vad, mut audio_producer, mut speech_consumer) =
        StreamingVad::new(silero, custom_params);
    let samples =
        load_wav_samples("tests/audio/sample_1.wav").expect("Failed to load sample_1.wav");

    let speech_samples = process_audio_and_collect_speech(
        &mut vad,
        &samples,
        &mut audio_producer,
        &mut speech_consumer,
    )
    .await;
    let speeches = vad.speeches();

    // Should still detect speech with custom parameters
    assert!(
        !speeches.is_empty(),
        "Expected to detect speech with custom parameters"
    );

    // Should have collected speech samples
    assert!(
        !speech_samples.is_empty(),
        "Expected to collect speech samples with custom parameters"
    );

    println!("Speech segments with custom params: {}", speeches.len());
    for speech in speeches {
        println!("Custom params speech: {}", speech);
    }

    let duration_seconds = speech_samples.len() as f32 / 16000.0;
    println!(
        "Speech duration with custom params: {:.2} seconds",
        duration_seconds
    );
}

#[tokio::test]
async fn test_empty_audio() {
    let (mut vad, mut audio_producer, mut speech_consumer) =
        create_streaming_vad().expect("Failed to create streaming VAD");

    let empty_samples: Vec<f32> = vec![];

    let speech_samples = process_audio_and_collect_speech(
        &mut vad,
        &empty_samples,
        &mut audio_producer,
        &mut speech_consumer,
    )
    .await;
    let speeches = vad.speeches();

    assert!(
        speeches.is_empty(),
        "Expected no speech detection in empty audio"
    );

    // Should not have collected any speech samples
    assert!(
        speech_samples.is_empty(),
        "Should not collect speech samples for empty audio"
    );

    println!("Empty audio test passed - no speech detected");
}

#[tokio::test]
async fn test_silence_audio() {
    let (mut vad, mut audio_producer, mut speech_consumer) =
        create_streaming_vad().expect("Failed to create streaming VAD");

    // Create 1 second of silence at 16kHz
    let silence_samples: Vec<f32> = vec![0f32; 16000];

    let speech_samples = process_audio_and_collect_speech(
        &mut vad,
        &silence_samples,
        &mut audio_producer,
        &mut speech_consumer,
    )
    .await;
    let speeches = vad.speeches();

    assert!(
        speeches.is_empty(),
        "Expected no speech detection in silence"
    );

    // Should not have collected any speech samples
    assert!(
        speech_samples.is_empty(),
        "Should not collect speech samples for silence"
    );

    println!("Silence audio test passed - no speech detected");
}

#[tokio::test]
async fn test_multiple_processing_calls() {
    let samples =
        load_wav_samples("tests/audio/sample_1.wav").expect("Failed to load sample_1.wav");

    // First processing
    let (mut vad1, mut audio_producer1, mut speech_consumer1) =
        create_streaming_vad().expect("Failed to create first streaming VAD");

    let speech_samples1 = process_audio_and_collect_speech(
        &mut vad1,
        &samples,
        &mut audio_producer1,
        &mut speech_consumer1,
    )
    .await;
    let first_speeches = vad1.speeches().len();

    // Second processing (fresh VAD instance)
    let (mut vad2, mut audio_producer2, mut speech_consumer2) =
        create_streaming_vad().expect("Failed to create second streaming VAD");

    let speech_samples2 = process_audio_and_collect_speech(
        &mut vad2,
        &samples,
        &mut audio_producer2,
        &mut speech_consumer2,
    )
    .await;
    let second_speeches = vad2.speeches().len();

    // Results should be consistent
    assert_eq!(
        first_speeches, second_speeches,
        "Speech detection should be consistent across multiple calls"
    );

    // Speech samples should be similar (allowing for small variations due to processing)
    let samples_diff = (speech_samples1.len() as i32 - speech_samples2.len() as i32).abs();
    assert!(
        samples_diff < 1000, // Allow up to 1000 samples difference (about 62ms at 16kHz)
        "Speech samples should be similar across multiple calls, but got {} vs {} samples (diff: {})",
        speech_samples1.len(),
        speech_samples2.len(),
        samples_diff
    );

    println!(
        "Multiple processing test: {} vs {} speeches, {} vs {} samples",
        first_speeches,
        second_speeches,
        speech_samples1.len(),
        speech_samples2.len()
    );
}
