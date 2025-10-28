use hound::WavReader;
use rtrb::Consumer;
use std::time::Duration;
use tokio::time::sleep;
use vad::{
    StreamingVad, VadEvent,
    silero::Silero,
    utils::{SampleRate, VadParams},
};

const MODEL_PATH: &str = "../../models/silero_vad.onnx";

fn load_wav_samples(path: &str) -> Result<Vec<i16>, Box<dyn std::error::Error>> {
    let mut reader = WavReader::open(path)?;
    let samples: Result<Vec<i16>, _> = reader.samples().collect();
    Ok(samples?)
}

fn create_streaming_vad() -> Result<(StreamingVad, Consumer<VadEvent>), ort::Error> {
    let silero = Silero::new(SampleRate::SixteenkHz, MODEL_PATH)?;
    let params = VadParams::default();
    Ok(StreamingVad::new(silero, params))
}

async fn process_audio_and_collect_events(
    vad: &mut StreamingVad,
    samples: &[i16],
    event_consumer: &mut Consumer<VadEvent>,
) -> Vec<VadEvent> {
    let chunk_size = 1600; // 100ms at 16kHz
    let mut events = Vec::new();

    // Process audio in chunks to simulate streaming
    for chunk in samples.chunks(chunk_size) {
        vad.process_audio(chunk)
            .expect("Failed to process audio chunk");

        // Collect any events that are available
        while let Ok(event) = event_consumer.pop() {
            events.push(event);
        }
    }

    // Finalize to ensure any ongoing speech is completed
    vad.finalize();

    // Give a small delay for final events to be processed
    sleep(Duration::from_millis(10)).await;

    // Collect final events
    while let Ok(event) = event_consumer.pop() {
        events.push(event);
    }

    events
}

#[tokio::test]
async fn test_speech_detection_sample_1() {
    let (mut vad, mut event_consumer) =
        create_streaming_vad().expect("Failed to create streaming VAD");

    let samples =
        load_wav_samples("tests/audio/sample_1.wav").expect("Failed to load sample_1.wav");

    let events = process_audio_and_collect_events(&mut vad, &samples, &mut event_consumer).await;
    let speeches = vad.speeches();

    // sample_1.wav should contain speech
    assert!(
        !speeches.is_empty(),
        "Expected to detect speech in sample_1.wav"
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

    // Verify we got the right events
    let speech_started_count = events
        .iter()
        .filter(|e| matches!(e, VadEvent::SpeechStarted { .. }))
        .count();
    let speech_ended_count = events
        .iter()
        .filter(|e| matches!(e, VadEvent::SpeechEnded { .. }))
        .count();

    assert!(
        speech_started_count > 0,
        "Expected at least one speech start event"
    );
    assert!(
        speech_ended_count > 0,
        "Expected at least one speech end event"
    );
    assert_eq!(
        speech_ended_count,
        speeches.len(),
        "Should have one end event per speech segment"
    );

    println!(
        "Total speech segments detected in sample_1.wav: {}",
        speeches.len()
    );
    println!(
        "Speech start events: {}, Speech end events: {}",
        speech_started_count, speech_ended_count
    );
}

#[tokio::test]
async fn test_no_speech_detection_birds() {
    let (mut vad, mut event_consumer) =
        create_streaming_vad().expect("Failed to create streaming VAD");

    let samples = load_wav_samples("tests/audio/birds.wav").expect("Failed to load birds.wav");

    let events = process_audio_and_collect_events(&mut vad, &samples, &mut event_consumer).await;
    let speeches = vad.speeches();

    // birds.wav should not contain speech
    assert!(
        speeches.is_empty(),
        "Expected no speech detection in birds.wav, but found {} segments",
        speeches.len()
    );

    // Should not have any speech events
    let speech_events = events
        .iter()
        .filter(|e| matches!(e, VadEvent::SpeechEnded { .. }))
        .count();
    assert_eq!(
        speech_events, 0,
        "Should not have any speech events for birds.wav"
    );

    println!("No speech detected in birds.wav (as expected)");
}

#[tokio::test]
async fn test_no_speech_detection_rooster() {
    let (mut vad, mut event_consumer) =
        create_streaming_vad().expect("Failed to create streaming VAD");

    let samples = load_wav_samples("tests/audio/rooster.wav").expect("Failed to load rooster.wav");

    let events = process_audio_and_collect_events(&mut vad, &samples, &mut event_consumer).await;
    let speeches = vad.speeches();

    // rooster.wav should not contain speech
    assert!(
        speeches.is_empty(),
        "Expected no speech detection in rooster.wav, but found {} segments",
        speeches.len()
    );

    // Should not have any speech events
    let speech_events = events
        .iter()
        .filter(|e| matches!(e, VadEvent::SpeechEnded { .. }))
        .count();
    assert_eq!(
        speech_events, 0,
        "Should not have any speech events for rooster.wav"
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

    let (mut vad, mut event_consumer) = StreamingVad::new(silero, custom_params);
    let samples =
        load_wav_samples("tests/audio/sample_1.wav").expect("Failed to load sample_1.wav");

    let events = process_audio_and_collect_events(&mut vad, &samples, &mut event_consumer).await;
    let speeches = vad.speeches();

    // Should still detect speech with custom parameters
    assert!(
        !speeches.is_empty(),
        "Expected to detect speech with custom parameters"
    );

    println!("Speech segments with custom params: {}", speeches.len());
    for speech in speeches {
        println!("Custom params speech: {}", speech);
    }

    // Verify events were generated
    let speech_events = events
        .iter()
        .filter(|e| matches!(e, VadEvent::SpeechEnded { .. }))
        .count();
    assert_eq!(
        speech_events,
        speeches.len(),
        "Should have events for all detected speeches"
    );
}

#[tokio::test]
async fn test_empty_audio() {
    let (mut vad, mut event_consumer) =
        create_streaming_vad().expect("Failed to create streaming VAD");

    let empty_samples: Vec<i16> = vec![];

    vad.process_audio(&empty_samples)
        .expect("Failed to process empty audio");
    vad.finalize();

    let speeches = vad.speeches();
    assert!(
        speeches.is_empty(),
        "Expected no speech detection in empty audio"
    );

    // Should not receive any events
    assert!(
        event_consumer.pop().is_err(),
        "Should not receive events for empty audio"
    );
}

#[tokio::test]
async fn test_silence_audio() {
    let (mut vad, mut event_consumer) =
        create_streaming_vad().expect("Failed to create streaming VAD");

    // Create 1 second of silence at 16kHz
    let silence_samples: Vec<i16> = vec![0; 16000];

    vad.process_audio(&silence_samples)
        .expect("Failed to process silence audio");
    vad.finalize();

    let speeches = vad.speeches();
    assert!(
        speeches.is_empty(),
        "Expected no speech detection in silence"
    );

    // Should not receive any events
    assert!(
        event_consumer.pop().is_err(),
        "Should not receive events for silence"
    );
}

#[tokio::test]
async fn test_multiple_processing_calls() {
    let samples =
        load_wav_samples("tests/audio/sample_1.wav").expect("Failed to load sample_1.wav");

    // First processing
    let (mut vad1, mut event_consumer1) =
        create_streaming_vad().expect("Failed to create first streaming VAD");

    let events1 = process_audio_and_collect_events(&mut vad1, &samples, &mut event_consumer1).await;
    let first_speeches = vad1.speeches().len();

    // Second processing (fresh VAD instance)
    let (mut vad2, mut event_consumer2) =
        create_streaming_vad().expect("Failed to create second streaming VAD");

    let events2 = process_audio_and_collect_events(&mut vad2, &samples, &mut event_consumer2).await;
    let second_speeches = vad2.speeches().len();

    // Results should be consistent
    assert_eq!(
        first_speeches, second_speeches,
        "Speech detection should be consistent across multiple calls"
    );

    let speech_events1 = events1
        .iter()
        .filter(|e| matches!(e, VadEvent::SpeechEnded { .. }))
        .count();
    let speech_events2 = events2
        .iter()
        .filter(|e| matches!(e, VadEvent::SpeechEnded { .. }))
        .count();

    assert_eq!(
        speech_events1, speech_events2,
        "Should generate same number of events across multiple calls"
    );
}
