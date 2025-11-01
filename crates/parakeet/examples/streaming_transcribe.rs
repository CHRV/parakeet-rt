use hound;
//use ndarray::{Array1, Array2};
use parakeet::decoder::ParakeetTDTDecoder;
use parakeet::execution::ModelConfig as ExecutionConfig;
use parakeet::parakeet_tdt::ParakeetTDTModel;
use parakeet::streaming::{ContextConfig, StreamingParakeetTDT};
use parakeet::vocabulary::Vocabulary;
use std::env;
use std::path::Path;

fn load_wav_file(path: &Path) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();

    println!("Audio file info:");
    println!("  Sample rate: {} Hz", spec.sample_rate);
    println!("  Channels: {}", spec.channels);
    println!("  Bits per sample: {}", spec.bits_per_sample);

    // Read all samples
    let samples: Result<Vec<f32>, _> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect(),
        hound::SampleFormat::Int => reader
            .samples::<i32>()
            .map(|s| s.map(|sample| sample as f32 / i32::MAX as f32))
            .collect(),
    };

    let samples = samples?;
    let duration = samples.len() as f32 / spec.sample_rate as f32;
    println!("  Duration: {:.2} seconds", duration);
    println!("  Total samples: {}", samples.len());

    // Convert to mono if stereo
    let mono_samples = if spec.channels == 2 {
        samples
            .chunks(2)
            .map(|chunk| (chunk[0] + chunk[1]) / 2.0)
            .collect::<Vec<f32>>()
    } else {
        samples
    };

    // Resample to 16kHz if needed (simple decimation for demo)
    let target_sample_rate = 16000;
    let resampled = if spec.sample_rate != target_sample_rate {
        let ratio = spec.sample_rate as f32 / target_sample_rate as f32;
        let mut resampled = Vec::new();
        let mut i = 0.0;
        while (i as usize) < mono_samples.len() {
            resampled.push(mono_samples[i as usize]);
            i += ratio;
        }
        println!(
            "  Resampled from {} Hz to {} Hz",
            spec.sample_rate, target_sample_rate
        );
        resampled
    } else {
        mono_samples
    };

    Ok(resampled)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    // Default to sample_1.wav if no argument provided
    let audio_path = if args.len() > 1 {
        &args[1]
    } else {
        "audio/sample_1.wav"
    };

    println!("Streaming transcription of audio file: {}", audio_path);

    // Check if audio file exists
    let audio_path = Path::new(audio_path);
    if !audio_path.exists() {
        eprintln!("Error: Audio file not found: {}", audio_path.display());
        eprintln!("Available test files:");
        eprintln!("  ../../audio/sample_1.wav");
        eprintln!("  ../../audio/birds.wav");
        eprintln!("  ../../audio/rooster.wav");
        return Ok(());
    }

    // Load audio
    println!("\n1. Loading audio file...");
    let audio_samples = load_wav_file(audio_path)?;

    // Load vocabulary
    println!("\n2. Loading vocabulary...");
    let vocab_path = "models/vocab.txt";
    if !Path::new(vocab_path).exists() {
        eprintln!("Error: Vocabulary file not found: {}", vocab_path);
        eprintln!("Please ensure the vocab.txt file is in the models/ directory");
        return Ok(());
    }
    let vocabulary = Vocabulary::from_file(vocab_path)?;
    println!(
        "  Loaded vocabulary with {} tokens",
        vocabulary.id_to_token.len()
    );

    // Load TDT model
    println!("\n3. Loading TDT model...");
    let model_dir = "models";
    if !Path::new(model_dir).exists() {
        eprintln!("Error: Models directory not found: {}", model_dir);
        return Ok(());
    }

    let exec_config = ExecutionConfig::default();
    let model = ParakeetTDTModel::from_pretrained(model_dir, exec_config)?;
    println!("  Model loaded successfully");

    // Configure streaming parameters
    let sample_rate = 16000;
    let context = ContextConfig::new(
        2.0,  // 2 seconds left context
        0.5,  // 0.5 second chunks (500ms latency + right context)
        0.5,  // 0.5 second right context
        sample_rate,
    );

    println!("\n4. Streaming configuration:");
    println!("  Left context: {:.1}s", context.left_samples as f32 / sample_rate as f32);
    println!("  Chunk size: {:.1}s", context.chunk_samples as f32 / sample_rate as f32);
    println!("  Right context: {:.1}s", context.right_samples as f32 / sample_rate as f32);
    println!("  Theoretical latency: {:.1}s", context.latency_secs(sample_rate));

    // Create streaming engine
    let (mut streaming_engine, mut audio_producer, mut token_consumer) =
        StreamingParakeetTDT::new(model, context, sample_rate);

    // Simulate streaming by feeding audio in chunks
    println!("\n5. Starting streaming inference...");
    let chunk_size = sample_rate / 10; // 100ms chunks for simulation
    let mut processed_samples = 0;
    let mut all_tokens = Vec::new();

    for chunk_start in (0..audio_samples.len()).step_by(chunk_size) {
        let chunk_end = std::cmp::min(chunk_start + chunk_size, audio_samples.len());
        let chunk = &audio_samples[chunk_start..chunk_end];

        // Add audio to ring buffer
        for &sample in chunk {
            if audio_producer.push(sample).is_err() {
                println!("Warning: Audio buffer full, dropping samples");
                break;
            }
        }
        processed_samples += chunk.len();

        // Process available audio
        streaming_engine.process_audio()?;

        // Read any new tokens from the output buffer
        let mut new_tokens = Vec::new();
        while let Ok(token_result) = token_consumer.pop() {
            new_tokens.push((token_result.token_id, token_result.timestamp));
        }

        if !new_tokens.is_empty() {
            println!("  [{:.2}s] New tokens: {:?}",
                processed_samples as f32 / sample_rate as f32,
                new_tokens.iter().map(|(token, _)| *token).collect::<Vec<_>>()
            );
            all_tokens.extend(new_tokens);
        }

        // Show progress
        if chunk_start % (sample_rate * 2) == 0 { // Every 2 seconds
            println!("  Processed {:.1}s / {:.1}s",
                processed_samples as f32 / sample_rate as f32,
                audio_samples.len() as f32 / sample_rate as f32
            );
        }
    }

    // Finish the stream and process remaining audio
    streaming_engine.finalize();

    // Read any final tokens
    let mut final_tokens = Vec::new();
    while let Ok(token_result) = token_consumer.pop() {
        final_tokens.push((token_result.token_id, token_result.timestamp));
    }
    if !final_tokens.is_empty() {
        println!("  [Final] New tokens: {:?}",
            final_tokens.iter().map(|(token, _)| *token).collect::<Vec<_>>()
        );
        all_tokens.extend(final_tokens);
    }

    // Extract tokens and timestamps from collected results
    let tokens: Vec<i32> = all_tokens.iter().map(|(token, _)| *token).collect();
    let timestamps: Vec<usize> = all_tokens.iter().map(|(_, timestamp)| *timestamp).collect();

    println!("\n=== STREAMING TRANSCRIPTION RESULTS ===");
    println!("Total tokens detected: {}", tokens.len());

    if !tokens.is_empty() {
        // Convert i32 tokens to usize for decoder
        let tokens_usize: Vec<usize> = tokens.iter().map(|&t| t as usize).collect();

        // Create decoder
        let decoder = ParakeetTDTDecoder::from_vocab(vocabulary.clone());

        // Decode with timestamps
        let hop_length = 160; // 10ms at 16kHz
        let durations = vec![1; tokens.len()]; // Placeholder durations

        let transcription = decoder.decode_with_timestamps(
            &tokens_usize,
            &timestamps,
            &durations,
            hop_length,
            sample_rate,
        )?;

        println!("Final transcription: \"{}\"", transcription.text);

        if !transcription.tokens.is_empty() {
            println!("\nToken timeline:");
            for (i, token) in transcription.tokens.iter().enumerate() {
                println!(
                    "  [{:3}] {:.3}s: \"{}\"",
                    i + 1,
                    token.start,
                    token.text
                );
            }
        }

        // Show streaming statistics
        println!("\nStreaming statistics:");
        println!("  Total processing chunks: {}", (audio_samples.len() + chunk_size - 1) / chunk_size);
        println!("  Average tokens per second: {:.1}",
            tokens.len() as f32 / (audio_samples.len() as f32 / sample_rate as f32)
        );
    } else {
        println!("No speech detected in the audio file.");
    }

    println!("\n=== STREAMING COMPLETE ===");
    Ok(())
}