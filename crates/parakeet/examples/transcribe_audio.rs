use ndarray::{Array1, Array2};
use parakeet::decoder::ParakeetTDTDecoder;
use parakeet::execution::ModelConfig as ExecutionConfig;
use parakeet::parakeet_tdt::ParakeetTDTModel;
use parakeet::vocabulary::Vocabulary;
use std::env;
use std::path::Path;

fn load_wav_file(path: &Path) -> Result<(Array2<f32>, i64), Box<dyn std::error::Error>> {
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

    // Create batch dimension [1, samples]
    let audio_array = Array2::from_shape_vec((1, resampled.len()), resampled)?;
    let length = audio_array.shape()[1] as i64;

    Ok((audio_array, length))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    // Default to sample_1.wav if no argument provided
    let audio_path = if args.len() > 1 {
        &args[1]
    } else {
        "../../audio/sample_1.wav"
    };

    println!("Transcribing audio file: {}", audio_path);

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
    let (audio_data, audio_length) = load_wav_file(audio_path)?;
    let audio_lengths = Array1::from_vec(vec![audio_length]);

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
    let mut model = ParakeetTDTModel::from_pretrained(model_dir, exec_config)?;
    println!("  Model loaded successfully");

    // Run inference
    println!("\n4. Running inference...");
    let results = model.forward(audio_data, audio_lengths)?;

    if results.is_empty() {
        println!("No speech detected in the audio file.");
        return Ok(());
    }

    // Process results for each sequence (typically just one)
    for (seq_idx, (tokens, timestamps)) in results.iter().enumerate() {
        println!("\n5. Decoding sequence {}...", seq_idx + 1);
        println!("  Detected {} tokens", tokens.len());

        if tokens.is_empty() {
            println!("  No tokens detected");
            continue;
        }

        // Convert i32 tokens to usize for decoder
        let tokens_usize: Vec<usize> = tokens.iter().map(|&t| t as usize).collect();

        // Create decoder
        let decoder = ParakeetTDTDecoder::from_vocab(vocabulary.clone());

        // Decode with timestamps
        // Using typical values for TDT models
        let hop_length = 160; // 10ms at 16kHz
        let sample_rate = 16000;
        let durations = vec![1; tokens.len()]; // Placeholder durations

        let transcription = decoder.decode_with_timestamps(
            &tokens_usize,
            timestamps,
            &durations,
            hop_length,
            sample_rate,
        )?;

        // Display results
        println!("\n=== TRANSCRIPTION RESULTS ===");
        println!("Full text: \"{}\"", transcription.text);

        if !transcription.tokens.is_empty() {
            println!("\nDetailed tokens with timestamps:");
            for (i, token) in transcription.tokens.iter().enumerate() {
                println!(
                    "  [{:3}] {:.3}s - {:.3}s: \"{}\"",
                    i + 1,
                    token.start,
                    token.end,
                    token.text
                );
            }
        }

        // Show raw token IDs for debugging
        println!("\nRaw token information:");
        println!("  Token IDs: {:?}", &tokens[..tokens.len().min(20)]);
        if tokens.len() > 20 {
            println!("  ... and {} more tokens", tokens.len() - 20);
        }
        println!(
            "  Timestamps: {:?}",
            &timestamps[..timestamps.len().min(20)]
        );
        if timestamps.len() > 20 {
            println!("  ... and {} more timestamps", timestamps.len() - 20);
        }
    }

    println!("\n=== TRANSCRIPTION COMPLETE ===");
    Ok(())
}
