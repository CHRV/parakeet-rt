use hound;
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
            .samples::<i16>()
            .map(|s| s.map(|s| s as f32 / 32768.0))
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

    // Show some audio statistics
    let max_val = resampled.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
    let rms = (resampled.iter().map(|&x| x * x).sum::<f32>() / resampled.len() as f32).sqrt();
    println!("  Max amplitude: {:.4}", max_val);
    println!("  RMS amplitude: {:.4}", rms);

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

    println!("=== DEBUG TRANSCRIPTION ===");
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
        return Ok(());
    }
    let vocabulary = Vocabulary::from_file(vocab_path)?;
    println!(
        "  Loaded vocabulary with {} tokens",
        vocabulary.id_to_token.len()
    );

    // Show some vocabulary samples
    println!("  Sample tokens:");
    for i in 0..10.min(vocabulary.id_to_token.len()) {
        if !vocabulary.id_to_token[i].is_empty() {
            println!("    {}: '{}'", i, vocabulary.id_to_token[i]);
        }
    }

    // Load TDT model
    println!("\n3. Loading TDT model...");
    let model_dir = "models";
    let exec_config = ExecutionConfig::default();
    let mut model = ParakeetTDTModel::from_pretrained(model_dir, exec_config)?;
    println!("  Model loaded successfully");

    // Run inference step by step
    println!("\n4. Running preprocessing...");
    let (features, features_len) = model.preprocess(audio_data, audio_lengths)?;
    println!("  Features shape: {:?}", features.shape());
    println!("  Features length: {:?}", features_len);

    println!("\n5. Running encoder...");
    let (encoder_out, encoder_len) = model.encode(features, features_len)?;
    println!("  Encoder output shape: {:?}", encoder_out.shape());
    println!("  Encoder length: {:?}", encoder_len);

    println!("\n6. Running decoding...");
    let results = model.decoding(encoder_out, encoder_len)?;
    println!("  Number of sequences: {}", results.len());

    if results.is_empty() {
        println!("  No sequences detected");
        return Ok(());
    }

    // Process results for each sequence
    for (seq_idx, (tokens, timestamps)) in results.iter().enumerate() {
        println!("\n7. Processing sequence {}...", seq_idx + 1);
        println!("  Raw tokens: {:?}", tokens);
        println!("  Timestamps: {:?}", timestamps);

        if tokens.is_empty() {
            println!("  No tokens detected in this sequence");
            continue;
        }

        // Convert i32 tokens to usize for decoder
        let tokens_usize: Vec<usize> = tokens.iter().map(|&t| t as usize).collect();

        // Show token meanings
        println!("  Token meanings:");
        for (i, &token_id) in tokens_usize.iter().enumerate() {
            if let Some(token_text) = vocabulary.id_to_text(token_id) {
                println!("    {}: {} -> '{}'", i, token_id, token_text);
            } else {
                println!("    {}: {} -> <unknown>", i, token_id);
            }
        }

        // Create decoder
        let decoder = ParakeetTDTDecoder::from_vocab(vocabulary.clone());

        // Decode with timestamps
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
        } else {
            println!("No valid tokens after filtering");
        }
    }

    println!("\n=== DEBUG COMPLETE ===");
    Ok(())
}
