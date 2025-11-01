use ndarray::Array1;
use parakeet::error::Result;
use parakeet::execution::ModelConfig as ExecutionConfig;
use parakeet::parakeet_tdt::ParakeetTDTModel;
use std::fs::{File, create_dir_all};
use std::io::Write;
use std::path::Path;

/// Save array data in numpy-compatible format and generate comparison files
fn save_debug_array<D>(
    array: &ndarray::ArrayBase<ndarray::OwnedRepr<f32>, D>,
    filename: &str,
    description: &str,
) -> Result<()>
where
    D: ndarray::Dimension,
{
    let output_dir = Path::new("../../python/debug_outputs");
    create_dir_all(output_dir)?;

    // Convert to flat vector for saving
    let data: Vec<f32> = array.iter().cloned().collect();
    let shape: Vec<usize> = array.shape().to_vec();

    // Save as numpy-compatible binary (simple format)
    let rust_filename = format!("rust_{}.npy", filename);
    let _npy_path = output_dir.join(&rust_filename);

    // Create a simple numpy file (we'll use a basic format)
    // For now, save as raw binary and create a Python script to convert
    let raw_path = output_dir.join(format!("rust_{}.raw", filename));
    let mut file = File::create(&raw_path)?;

    // Write shape first (as u64 values)
    file.write_all(&(shape.len() as u64).to_le_bytes())?;
    for &dim in &shape {
        file.write_all(&(dim as u64).to_le_bytes())?;
    }

    // Write data
    for &value in &data {
        file.write_all(&value.to_le_bytes())?;
    }

    // Create stats JSON
    let min_val = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mean_val = data.iter().sum::<f32>() / data.len() as f32;
    let variance = data.iter().map(|&x| (x - mean_val).powi(2)).sum::<f32>() / data.len() as f32;
    let std_val = variance.sqrt();

    let stats_content = format!(
        r#"{{
  "description": "Rust: {}",
  "shape": {:?},
  "dtype": "float32",
  "min": {},
  "max": {},
  "mean": {},
  "std": {}
}}"#,
        description, shape, min_val, max_val, mean_val, std_val
    );

    let stats_path = output_dir.join(format!("rust_{}_stats.json", filename));
    std::fs::write(&stats_path, stats_content)?;

    // Create sample text file
    let sample_size = 20.min(data.len());
    let mut sample_content = format!(
        "Shape: {:?}\nDtype: float32\nFirst {} values:\n",
        shape, sample_size
    );
    for (i, &value) in data.iter().take(sample_size).enumerate() {
        sample_content.push_str(&format!("[{}]: {}\n", i, value));
    }

    let sample_path = output_dir.join(format!("rust_{}_sample.txt", filename));
    std::fs::write(&sample_path, sample_content)?;

    println!(
        "Saved rust_{}: shape={:?}, range=[{:.6}, {:.6}]",
        filename, shape, min_val, max_val
    );

    Ok(())
}

/// Load WAV file (simple implementation)
fn load_wav_simple(path: &str) -> Result<(Array1<f32>, u32)> {
    use hound::WavReader;

    let mut reader = WavReader::open(path).map_err(|e| {
        parakeet::error::Error::Io(std::io::Error::other(
            e.to_string(),
        ))
    })?;

    let spec = reader.spec();
    let samples: std::result::Result<Vec<f32>, hound::Error> = reader
        .samples::<i16>()
        .map(|s| s.map(|sample| sample as f32 / 32768.0))
        .collect();

    let samples = samples.map_err(|e| {
        parakeet::error::Error::Io(std::io::Error::other(
            e.to_string(),
        ))
    })?;

    let audio = Array1::from_vec(samples);
    Ok((audio, spec.sample_rate))
}

fn main() -> Result<()> {
    println!("=== Rust TDT Model Debug Output Generation ===");

    // 1. Load model
    println!("1. Loading TDT model...");
    let model_dir = "../../models";
    let exec_config = ExecutionConfig::default();
    let mut model = ParakeetTDTModel::from_pretrained(model_dir, exec_config)?;
    println!("   Model loaded successfully");

    // 2. Load audio
    let audio_path = "../../audio/sample_1.wav";
    println!("\n2. Loading audio: {}", audio_path);

    let (audio, sample_rate) = load_wav_simple(audio_path)?;
    println!("   Audio shape: {:?}", audio.shape());
    println!("   Sample rate: {}", sample_rate);
    println!(
        "   Duration: {:.2}s",
        audio.len() as f32 / sample_rate as f32
    );

    // Save raw audio
    save_debug_array(
        &audio,
        "01_raw_audio",
        "Raw audio samples (float32, normalized)",
    )?;

    // Add batch dimension
    let audio_len = audio.len();
    let audio_batch = audio.insert_axis(ndarray::Axis(0));
    println!("   Audio batch shape: {:?}", audio_batch.shape());

    save_debug_array(&audio_batch, "02_audio_batch", "Audio with batch dimension")?;

    // 3. Preprocessing
    println!("\n3. Preprocessing...");
    let audio_lens = Array1::from_vec(vec![audio_len as i64]);
    let (features, features_lens) = model.preprocess(audio_batch.clone(), audio_lens.clone())?;

    println!("   Features shape: {:?}", features.shape());
    println!("   Features lens: {:?}", features_lens);
    println!("   Features dtype: float32");

    save_debug_array(
        &features,
        "03_preprocessed_features",
        "Preprocessed features from audio",
    )?;
    save_debug_array(
        &features_lens.mapv(|x| x as f32),
        "04_features_lengths",
        "Feature sequence lengths",
    )?;

    // 4. Encoding
    println!("\n4. Encoding...");
    let features_shape = features.shape().to_vec();
    let (encoder_out, encoder_out_lens) = model.encode(features.clone(), features_lens.clone())?;

    println!("   Encoder output shape: {:?}", encoder_out.shape());
    println!("   Encoder output lens: {:?}", encoder_out_lens);
    println!("   Encoder output dtype: float32");

    save_debug_array(
        &encoder_out,
        "05_encoder_output",
        "Encoder output (encoded features)",
    )?;
    save_debug_array(
        &encoder_out_lens.mapv(|x| x as f32),
        "06_encoder_lengths",
        "Encoder output sequence lengths",
    )?;

    // 5. Test full transcription
    println!("\n5. Testing full transcription...");
    match model.forward(audio_batch, audio_lens) {
        Ok(results) => {
            println!("   Results: {:?}", results);
            if let Some((tokens, timestamps)) = results.first() {
                println!("   First result tokens: {:?}", tokens);
                println!("   First result timestamps: {:?}", timestamps);

                // Save transcription result
                let output_dir = Path::new("../../python/debug_outputs");
                let result_content =
                    format!("Tokens: {:?}\nTimestamps: {:?}\n", tokens, timestamps);
                std::fs::write(
                    output_dir.join("rust_transcription_result.txt"),
                    result_content,
                )?;
            }
        }
        Err(e) => {
            println!("   Error: {}", e);
        }
    }

    // 6. Save model configuration
    let config_content = format!(
        r#"{{
  "vocab_size": 8193,
  "blank_idx": 8192,
  "preprocessor_name": "nemo128",
  "audio_sample_rate": {},
  "audio_duration_seconds": {},
  "original_audio_length": {},
  "features_shape": {:?},
  "encoder_output_shape": {:?}
}}"#,
        sample_rate,
        audio_len as f32 / sample_rate as f32,
        audio_len,
        features_shape,
        encoder_out.shape()
    );

    let output_dir = Path::new("../../python/debug_outputs");
    std::fs::write(output_dir.join("rust_model_config.json"), config_content)?;

    println!("\n=== Rust debug outputs saved to ../../python/debug_outputs/ directory ===");
    println!("Files created:");
    println!("  - rust_01_raw_audio.raw - Raw audio samples");
    println!("  - rust_02_audio_batch.raw - Audio with batch dimension");
    println!("  - rust_03_preprocessed_features.raw - Preprocessed features");
    println!("  - rust_04_features_lengths.raw - Feature lengths");
    println!("  - rust_05_encoder_output.raw - Encoder output");
    println!("  - rust_06_encoder_lengths.raw - Encoder output lengths");
    println!("  - rust_model_config.json - Model configuration");
    println!("  - rust_*_stats.json - Statistics for each array");
    println!("  - rust_*_sample.txt - Sample values for quick inspection");

    Ok(())
}
