/// Simple streaming example demonstrating the key concepts
/// This example shows how to use the streaming API with batch_size=1
use parakeet::execution::ModelConfig as ExecutionConfig;
use parakeet::parakeet_tdt::ParakeetTDTModel;
use parakeet::streaming::{ContextConfig, StreamingParakeetTDT};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Simple Streaming Parakeet TDT Example ===");
    println!("This demonstrates streaming inference with batch_size=1");

    // Check if models exist
    let model_dir = "models";
    if !Path::new(model_dir).exists() {
        eprintln!("Error: Models directory not found: {}", model_dir);
        eprintln!("Please ensure ONNX models are in the models/ directory");
        return Ok(());
    }

    // Load model
    println!("\n1. Loading TDT model...");
    let exec_config = ExecutionConfig::default();
    let model = ParakeetTDTModel::from_pretrained(model_dir, exec_config)?;
    println!("   ✓ Model loaded successfully");

    // Configure streaming with low latency settings
    let sample_rate = 16000;
    let context = ContextConfig::new(
        1.0,   // 1 second left context (for quality)
        0.25,  // 250ms chunks (low latency)
        0.25,  // 250ms right context
        sample_rate,
    );

    println!("\n2. Streaming configuration:");
    println!("   • Left context: {:.1}s (improves quality)",
        context.left_samples as f32 / sample_rate as f32);
    println!("   • Chunk size: {:.0}ms (processing latency)",
        context.chunk_samples as f32 / sample_rate as f32 * 1000.0);
    println!("   • Right context: {:.0}ms",
        context.right_samples as f32 / sample_rate as f32 * 1000.0);
    println!("   • Total latency: {:.0}ms",
        context.latency_secs(sample_rate) * 1000.0);

    // Create streaming engine (batch_size is always 1)
    let (mut streaming_engine, mut audio_producer, mut token_consumer) =
        StreamingParakeetTDT::new(model, context.clone(), sample_rate);

    // Generate some test audio (sine wave for demonstration)
    println!("\n3. Generating test audio (2 second sine wave)...");
    let duration_secs = 2.0;
    let frequency = 440.0; // A4 note
    let total_samples = (duration_secs * sample_rate as f32) as usize;

    let test_audio: Vec<f32> = (0..total_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            0.1 * (2.0 * std::f32::consts::PI * frequency * t).sin()
        })
        .collect();

    println!("   ✓ Generated {} samples ({:.1}s)", test_audio.len(), duration_secs);

    // Simulate real-time streaming
    println!("\n4. Streaming inference simulation:");
    println!("   Feeding audio in 50ms chunks to simulate real-time...");

    let realtime_chunk_size = sample_rate / 20; // 50ms chunks
    let mut total_processed = 0;
    let mut all_tokens = Vec::new();

    for (chunk_idx, chunk_start) in (0..test_audio.len()).step_by(realtime_chunk_size).enumerate() {
        let chunk_end = std::cmp::min(chunk_start + realtime_chunk_size, test_audio.len());
        let chunk = &test_audio[chunk_start..chunk_end];

        // Feed audio to producer
        for &sample in chunk {
            if audio_producer.push(sample).is_err() {
                // Buffer full, skip sample
                break;
            }
        }
        total_processed += chunk.len();

        // Process available audio
        streaming_engine.process_audio()?;

        // Read any new tokens from consumer
        let mut new_tokens = Vec::new();
        while let Ok(token_result) = token_consumer.pop() {
            new_tokens.push(token_result);
        }

        let time_processed = total_processed as f32 / sample_rate as f32;

        if !new_tokens.is_empty() {
            println!("   [{:5.2}s] Chunk {:2}: {} new tokens detected",
                time_processed, chunk_idx + 1, new_tokens.len());

            // Show token details
            for token_result in &new_tokens {
                let token_time = token_result.timestamp as f32 / sample_rate as f32;
                println!("     Token {} at {:.3}s (conf: {:.2})",
                    token_result.token_id, token_time, token_result.confidence);
            }
            all_tokens.extend(new_tokens);
        } else {
            println!("   [{:5.2}s] Chunk {:2}: no tokens", time_processed, chunk_idx + 1);
        }
    }

    // Finish stream and get final results
    streaming_engine.finalize();

    // Read any final tokens
    let mut final_tokens = Vec::new();
    while let Ok(token_result) = token_consumer.pop() {
        final_tokens.push(token_result);
    }

    if !final_tokens.is_empty() {
        println!("   [Final] {} additional tokens from remaining audio", final_tokens.len());
        all_tokens.extend(final_tokens);
    }

    println!("\n5. Results Summary:");
    println!("   • Total tokens detected: {}", all_tokens.len());
    println!("   • Processing completed for {:.1}s of audio", duration_secs);

    if !all_tokens.is_empty() {
        let token_ids: Vec<i32> = all_tokens.iter().map(|t| t.token_id).collect();
        println!("   • Token IDs: {:?}", &token_ids[..token_ids.len().min(10)]);
        if token_ids.len() > 10 {
            println!("     ... and {} more", token_ids.len() - 10);
        }

        println!("   • First timestamp: {:.3}s", all_tokens[0].timestamp as f32 / sample_rate as f32);
        println!("   • Last timestamp: {:.3}s",
            all_tokens.last().map(|t| t.timestamp).unwrap_or(0) as f32 / sample_rate as f32);
    }

    println!("\n=== Key Streaming Concepts Demonstrated ===");
    println!("✓ Batch size is always 1 (single audio stream)");
    println!("✓ Context-based chunking (left + chunk + right)");
    println!("✓ Incremental processing with state management");
    println!("✓ Low-latency inference ({}ms theoretical latency)",
        (context.latency_secs(sample_rate) * 1000.0) as u32);
    println!("✓ Real-time audio feeding simulation");

    Ok(())
}