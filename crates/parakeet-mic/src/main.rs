use anyhow::{anyhow, Result};
use clap::Parser;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, Host, Sample, SampleFormat, SampleRate, Stream, StreamConfig};
use frame_processor::FrameProcessor;
use parakeet::execution::ModelConfig as ExecutionConfig;
use parakeet::model::ParakeetTDTModel;
use parakeet::streaming::{ContextConfig, StreamingParakeetTDT, TokenResult};
use parakeet::vocab::Vocabulary;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

/// Convert a single token to text with proper spacing
fn token_to_text(token_text: &str, is_first_token: bool) -> Option<String> {
    if token_text.starts_with('<') && token_text.ends_with('>') {
        // Special tokens like <|endoftext|> - skip them for clean output
        println!("Skipping special token: {}", token_text);
        return None;
    }

    if let Some(text_part) = token_text.strip_prefix("▁") {
        // SentencePiece space prefix - replace with actual space
        // Skip the ▁ character (3 bytes in UTF-8)
        if is_first_token {
            Some(text_part.to_string())
        } else {
            Some(format!(" {}", text_part))
        }
    } else {
        // Regular token - append directly (no space)
        Some(token_text.to_string())
    }
}

mod output;
use output::{AudioWriter, OutputFormat, OutputWriter};

#[derive(Parser)]
#[command(name = "parakeet-mic")]
#[command(about = "Real-time speech transcription using microphone input")]
struct Args {
    /// Path to the models directory
    #[arg(short, long, default_value = "models")]
    models: String,

    /// Sample rate for audio capture
    #[arg(short, long, default_value = "16000")]
    sample_rate: u32,

    /// Left context duration in seconds
    #[arg(long, default_value = "1.0")]
    left_context: f32,

    /// Chunk size in seconds
    #[arg(long, default_value = "0.25")]
    chunk_size: f32,

    /// Right context duration in seconds
    #[arg(long, default_value = "0.25")]
    right_context: f32,

    /// List available audio devices
    #[arg(long)]
    list_devices: bool,

    /// Device index to use (use --list-devices to see available devices)
    #[arg(short, long)]
    device: Option<usize>,

    /// Output file to save transcription results
    #[arg(short, long)]
    output: Option<String>,

    /// Output format: json, csv, or txt
    #[arg(long, default_value = "txt")]
    format: String,

    /// Append to output file instead of overwriting
    #[arg(long)]
    append: bool,

    /// Save received audio to WAV file
    #[arg(long)]
    save_audio: Option<String>,
}

/// Recording thread that captures audio from microphone
///
/// When this thread exits (either via shutdown signal or error), the audio_producer
/// is automatically dropped. The processing thread detects this abandonment via
/// `audio_consumer.is_abandoned()` and gracefully processes remaining buffered data
/// before shutting down. No explicit coordination is needed.
fn recording_thread(
    device: Device,
    config: StreamConfig,
    audio_producer: rtrb::Producer<f32>,
    audio_writer: AudioWriter,
    shutdown: Arc<AtomicBool>,
) -> Result<()> {
    let default_config = device.default_input_config()?;
    let sample_format = default_config.sample_format();
    let channels = config.channels as usize;

    // Build the audio input stream (moves audio_producer into the closure)
    let stream = match sample_format {
        SampleFormat::F32 => {
            build_audio_stream::<f32>(&device, &config, audio_producer, audio_writer, channels)?
        }
        SampleFormat::I16 => {
            build_audio_stream::<i16>(&device, &config, audio_producer, audio_writer, channels)?
        }
        SampleFormat::U16 => {
            build_audio_stream::<u16>(&device, &config, audio_producer, audio_writer, channels)?
        }
        _ => return Err(anyhow!("Unsupported sample format: {:?}", sample_format)),
    };

    stream.play()?;

    // Keep recording until shutdown signal
    while !shutdown.load(Ordering::Relaxed) {
        thread::sleep(Duration::from_millis(100));
    }

    // Drop stream to stop recording and release the audio_producer
    drop(stream);

    Ok(())
}

/// Build audio input stream for a specific sample type
fn build_audio_stream<T>(
    device: &Device,
    config: &StreamConfig,
    mut audio_producer: rtrb::Producer<f32>,
    audio_writer: AudioWriter,
    channels: usize,
) -> Result<Stream>
where
    T: cpal::Sample + cpal::SizedSample + Send + 'static,
    f32: cpal::FromSample<T>,
{
    let stream = device.build_input_stream(
        config,
        move |data: &[T], _: &cpal::InputCallbackInfo| {
            // Convert to mono by averaging channels
            for chunk in data.chunks(channels) {
                let mono_sample = if channels == 1 {
                    f32::from_sample(chunk[0])
                } else {
                    let sum: f32 = chunk.iter().map(|&s| f32::from_sample(s)).sum();
                    sum / channels as f32
                };

                // Send sample to streaming engine (ignore if buffer is full)
                let _ = audio_producer.push(mono_sample);

                // Write sample to audio file (ignore errors to avoid blocking)
                let _ = audio_writer.write_sample(mono_sample);
            }
        },
        |err| eprintln!("Audio stream error: {}", err),
        None,
    )?;

    Ok(stream)
}

/// Processing thread that uses FrameProcessor trait
///
/// This thread leverages automatic abandonment detection via `process_loop()`:
/// - When the recording thread exits (audio producer dropped), the processor
///   automatically detects abandonment via `audio_consumer.is_abandoned()`
/// - When the output thread exits (token consumer dropped), the processor
///   automatically detects abandonment via `token_producer.is_abandoned()`
/// - `process_loop()` handles all frame processing and exits automatically when
///   abandonment is detected or the stream is finished
/// - No explicit `mark_finished()` calls or shutdown coordination needed
async fn processing_thread(mut processor: StreamingParakeetTDT) -> Result<()> {
    // process_loop() runs until abandonment is detected or stream finishes
    // It handles all buffered data and calls finalize() automatically
    processor.process_loop().await.map_err(|e| anyhow!("{}", e))?;

    Ok(())
}

/// Output thread that consumes tokens from ring buffer
///
/// When this thread exits (either via shutdown signal or error), the token_consumer
/// is automatically dropped. The processing thread detects this abandonment via
/// `token_producer.is_abandoned()` and stops producing tokens, then shuts down
/// gracefully. This creates bidirectional shutdown signaling through the pipeline.
fn output_thread(
    mut token_consumer: rtrb::Consumer<TokenResult>,
    output_writer: OutputWriter,
    shutdown: Arc<AtomicBool>,
) -> Result<OutputWriter> {
    let mut text_buffer = String::new();

    // Continue until shutdown AND no more tokens available
    while !shutdown.load(Ordering::Relaxed) || token_consumer.slots() > 0 {
        // Read tokens from the consumer
        let mut new_tokens = Vec::new();
        while let Ok(token_result) = token_consumer.pop() {
            if let Some(ref token_text) = token_result.text {
                // Convert token to text with proper spacing
                if let Some(text_part) = token_to_text(token_text, text_buffer.is_empty()) {
                    // Add to buffer for sentence detection
                    text_buffer.push_str(&text_part);

                    // Stream output immediately
                    print!("{}", text_part);
                    use std::io::{self, Write};
                    io::stdout().flush().unwrap();

                    // Check for sentence endings and add newlines
                    if text_part.contains('.') || text_part.contains('?') || text_part.contains('!')
                    {
                        println!();
                        io::stdout().flush().unwrap();
                        text_buffer.clear();
                    }
                }
            }
            new_tokens.push(token_result);
        }

        // Write tokens to output file
        if !new_tokens.is_empty() {
            if let Err(e) = output_writer.write_tokens(&new_tokens) {
                eprintln!("Error writing to output file: {}", e);
            }
        }

        // Short sleep to avoid busy waiting
        thread::sleep(Duration::from_millis(50));
    }

    // Print any remaining text in buffer when stopping
    if !text_buffer.trim().is_empty() {
        println!("{}", text_buffer.trim());
    }

    Ok(output_writer)
}

fn list_audio_devices() -> Result<()> {
    let host = cpal::default_host();

    println!("Available audio input devices:");

    let devices: Vec<_> = host.input_devices()?.collect();

    if devices.is_empty() {
        println!("No input devices found");
        return Ok(());
    }

    for (index, device) in devices.iter().enumerate() {
        let name = device.name().unwrap_or_else(|_| "Unknown".to_string());
        println!("  {}: {}", index, name);

        if let Ok(config) = device.default_input_config() {
            println!(
                "    Default config: {} channels, {} Hz, {:?}",
                config.channels(),
                config.sample_rate().0,
                config.sample_format()
            );
        }
    }

    Ok(())
}

fn get_audio_device(host: &Host, device_index: Option<usize>) -> Result<Device> {
    match device_index {
        Some(index) => {
            let devices: Vec<_> = host.input_devices()?.collect();
            devices
                .into_iter()
                .nth(index)
                .ok_or_else(|| anyhow!("Device index {} not found", index))
        }
        None => host
            .default_input_device()
            .ok_or_else(|| anyhow!("No default input device available")),
    }
}

fn setup_audio_config(device: &Device, target_sample_rate: u32) -> Result<StreamConfig> {
    let default_config = device.default_input_config()?;

    // Try to use the target sample rate, fall back to default if not supported
    let sample_rate = SampleRate(target_sample_rate);

    let config = StreamConfig {
        channels: 1, // Force mono
        sample_rate,
        buffer_size: cpal::BufferSize::Default,
    };

    println!(
        "Audio config: {} channels, {} Hz, {:?}",
        config.channels,
        config.sample_rate.0,
        default_config.sample_format()
    );

    Ok(config)
}

#[tokio::main]
async fn main() -> Result<()> {
    // This application uses a three-thread architecture with automatic abandonment detection:
    //
    // Recording Thread → [audio_producer] → Processing Thread → [token_producer] → Output Thread
    //
    // Shutdown behavior:
    // - Ctrl+C sets shutdown flag, causing all threads to exit their main loops
    // - When recording thread exits, audio_producer is dropped
    // - Processing thread detects abandonment and processes remaining buffered audio
    // - When processing completes, token_producer is dropped
    // - Output thread detects abandonment and drains remaining tokens
    // - All threads coordinate shutdown automatically via ring buffer abandonment detection
    //
    // This eliminates the need for explicit mark_finished() calls and simplifies coordination.

    let args = Args::parse();

    if args.list_devices {
        return list_audio_devices();
    }

    // Check if models exist
    if !Path::new(&args.models).exists() {
        return Err(anyhow!("Models directory not found: {}", args.models));
    }

    println!("=== Parakeet Real-time Microphone Transcription ===");

    // Parse output format
    let output_format:OutputFormat = args.format.parse()?;

    // Setup output writer
    let output_writer = OutputWriter::new(args.output.clone(), output_format, args.append)?;

    if let Some(ref output_path) = args.output {
        let action = if args.append { "Appending" } else { "Writing" };
        println!(
            "{} output to: {} (format: {})",
            action, output_path, args.format
        );
    }

    // Setup audio writer
    let audio_writer = AudioWriter::new(args.save_audio.clone(), args.sample_rate)?;

    if let Some(ref audio_path) = args.save_audio {
        println!("Saving audio to: {}", audio_path);
    }

    // Load the TDT model
    println!("Loading TDT model from {}...", args.models);
    let exec_config = ExecutionConfig::default();
    let model = ParakeetTDTModel::from_pretrained(&args.models, exec_config)?;
    println!("✓ Model loaded successfully");

    // Load vocabulary
    let vocab_path = Path::new(&args.models).join("vocab.txt");
    let vocab = if vocab_path.exists() {
        match Vocabulary::from_file(&vocab_path) {
            Ok(v) => {
                println!("✓ Vocabulary loaded ({} tokens)", v.size());
                Some(v)
            }
            Err(e) => {
                println!("⚠ Failed to load vocabulary: {}", e);
                println!("  Tokens will be displayed as IDs");
                None
            }
        }
    } else {
        println!("⚠ Vocabulary file not found at {}", vocab_path.display());
        println!("  Tokens will be displayed as IDs");
        None
    };

    // Setup streaming configuration
    let context = ContextConfig::new(
        args.left_context,
        args.chunk_size,
        args.right_context,
        args.sample_rate as usize,
    );

    println!("Streaming configuration:");
    println!("  • Left context: {:.1}s", args.left_context);
    println!("  • Chunk size: {:.0}ms", args.chunk_size * 1000.0);
    println!("  • Right context: {:.0}ms", args.right_context * 1000.0);
    println!(
        "  • Total latency: {:.0}ms",
        context.latency_secs(args.sample_rate as usize) * 1000.0
    );

    // Create streaming engine
    let (streaming_engine, audio_producer, token_consumer) =
        StreamingParakeetTDT::new_with_vocab(model, context, vocab);

    // Setup audio capture
    let host = cpal::default_host();
    let device = get_audio_device(&host, args.device)?;
    let device_name = device.name().unwrap_or_else(|_| "Unknown".to_string());
    println!("Using audio device: {}", device_name);

    let config = setup_audio_config(&device, args.sample_rate)?;

    // Setup shutdown signal
    let shutdown = Arc::new(AtomicBool::new(false));
    let shutdown_clone = shutdown.clone();

    ctrlc::set_handler(move || {
        println!("\nShutdown signal received...");
        shutdown_clone.store(true, Ordering::Relaxed);
    })?;

    println!("✓ Audio capture ready");
    println!("\n🎤 Listening for speech... (Press Ctrl+C to stop)");
    println!("Speak into your microphone to see real-time transcription");

    // Spawn recording thread
    let recording_handle = {
        let shutdown = shutdown.clone();
        let audio_writer_clone = audio_writer.clone();
        thread::spawn(move || {
            recording_thread(device, config, audio_producer, audio_writer_clone, shutdown)
        })
    };

    // Spawn processing thread
    let processing_handle = tokio::spawn(async move {
        processing_thread(streaming_engine).await
    });

    // Spawn output thread
    let output_handle = {
        let shutdown = shutdown.clone();
        thread::spawn(move || {
            output_thread(token_consumer, output_writer, shutdown)
        })
    };

    // Wait for all threads to complete and collect errors
    let recording_result = recording_handle.join().unwrap();
    let processing_result = processing_handle.await.unwrap();
    let output_result = output_handle.join().unwrap();

    // Report any errors
    if let Err(e) = recording_result {
        eprintln!("Recording thread error: {}", e);
    }
    if let Err(e) = processing_result {
        eprintln!("Processing thread error: {}", e);
    }

    // Finalize output writer
    let output_writer = match output_result {
        Ok(writer) => writer,
        Err(e) => {
            eprintln!("Output thread error: {}", e);
            return Err(e);
        }
    };

    // Finalize the files
    if let Err(e) = output_writer.finalize() {
        eprintln!("Error finalizing output file: {}", e);
    }

    if let Err(e) = audio_writer.finalize() {
        eprintln!("Error finalizing audio file: {}", e);
    } else if args.save_audio.is_some() {
        println!("✓ Audio file saved successfully");
    }

    println!("Transcription stopped");
    Ok(())
}
