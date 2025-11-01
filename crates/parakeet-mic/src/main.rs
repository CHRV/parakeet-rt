use anyhow::{anyhow, Result};
use clap::Parser;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, Host, Sample, SampleFormat, SampleRate, Stream, StreamConfig};
use parakeet::execution::ModelConfig as ExecutionConfig;
use parakeet::parakeet_tdt::ParakeetTDTModel;
use parakeet::streaming::{ContextConfig, StreamingParakeetTDT, TokenResult};
use rtrb;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

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

struct AudioCapture {
    _stream: Stream,
}



impl AudioCapture {
    fn new(
        device: &Device,
        config: &StreamConfig,
        audio_producer: rtrb::Producer<f32>,
        audio_writer: AudioWriter,
    ) -> Result<Self> {
        let default_config = device.default_input_config()?;
        let sample_format = default_config.sample_format();
        let channels = config.channels as usize;

        let stream = match sample_format {
            SampleFormat::F32 => Self::build_stream::<f32>(device, config, audio_producer, audio_writer, channels)?,
            SampleFormat::I16 => Self::build_stream::<i16>(device, config, audio_producer, audio_writer, channels)?,
            SampleFormat::U16 => Self::build_stream::<u16>(device, config, audio_producer, audio_writer, channels)?,
            _ => return Err(anyhow!("Unsupported sample format: {:?}", sample_format)),
        };

        stream.play()?;

        Ok(AudioCapture {
            _stream: stream,
        })
    }

    fn build_stream<T>(
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
            println!("    Default config: {} channels, {} Hz, {:?}",
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
            devices.into_iter()
                .nth(index)
                .ok_or_else(|| anyhow!("Device index {} not found", index))
        }
        None => host.default_input_device()
            .ok_or_else(|| anyhow!("No default input device available"))
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

    println!("Audio config: {} channels, {} Hz, {:?}",
        config.channels,
        config.sample_rate.0,
        default_config.sample_format()
    );

    Ok(config)
}

fn transcription_worker(
    mut streaming_engine: StreamingParakeetTDT,
    mut token_consumer: rtrb::Consumer<TokenResult>,
    output_writer: OutputWriter,
    running: Arc<AtomicBool>,
) -> (OutputWriter, usize) {
    println!("Transcription worker started");

    let mut last_print_time = std::time::Instant::now();
    let mut total_tokens = 0;

    while running.load(Ordering::Relaxed) {
        // Process audio through the streaming engine
        if let Err(e) = streaming_engine.process_audio() {
            eprintln!("Error processing audio: {}", e);
            continue;
        }

        // Read any new tokens from the consumer
        let mut new_tokens = Vec::new();
        while let Ok(token_result) = token_consumer.pop() {
            new_tokens.push(token_result);
        }

        // Process tokens if any were detected
        if !new_tokens.is_empty() {
            total_tokens += new_tokens.len();

            // Print tokens to console
            println!("Detected {} tokens:", new_tokens.len());
            for token_result in &new_tokens {
                let token_time = token_result.timestamp as f32 / 16000.0;
                println!("  Token {} at {:.3}s (conf: {:.3})",
                    token_result.token_id, token_time, token_result.confidence);
            }

            // Write tokens to output file
            if let Err(e) = output_writer.write_tokens(&new_tokens) {
                eprintln!("Error writing to output file: {}", e);
            }
        }

        // Print status every 5 seconds
        if last_print_time.elapsed() > Duration::from_secs(5) {
            println!("Processing audio... (Total tokens: {})", total_tokens);
            last_print_time = std::time::Instant::now();
        }

        // Small sleep to prevent busy waiting
        thread::sleep(Duration::from_millis(10));
    }

    println!("Transcription worker stopped (Total tokens processed: {})", total_tokens);
    (output_writer, total_tokens)
}

#[tokio::main]
async fn main() -> Result<()> {
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
    let output_format = OutputFormat::from_str(&args.format)?;

    // Setup output writer
    let output_writer = OutputWriter::new(args.output.clone(), output_format, args.append)?;

    if let Some(ref output_path) = args.output {
        let action = if args.append { "Appending" } else { "Writing" };
        println!("{} output to: {} (format: {})", action, output_path, args.format);
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
    println!("  • Total latency: {:.0}ms", context.latency_secs(args.sample_rate as usize) * 1000.0);

    // Create streaming engine
    let (streaming_engine, audio_producer, token_consumer) =
        StreamingParakeetTDT::new(model, context, args.sample_rate as usize);

    // Setup audio capture
    let host = cpal::default_host();
    let device = get_audio_device(&host, args.device)?;
    let device_name = device.name().unwrap_or_else(|_| "Unknown".to_string());
    println!("Using audio device: {}", device_name);

    let config = setup_audio_config(&device, args.sample_rate)?;

    // Start audio capture - directly connected to streaming engine
    let audio_capture = AudioCapture::new(&device, &config, audio_producer, audio_writer.clone())?;
    println!("✓ Audio capture started");

    // Setup shutdown signal
    let running = Arc::new(AtomicBool::new(true));
    let running_clone = running.clone();

    ctrlc::set_handler(move || {
        println!("\nShutdown signal received...");
        running_clone.store(false, Ordering::Relaxed);
    })?;

    println!("\n🎤 Listening for speech... (Press Ctrl+C to stop)");
    println!("Speak into your microphone to see real-time transcription");

    // Start transcription worker thread
    let transcription_handle = thread::spawn(move || {
        transcription_worker(
            streaming_engine,
            token_consumer,
            output_writer,
            running,
        )
    });

    // Wait for shutdown
    let (output_writer, _total_tokens) = transcription_handle.join().unwrap();

    // Drop audio capture to release the AudioWriter reference
    drop(audio_capture);

    // Now finalize the files after all references are dropped
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