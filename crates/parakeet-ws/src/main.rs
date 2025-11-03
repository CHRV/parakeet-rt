use anyhow::{anyhow, Result};
use clap::Parser;
use frame_processor::FrameProcessor;
use futures_util::{SinkExt, StreamExt};
use parakeet::execution::ModelConfig as ExecutionConfig;
use parakeet::model::ParakeetTDTModel;
use parakeet::streaming::{ContextConfig, StreamingParakeetTDT, TokenResult};
use parakeet::vocab::Vocabulary;
use std::net::SocketAddr;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::net::{TcpListener, TcpStream};
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::WebSocketStream;

mod output;
use output::{AudioWriter, OutputFormat, OutputWriter};

/// Convert a single token to text with proper spacing
fn token_to_text(token_text: &str, is_first_token: bool) -> Option<String> {
    if token_text.starts_with('<') && token_text.ends_with('>') {
        return None;
    }

    if let Some(text_part) = token_text.strip_prefix("▁") {
        if is_first_token {
            Some(text_part.to_string())
        } else {
            Some(format!(" {}", text_part))
        }
    } else {
        Some(token_text.to_string())
    }
}

#[derive(Parser)]
#[command(name = "parakeet-ws")]
#[command(about = "Real-time speech transcription using WebSocket audio input")]
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

    /// WebSocket server address
    #[arg(short = 'a', long, default_value = "127.0.0.1:8080")]
    address: String,

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

/// Processing thread that uses FrameProcessor trait
async fn processing_thread(mut processor: StreamingParakeetTDT) -> Result<()> {
    processor.process_loop().await.map_err(|e| anyhow!("{}", e))?;
    Ok(())
}

/// Output thread that consumes tokens and sends them back via WebSocket
async fn output_thread(
    mut token_consumer: rtrb::Consumer<TokenResult>,
    output_writer: OutputWriter,
    ws_sender: Arc<tokio::sync::Mutex<futures_util::stream::SplitSink<WebSocketStream<TcpStream>, Message>>>,
    shutdown: Arc<AtomicBool>,
) -> Result<OutputWriter> {
    let mut text_buffer = String::new();
    println!("Output thread started");

    while !shutdown.load(Ordering::Relaxed) || token_consumer.slots() > 0 {
        let mut new_tokens = Vec::new();

        while let Ok(token_result) = token_consumer.pop() {
            println!("Received token: id={}, text={:?}", token_result.token_id, token_result.text);

            if let Some(ref token_text) = token_result.text {
                if let Some(text_part) = token_to_text(token_text, text_buffer.is_empty()) {
                    text_buffer.push_str(&text_part);

                    // Print to console
                    print!("{}", text_part);
                    use std::io::{self, Write};
                    io::stdout().flush().unwrap();

                    // Send to browser via WebSocket
                    let mut sender = ws_sender.lock().await;
                    if let Err(e) = sender.send(Message::Text(text_part.clone())).await {
                        eprintln!("Failed to send text to WebSocket: {}", e);
                    } else {
                        println!(" [sent to browser]");
                    }

                    if text_part.contains('.') || text_part.contains('?') || text_part.contains('!') {
                        println!();
                        io::stdout().flush().unwrap();
                        text_buffer.clear();
                    }
                }
            }
            new_tokens.push(token_result);
        }

        if !new_tokens.is_empty() {
            if let Err(e) = output_writer.write_tokens(&new_tokens) {
                eprintln!("Error writing to output file: {}", e);
            }
        }

        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
    }

    if !text_buffer.trim().is_empty() {
        println!("{}", text_buffer.trim());
    }

    Ok(output_writer)
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    if !Path::new(&args.models).exists() {
        return Err(anyhow!("Models directory not found: {}", args.models));
    }

    println!("=== Parakeet WebSocket Transcription Server ===");

    let output_format: OutputFormat = args.format.parse()?;
    let output_writer = OutputWriter::new(args.output.clone(), output_format, args.append)?;

    if let Some(ref output_path) = args.output {
        let action = if args.append { "Appending" } else { "Writing" };
        println!("{} output to: {} (format: {})", action, output_path, args.format);
    }

    let audio_writer = AudioWriter::new(args.save_audio.clone(), args.sample_rate)?;

    if let Some(ref audio_path) = args.save_audio {
        println!("Saving audio to: {}", audio_path);
    }

    println!("Loading TDT model from {}...", args.models);
    let exec_config = ExecutionConfig::default();
    let model = ParakeetTDTModel::from_pretrained(&args.models, exec_config)?;
    println!("✓ Model loaded successfully");

    let vocab_path = Path::new(&args.models).join("vocab.txt");
    let vocab = if vocab_path.exists() {
        match Vocabulary::from_file(&vocab_path) {
            Ok(v) => {
                println!("✓ Vocabulary loaded ({} tokens)", v.size());
                Some(v)
            }
            Err(e) => {
                println!("⚠ Failed to load vocabulary: {}", e);
                None
            }
        }
    } else {
        println!("⚠ Vocabulary file not found");
        None
    };

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

    let (streaming_engine, mut audio_producer, token_consumer) =
        StreamingParakeetTDT::new_with_vocab(model, context, vocab);

    let shutdown = Arc::new(AtomicBool::new(false));
    let shutdown_clone = shutdown.clone();

    tokio::spawn(async move {
        tokio::signal::ctrl_c().await.ok();
        println!("\nShutdown signal received...");
        shutdown_clone.store(true, Ordering::Relaxed);
    });

    let addr: SocketAddr = args.address.parse()?;
    let listener = TcpListener::bind(&addr).await?;
    println!("✓ WebSocket server listening on ws://{}", addr);
    println!("\n🌐 Waiting for client connection...");
    println!("Open the HTML page in your browser to start transcription");

    // Accept one connection
    let (stream, _) = listener.accept().await?;
    let ws_stream = tokio_tungstenite::accept_async(stream).await?;

    let (ws_sender, mut ws_receiver) = ws_stream.split();
    let ws_sender = Arc::new(tokio::sync::Mutex::new(ws_sender));

    println!("Client connected");

    // Spawn processing thread
    let processing_handle = tokio::spawn(async move {
        processing_thread(streaming_engine).await
    });

    // Spawn output thread
    let output_handle = {
        let shutdown = shutdown.clone();
        let ws_sender = ws_sender.clone();
        tokio::spawn(async move {
            output_thread(token_consumer, output_writer, ws_sender, shutdown).await
        })
    };

    // Handle WebSocket messages in main task
    let ws_handle = {
        let shutdown = shutdown.clone();
        let mut audio_count = 0usize;
        tokio::spawn(async move {
            while !shutdown.load(Ordering::Relaxed) {
                tokio::select! {
                    msg = ws_receiver.next() => {
                        match msg {
                            Some(Ok(Message::Binary(data))) => {
                                if data.len() % 4 != 0 {
                                    eprintln!("Invalid audio data length: {}", data.len());
                                    continue;
                                }

                                let sample_count = data.len() / 4;
                                audio_count += sample_count;

                                // Log every second of audio (16000 samples)
                                if audio_count % 16000 < sample_count {
                                    println!("Received {} audio samples ({:.1}s total)", audio_count, audio_count as f32 / 16000.0);
                                }

                                for chunk in data.chunks_exact(4) {
                                    let sample = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);

                                    if audio_producer.push(sample).is_err() {
                                        // Buffer full, skip sample
                                    }

                                    let _ = audio_writer.write_sample(sample);
                                }
                            }
                            Some(Ok(Message::Text(text))) => {
                                println!("Received text message: {}", text);
                            }
                            Some(Ok(Message::Close(_))) => {
                                println!("Client disconnected");
                                shutdown.store(true, Ordering::Relaxed);
                                break;
                            }
                            Some(Err(e)) => {
                                eprintln!("WebSocket error: {}", e);
                                shutdown.store(true, Ordering::Relaxed);
                                break;
                            }
                            None => {
                                println!("WebSocket stream ended");
                                shutdown.store(true, Ordering::Relaxed);
                                break;
                            }
                            _ => {}
                        }
                    }
                    _ = tokio::time::sleep(tokio::time::Duration::from_millis(10)) => {}
                }
            }
        })
    };

    // Wait for all tasks
    let _ = ws_handle.await;
    let _ = processing_handle.await;
    let output_result = output_handle.await.unwrap();

    let output_writer = match output_result {
        Ok(writer) => writer,
        Err(e) => {
            eprintln!("Output thread error: {}", e);
            return Err(e);
        }
    };

    if let Err(e) = output_writer.finalize() {
        eprintln!("Error finalizing output file: {}", e);
    }

    println!("Transcription stopped");
    Ok(())
}
