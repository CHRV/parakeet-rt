use anyhow::Result;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, Sample, SampleFormat, Stream, StreamConfig};
use dasp_interpolate::sinc::Sinc;
use dasp_ring_buffer::Fixed;
use dasp_signal::Signal;
use frame_processor::FrameProcessor;
use parakeet::execution::ModelConfig;
use parakeet::model::ParakeetTDTModel;
use parakeet::streaming::StreamingParakeetTDT;
use parakeet::vocab::Vocabulary;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use tauri::{Emitter, Manager};
use tokio::task::JoinHandle;

// ============================================================================
// Configuration
// ============================================================================

/// Application configuration with model paths and ORT settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub encoder_model_path: String,
    pub decoder_model_path: String,
    pub vocab_path: String,
    pub ort_num_threads: u32,
    pub left_context_size: f32,
    pub chunk_size: f32,
    pub right_context_size: f32,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            encoder_model_path: "models".to_string(),
            decoder_model_path: "models".to_string(),
            vocab_path: "models/vocab.txt".to_string(),
            ort_num_threads: 4,
            left_context_size: 1.0,
            chunk_size: 0.25,
            right_context_size: 0.25,
        }
    }
}

// ============================================================================
// Application State
// ============================================================================

/// Global application state managing recording status, tasks, and models
pub struct AppState {
    /// Whether recording is currently active
    recording: Arc<Mutex<bool>>,
    /// Handle to the audio recording task
    recording_task: Arc<Mutex<Option<JoinHandle<Result<()>>>>>,
    /// Handle to the audio processing task
    processing_task: Arc<Mutex<Option<JoinHandle<Result<()>>>>>,
    /// Handle to the output/token processing task
    output_task: Arc<Mutex<Option<JoinHandle<Result<()>>>>>,
    /// Shutdown signal for all tasks
    shutdown: Arc<AtomicBool>,
    /// Current audio level (RMS) for visualization
    audio_level: Arc<Mutex<f32>>,
    /// Loaded Parakeet TDT model
    model: Arc<Mutex<Option<ParakeetTDTModel>>>,
    /// Loaded vocabulary for token decoding
    vocab: Arc<Mutex<Option<Vocabulary>>>,
    /// Current application configuration
    config: Arc<Mutex<AppConfig>>,
}

impl AppState {
    /// Create a new AppState with default values
    pub fn new() -> Self {
        Self {
            recording: Arc::new(Mutex::new(false)),
            recording_task: Arc::new(Mutex::new(None)),
            processing_task: Arc::new(Mutex::new(None)),
            output_task: Arc::new(Mutex::new(None)),
            shutdown: Arc::new(AtomicBool::new(false)),
            audio_level: Arc::new(Mutex::new(0.0)),
            model: Arc::new(Mutex::new(None)),
            vocab: Arc::new(Mutex::new(None)),
            config: Arc::new(Mutex::new(AppConfig::default())),
        }
    }
}

// ============================================================================
// Configuration Management
// ============================================================================

/// Load configuration from app data directory
/// Returns default config if file doesn't exist
fn load_config(app_handle: &tauri::AppHandle) -> Result<AppConfig> {
    let app_data_dir = app_handle
        .path()
        .app_data_dir()
        .map_err(|e| anyhow::anyhow!("Failed to get app data directory: {}", e))?;

    let config_path = app_data_dir.join("config.json");

    if !config_path.exists() {
        // Return default config if file doesn't exist
        return Ok(AppConfig::default());
    }

    let config_str = std::fs::read_to_string(&config_path)
        .map_err(|e| anyhow::anyhow!("Failed to read config file: {}", e))?;

    let config: AppConfig = serde_json::from_str(&config_str)
        .map_err(|e| anyhow::anyhow!("Failed to parse config file: {}", e))?;

    Ok(config)
}

/// Save configuration to app data directory
fn save_config(app_handle: &tauri::AppHandle, config: &AppConfig) -> Result<()> {
    let app_data_dir = app_handle
        .path()
        .app_data_dir()
        .map_err(|e| anyhow::anyhow!("Failed to get app data directory: {}", e))?;

    // Create directory if it doesn't exist
    std::fs::create_dir_all(&app_data_dir)
        .map_err(|e| anyhow::anyhow!("Failed to create app data directory: {}", e))?;

    let config_path = app_data_dir.join("config.json");

    let config_str = serde_json::to_string_pretty(config)
        .map_err(|e| anyhow::anyhow!("Failed to serialize config: {}", e))?;

    std::fs::write(&config_path, config_str)
        .map_err(|e| anyhow::anyhow!("Failed to write config file: {}", e))?;

    Ok(())
}

// ============================================================================
// Model Loading
// ============================================================================

/// Load Parakeet TDT model and vocabulary from config paths
fn load_models_internal(config: &AppConfig) -> Result<(ParakeetTDTModel, Vocabulary)> {
    // Create execution config with thread settings
    let exec_config = ModelConfig::default()
        .with_intra_threads(config.ort_num_threads as usize)
        .with_inter_threads(1);

    // Load model from directory (encoder and decoder paths should point to same directory)
    let model_dir = PathBuf::from(&config.encoder_model_path);
    let model = ParakeetTDTModel::from_pretrained(&model_dir, exec_config)
        .map_err(|e| anyhow::anyhow!("Failed to load Parakeet model: {}", e))?;

    // Load vocabulary
    let vocab = Vocabulary::from_file(&config.vocab_path)
        .map_err(|e| anyhow::anyhow!("Failed to load vocabulary: {}", e))?;

    Ok((model, vocab))
}

// ============================================================================
// Audio Configuration
// ============================================================================

const SAMPLE_RATE: u32 = 16000;
const CHANNELS: u16 = 1;

// ============================================================================
// Streaming Configuration (defaults - overridden by AppConfig)
// ============================================================================

const DEFAULT_LEFT_CONTEXT: f32 = 3.0; // seconds
const DEFAULT_CHUNK_SIZE: f32 = 0.5; // seconds
const DEFAULT_RIGHT_CONTEXT: f32 = 0.5; // seconds

// ============================================================================
// Audio Capture
// ============================================================================

/// Initialize audio capture device and configuration
/// This function runs in a blocking context via tokio::task::spawn_blocking
async fn initialize_audio_device() -> Result<(Device, StreamConfig)> {
    tokio::task::spawn_blocking(|| {
        // Get default audio host
        let host = cpal::default_host();

        // Get default input device
        let device = host
            .default_input_device()
            .ok_or_else(|| anyhow::anyhow!("No default input device available"))?;

        let device_name = device.name().unwrap_or_else(|_| "Unknown".to_string());

        // Get device's default input configuration
        let default_config = device
            .default_input_config()
            .map_err(|e| anyhow::anyhow!("Failed to get default input config: {}", e))?;

        let native_sample_rate = default_config.sample_rate().0;

        println!("Using audio device: {}", device_name);
        println!(
            "Device native sample rate: {} Hz (target: {} Hz)",
            native_sample_rate, SAMPLE_RATE
        );

        // Validate sample rate is within reasonable bounds
        if native_sample_rate < 8000 || native_sample_rate > 192000 {
            return Err(anyhow::anyhow!(
                "Unsupported sample rate: {} Hz. Supported range: 8000-192000 Hz",
                native_sample_rate
            ));
        }

        // Configure stream using device's native sample rate
        let config = StreamConfig {
            channels: CHANNELS,
            sample_rate: cpal::SampleRate(native_sample_rate),
            buffer_size: cpal::BufferSize::Default,
        };

        if native_sample_rate != SAMPLE_RATE {
            println!(
                "Resampling will be applied: {} Hz → {} Hz",
                native_sample_rate, SAMPLE_RATE
            );
        } else {
            println!("No resampling needed (native rate matches target)");
        }

        Ok((device, config))
    })
    .await
    .map_err(|e| anyhow::anyhow!("Failed to spawn blocking task: {}", e))?
}

/// Build audio input stream for a specific sample type
/// Handles generic sample formats (F32, I16, U16) and converts to mono f32
fn build_audio_stream<T>(
    device: &Device,
    config: &StreamConfig,
    mut audio_producer: rtrb::Producer<f32>,
    channels: usize,
    audio_level: Arc<Mutex<f32>>,
    native_sample_rate: u32,
) -> Result<Stream>
where
    T: Sample + cpal::SizedSample + Send + 'static,
    f32: cpal::FromSample<T>,
{
    // Create resampler state if native sample rate differs from target (16kHz)
    let needs_resampling = native_sample_rate != SAMPLE_RATE;
    let mut sample_buffer: Vec<f32> = Vec::new();

    let stream = device.build_input_stream(
        config,
        move |data: &[T], _: &cpal::InputCallbackInfo| {
            let mut sum_squares = 0.0f32;
            let mut sample_count = 0;

            // Convert to mono by averaging channels and collect into buffer
            for chunk in data.chunks(channels) {
                let mono_sample = if channels == 1 {
                    f32::from_sample(chunk[0])
                } else {
                    let sum: f32 = chunk.iter().map(|&s| f32::from_sample(s)).sum();
                    sum / channels as f32
                };
                sample_buffer.push(mono_sample);
            }

            // Apply resampling if needed
            if needs_resampling {
                // Calculate expected output sample count
                let input_count = sample_buffer.len();
                let output_count = ((input_count as f64 * SAMPLE_RATE as f64)
                    / native_sample_rate as f64)
                    .ceil() as usize;

                // Create resampler from buffered samples
                let source_signal = dasp_signal::from_iter(sample_buffer.drain(..));
                let mut resampler = source_signal.from_hz_to_hz(
                    Sinc::new(Fixed::from([0.0; 64])),
                    native_sample_rate as f64,
                    SAMPLE_RATE as f64,
                );

                // Collect and push resampled output
                for _ in 0..output_count {
                    let resampled_sample = resampler.next();

                    // Calculate RMS for audio level visualization
                    sum_squares += resampled_sample * resampled_sample;
                    sample_count += 1;

                    // Push sample to ring buffer
                    if audio_producer.push(resampled_sample).is_err() {
                        // Buffer full - silently drop samples to avoid blocking
                    }
                }
            } else {
                // No resampling needed - push samples directly
                for sample in sample_buffer.drain(..) {
                    // Calculate RMS for audio level visualization
                    sum_squares += sample * sample;
                    sample_count += 1;

                    // Push sample to ring buffer
                    if audio_producer.push(sample).is_err() {
                        // Buffer full - silently drop samples to avoid blocking
                    }
                }
            }

            // Update audio level (RMS)
            if sample_count > 0 {
                let rms = (sum_squares / sample_count as f32).sqrt();
                if let Ok(mut level) = audio_level.lock() {
                    *level = rms;
                }
            }
        },
        |err| {
            eprintln!("Audio stream error: {}", err);
        },
        None,
    )?;

    Ok(stream)
}

/// Recording task that captures audio from microphone
/// Runs in a tokio task and keeps the cpal stream alive
async fn recording_task(
    device: Device,
    config: StreamConfig,
    audio_producer: rtrb::Producer<f32>,
    audio_level: Arc<Mutex<f32>>,
    shutdown: Arc<AtomicBool>,
) -> Result<()> {
    // Get device info for sample format
    let default_config = device
        .default_input_config()
        .map_err(|e| anyhow::anyhow!("Failed to get default input config: {}", e))?;

    let sample_format = default_config.sample_format();
    let channels = config.channels as usize;
    let native_sample_rate = default_config.sample_rate().0;

    // Validate sample rate is within reasonable bounds
    if native_sample_rate < 8000 || native_sample_rate > 192000 {
        return Err(anyhow::anyhow!(
            "Unsupported sample rate: {} Hz. Supported range: 8000-192000 Hz",
            native_sample_rate
        ));
    }

    println!(
        "Audio stream configuration: {} Hz → {} Hz (resampling: {})",
        native_sample_rate,
        SAMPLE_RATE,
        if native_sample_rate != SAMPLE_RATE {
            "enabled"
        } else {
            "disabled"
        }
    );
    println!("Sample format: {:?}, Channels: {}", sample_format, channels);

    // Build the audio input stream based on sample format
    let stream = match sample_format {
        SampleFormat::F32 => build_audio_stream::<f32>(
            &device,
            &config,
            audio_producer,
            channels,
            audio_level,
            native_sample_rate,
        )?,
        SampleFormat::I16 => build_audio_stream::<i16>(
            &device,
            &config,
            audio_producer,
            channels,
            audio_level,
            native_sample_rate,
        )?,
        SampleFormat::U16 => build_audio_stream::<u16>(
            &device,
            &config,
            audio_producer,
            channels,
            audio_level,
            native_sample_rate,
        )?,
        _ => {
            return Err(anyhow::anyhow!(
                "Unsupported sample format: {:?}",
                sample_format
            ))
        }
    };

    // Start the stream
    stream
        .play()
        .map_err(|e| anyhow::anyhow!("Failed to start audio stream: {}", e))?;

    println!("Audio stream started");

    // Keep stream alive until shutdown signal
    while !shutdown.load(Ordering::Relaxed) {
        tokio::task::yield_now().await;
    }

    println!("Shutdown signal detected, stopping recording");

    // Drop stream to stop recording
    drop(stream);

    println!("Audio stream stopped");

    Ok(())
}

// ============================================================================
// Processing Task
// ============================================================================

/// Processing task that uses StreamingParakeetTDT with FrameProcessor trait
///
/// This task leverages automatic abandonment detection via `process_loop()`:
/// - When the recording task exits (audio producer dropped), the processor
///   automatically detects abandonment via `audio_consumer.is_abandoned()`
/// - When the output task exits (token consumer dropped), the processor
///   automatically detects abandonment via `token_producer.is_abandoned()`
/// - `process_loop()` handles all frame processing and exits automatically when
///   abandonment is detected or the stream is finished
/// - No explicit `mark_finished()` calls or shutdown coordination needed
async fn processing_task(mut processor: StreamingParakeetTDT) -> Result<()> {
    tracing::info!("Processing task started");

    // process_loop() runs until abandonment is detected or stream finishes
    // It handles all buffered data and calls finalize() automatically
    processor.process_loop().await.map_err(|e| {
        tracing::error!(error = %e, "Processing loop error");
        anyhow::anyhow!("{}", e)
    })?;

    tracing::info!("Processing task completed");
    Ok(())
}

// ============================================================================
// Output Task
// ============================================================================

/// Output task that consumes tokens from ring buffer and emits events
///
/// This task:
/// - Polls the token ring buffer in an async loop
/// - Converts tokens to text using token_to_text() function
/// - Tracks text buffer for sentence detection
/// - Continues until shutdown AND ring buffer is empty
async fn output_task(
    mut token_consumer: rtrb::Consumer<parakeet::streaming::TokenResult>,
    app_handle: tauri::AppHandle,
    audio_level: Arc<Mutex<f32>>,
    shutdown: Arc<AtomicBool>,
) -> Result<()> {
    tracing::info!("Output task started");

    let mut text_buffer = String::new();
    let mut last_audio_level_emit = std::time::Instant::now();
    const AUDIO_LEVEL_THROTTLE: std::time::Duration = std::time::Duration::from_millis(33); // ~30 FPS

    // Continue until shutdown AND ring buffer is empty
    while !shutdown.load(Ordering::Relaxed) || token_consumer.slots() > 0 {
        // Poll token ring buffer
        let mut has_tokens = false;
        while let Ok(token_result) = token_consumer.pop() {
            has_tokens = true;

            // Convert token to text using token_to_text() function
            if let Some(ref token_text) = token_result.text {
                if let Some(text_part) = token_to_text(token_text, false) {
                    // Add to buffer for sentence detection
                    text_buffer.push_str(&text_part);

                    // Check for sentence endings and emit with newline
                    if text_part.contains('.') || text_part.contains('?') || text_part.contains('!')
                    {
                        // Emit transcription event with text payload
                        if let Err(e) = app_handle.emit("transcription", text_buffer.clone()) {
                            tracing::warn!(error = %e, "Failed to emit transcription event");
                        }
                        text_buffer.clear();
                    }
                }
            }
        }

        // If we processed tokens but buffer is not empty (no sentence ending yet),
        // emit the partial text
        if has_tokens && !text_buffer.is_empty() {
            if let Err(e) = app_handle.emit("transcription", text_buffer.clone()) {
                tracing::warn!(error = %e, "Failed to emit transcription event");
            }
            text_buffer.clear();
        }

        // Emit audio level events (throttled to ~30 FPS)
        let now = std::time::Instant::now();
        if now.duration_since(last_audio_level_emit) >= AUDIO_LEVEL_THROTTLE {
            let level = *audio_level.lock().unwrap();
            if let Err(e) = app_handle.emit("audio-level", level) {
                tracing::warn!(error = %e, "Failed to emit audio-level event");
            }
            last_audio_level_emit = now;
        }

        // Use tokio::time::sleep for yielding between iterations
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
    }

    tracing::info!("Output task completed");
    Ok(())
}

// ============================================================================
// Token Processing
// ============================================================================

/// Convert a single token to text with proper spacing
/// Filters special tokens and handles word-initial marker (▁) for spacing
fn token_to_text(token_text: &str, is_first_token: bool) -> Option<String> {
    // Filter special tokens (tokens starting with < and ending with >)
    if token_text.starts_with('<') && token_text.ends_with('>') {
        return None;
    }

    // Handle word-initial marker (▁) for spacing
    if let Some(text_part) = token_text.strip_prefix("▁") {
        println!("Text: {:?}", token_text);
        // SentencePiece space prefix - replace with actual space
        if is_first_token {
            // First token: no leading space
            Some(text_part.to_string())
        } else {
            // Subsequent tokens: add space before word
            Some(format!(" {}", text_part))
        }
    } else {
        // Regular token - append directly (no space)
        Some(token_text.to_string())
    }
}

// ============================================================================
// Tauri Commands
// ============================================================================

/// Start recording command
/// Initializes audio capture and processing pipeline
#[tauri::command]
async fn start_recording(
    app_handle: tauri::AppHandle,
    state: tauri::State<'_, AppState>,
) -> Result<(), String> {
    // Check if already recording
    {
        let recording = state.recording.lock().unwrap();
        if *recording {
            return Err("Already recording".to_string());
        }
    }

    // Check if models are loaded
    {
        let model_guard = state.model.lock().unwrap();
        let vocab_guard = state.vocab.lock().unwrap();

        if model_guard.is_none() || vocab_guard.is_none() {
            // Try to load models
            let config = state.config.lock().unwrap().clone();
            drop(model_guard);
            drop(vocab_guard);

            match load_models_internal(&config) {
                Ok((loaded_model, loaded_vocab)) => {
                    *state.model.lock().unwrap() = Some(loaded_model);
                    *state.vocab.lock().unwrap() = Some(loaded_vocab);
                }
                Err(e) => {
                    let error_msg = format!("Failed to load models: {}", e);
                    let _ = app_handle.emit("error", error_msg.clone());
                    return Err(error_msg);
                }
            }
        }
    }

    // Reset shutdown signal
    state.shutdown.store(false, Ordering::Relaxed);

    // Get model and vocab from state
    let model = state
        .model
        .lock()
        .unwrap()
        .take()
        .ok_or_else(|| "Models not loaded".to_string())?;
    let vocab = state.vocab.lock().unwrap().take();

    // Get context configuration from AppConfig
    let config = state.config.lock().unwrap().clone();

    // Create StreamingParakeetTDT with context configuration
    // This creates the ring buffers internally and returns them
    let context_config = parakeet::streaming::ContextConfig::new(
        config.left_context_size,
        config.chunk_size,
        config.right_context_size,
        SAMPLE_RATE as usize,
    );

    let (processor, audio_producer, token_consumer) =
        StreamingParakeetTDT::new_with_vocab(model, context_config, vocab);

    // Initialize audio device
    let (device, config) = match initialize_audio_device().await {
        Ok(result) => result,
        Err(e) => {
            let error_msg = if e.to_string().contains("Unsupported sample rate") {
                format!(
                    "Audio device error: {}. Please try a different microphone or check device settings.",
                    e
                )
            } else {
                format!("Failed to initialize audio device: {}", e)
            };
            let _ = app_handle.emit("error", error_msg.clone());
            return Err(error_msg);
        }
    };

    // Spawn audio capture task (must use spawn_blocking because cpal Stream is not Send)
    let recording_handle = {
        let audio_level = state.audio_level.clone();
        let shutdown = state.shutdown.clone();
        tokio::task::spawn_blocking(move || {
            tokio::runtime::Handle::current().block_on(async move {
                recording_task(device, config, audio_producer, audio_level, shutdown).await
            })
        })
    };

    // Spawn processing task with StreamingParakeetTDT
    let processing_handle = tokio::task::spawn(async move { processing_task(processor).await });

    // Spawn output task for token consumption
    let output_handle = {
        let app_handle_clone = app_handle.clone();
        let audio_level = state.audio_level.clone();
        let shutdown = state.shutdown.clone();
        tokio::task::spawn(async move {
            output_task(token_consumer, app_handle_clone, audio_level, shutdown).await
        })
    };

    // Store task handles in AppState
    *state.recording_task.lock().unwrap() = Some(recording_handle);
    *state.processing_task.lock().unwrap() = Some(processing_handle);
    *state.output_task.lock().unwrap() = Some(output_handle);

    // Update AppState to mark recording as active
    *state.recording.lock().unwrap() = true;

    println!("Recording started");
    Ok(())
}

/// Stop recording command
/// Signals all tasks to stop and waits for cleanup
#[tauri::command]
async fn stop_recording(state: tauri::State<'_, AppState>) -> Result<(), String> {
    // Check if recording
    {
        let recording = state.recording.lock().unwrap();
        if !*recording {
            return Err("Not currently recording".to_string());
        }
    }

    println!("Stopping recording...");

    // Set shutdown flag to signal tasks to stop
    state.shutdown.store(true, Ordering::Relaxed);

    // Take task handles from state
    let recording_handle = state.recording_task.lock().unwrap().take();
    let processing_handle = state.processing_task.lock().unwrap().take();
    let output_handle = state.output_task.lock().unwrap().take();

    // Await all task handles to complete
    if let Some(handle) = recording_handle {
        match tokio::time::timeout(tokio::time::Duration::from_secs(5), handle).await {
            Ok(result) => match result {
                Ok(task_result) => {
                    if let Err(e) = task_result {
                        eprintln!("Recording task error: {}", e);
                    }
                }
                Err(e) => {
                    eprintln!("Recording task panicked: {}", e);
                }
            },
            Err(_) => {
                eprintln!("Recording task timeout");
            }
        }
    }

    if let Some(handle) = processing_handle {
        match tokio::time::timeout(tokio::time::Duration::from_secs(5), handle).await {
            Ok(result) => match result {
                Ok(task_result) => {
                    if let Err(e) = task_result {
                        eprintln!("Processing task error: {}", e);
                    }
                }
                Err(e) => {
                    eprintln!("Processing task panicked: {}", e);
                }
            },
            Err(_) => {
                eprintln!("Processing task timeout");
            }
        }
    }

    if let Some(handle) = output_handle {
        match tokio::time::timeout(tokio::time::Duration::from_secs(5), handle).await {
            Ok(result) => match result {
                Ok(task_result) => {
                    if let Err(e) = task_result {
                        eprintln!("Output task error: {}", e);
                    }
                }
                Err(e) => {
                    eprintln!("Output task panicked: {}", e);
                }
            },
            Err(_) => {
                eprintln!("Output task timeout");
            }
        }
    }

    // Update AppState to mark recording as inactive
    *state.recording.lock().unwrap() = false;

    println!("Recording stopped");
    Ok(())
}

/// Get settings command
/// Returns current configuration
#[tauri::command]
async fn get_settings(state: tauri::State<'_, AppState>) -> Result<AppConfig, String> {
    let config = state.config.lock().unwrap().clone();
    Ok(config)
}

/// Save settings command
/// Validates and persists configuration to disk
#[tauri::command]
async fn save_settings(
    app_handle: tauri::AppHandle,
    state: tauri::State<'_, AppState>,
    config: AppConfig,
) -> Result<(), String> {
    // Check if currently recording
    {
        let recording = state.recording.lock().unwrap();
        if *recording {
            return Err("Cannot change settings while recording".to_string());
        }
    }

    // Validate settings
    // Check thread count in range 1-16
    if config.ort_num_threads < 1 || config.ort_num_threads > 16 {
        return Err("Thread count must be between 1 and 16".to_string());
    }

    // Validate context sizes
    if config.left_context_size < 0.0 || config.left_context_size > 10.0 {
        return Err("Left context size must be between 0 and 10 seconds".to_string());
    }

    if config.chunk_size < 0.01 || config.chunk_size > 5.0 {
        return Err("Chunk size must be between 0.01 and 5 seconds".to_string());
    }

    if config.right_context_size < 0.0 || config.right_context_size > 5.0 {
        return Err("Right context size must be between 0 and 5 seconds".to_string());
    }

    // Check if file paths exist
    let encoder_path = PathBuf::from(&config.encoder_model_path);
    let decoder_path = PathBuf::from(&config.decoder_model_path);
    let vocab_path = PathBuf::from(&config.vocab_path);

    if !encoder_path.exists() {
        return Err(format!(
            "Encoder model path does not exist: {}",
            config.encoder_model_path
        ));
    }

    if !decoder_path.exists() {
        return Err(format!(
            "Decoder model path does not exist: {}",
            config.decoder_model_path
        ));
    }

    if !vocab_path.exists() {
        return Err(format!(
            "Vocabulary path does not exist: {}",
            config.vocab_path
        ));
    }

    // Save config to disk using save_config() function
    if let Err(e) = save_config(&app_handle, &config) {
        let error_msg = format!("Failed to save config: {}", e);
        let _ = app_handle.emit("error", error_msg.clone());
        return Err(error_msg);
    }

    // Update AppState with new config
    *state.config.lock().unwrap() = config;

    println!("Settings saved successfully");
    Ok(())
}

/// Load models command
/// Reloads models with current configuration
#[tauri::command]
async fn load_models(
    app_handle: tauri::AppHandle,
    state: tauri::State<'_, AppState>,
) -> Result<(), String> {
    // Check if currently recording
    {
        let recording = state.recording.lock().unwrap();
        if *recording {
            return Err("Cannot load models while recording".to_string());
        }
    }

    // Load models using current config paths
    let config = state.config.lock().unwrap().clone();

    match load_models_internal(&config) {
        Ok((model, vocab)) => {
            // Update AppState with new model and vocabulary
            *state.model.lock().unwrap() = Some(model);
            *state.vocab.lock().unwrap() = Some(vocab);
            println!("Models loaded successfully");
            Ok(())
        }
        Err(e) => {
            let error_msg = format!("Failed to load models: {}", e);
            let _ = app_handle.emit("error", error_msg.clone());
            Err(error_msg)
        }
    }
}

// ============================================================================
// Application Entry Point
// ============================================================================

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            // Create application state
            let state = AppState::new();

            // Load configuration
            let config = load_config(app.handle()).unwrap_or_else(|e| {
                eprintln!("Failed to load config, using defaults: {}", e);
                AppConfig::default()
            });

            // Update state with loaded config
            *state.config.lock().unwrap() = config.clone();

            // Try to load models on startup
            match load_models_internal(&config) {
                Ok((model, vocab)) => {
                    *state.model.lock().unwrap() = Some(model);
                    *state.vocab.lock().unwrap() = Some(vocab);
                    println!("Models loaded successfully");
                }
                Err(e) => {
                    eprintln!("Failed to load models on startup: {}", e);
                    eprintln!("You can configure model paths in settings");
                }
            }

            // Store state in Tauri's managed state
            app.manage(state);

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            start_recording,
            stop_recording,
            get_settings,
            save_settings,
            load_models
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
