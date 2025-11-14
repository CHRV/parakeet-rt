# Design Document

## Overview

This design document outlines the architecture for a Tauri desktop application that provides real-time speech transcription using the Parakeet TDT model. The application consists of a Rust backend handling audio capture via cpal and transcription processing, and a Svelte frontend providing a simple, clean user interface matching the existing parakeet-ws HTML interface, plus a settings page for configuration.

The design follows a multi-threaded architecture with:
- Main Tauri thread handling commands and events
- Audio capture thread using cpal for microphone input
- Processing thread running the StreamingParakeetTDT engine
- Output thread consuming tokens and emitting events to the frontend
- Configuration management for persisting user settings

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Svelte Frontend                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Transcript  │  │  Visualizer  │  │   Controls   │      │
│  │   Display    │  │   Canvas     │  │   (Buttons)  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│  ┌──────────────────────────────────────────────────┐      │
│  │              Settings Page                        │      │
│  │  (Model Paths, ORT Threads, etc.)                │      │
│  └──────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────┘
                          ↕ Tauri Events & Commands
┌─────────────────────────────────────────────────────────────┐
│                      Rust Backend                            │
│                                                               │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐    │
│  │    cpal      │ → │   Audio      │ → │  Processing  │    │
│  │   Capture    │   │ Ring Buffer  │   │    Thread    │    │
│  └──────────────┘   └──────────────┘   └──────────────┘    │
│                                              ↓               │
│                                         ┌──────────────┐    │
│                                         │    Token     │    │
│                                         │ Ring Buffer  │    │
│                                         └──────────────┘    │
│                                              ↓               │
│                                         ┌──────────────┐    │
│                                         │    Output    │    │
│                                         │    Thread    │    │
│                                         └──────────────┘    │
│                                              ↓               │
│  ┌──────────────┐                 Emit "transcription" event│
│  │   Config     │ ← Load/Save settings                      │
│  │ Persistence  │                                            │
│  └──────────────┘                                            │
└─────────────────────────────────────────────────────────────┘
```

### Component Breakdown

#### Frontend (Svelte)

1. **Main Page Component** (`+page.svelte`)
   - Full-page transcript display area
   - Floating control panel
   - Manages application state (idle, recording, error)
   - Listens for Tauri events

2. **Transcript Display**
   - Renders transcribed text with proper formatting
   - Auto-scrolls to show latest content
   - Handles sentence breaks and spacing

3. **Audio Visualizer**
   - Canvas-based waveform display
   - Updates based on audio level events from backend
   - Shows empty state when not recording

4. **Control Panel**
   - Start/Stop buttons
   - Status indicator
   - Application title and branding
   - Settings button to navigate to settings page

5. **Settings Page Component** (`settings/+page.svelte`)
   - Form inputs for model paths (encoder, decoder, vocabulary)
   - Number input for ORT thread count (1-16)
   - Save and Cancel buttons
   - Input validation and error display
   - Navigation back to main page using SvelteKit navigation
   - Loads current settings on mount using `get_settings` command
   - Saves settings using `save_settings` command
   - Shows success/error messages after save

#### Backend (Rust)

1. **Tauri Commands**
   - `start_recording()`: Initializes audio capture and processing
     - Checks if already recording (return error if true)
     - Verifies models are loaded
     - Spawns recording, processing, and output tasks
     - Updates state to recording=true
   - `stop_recording()`: Stops all threads and cleans up resources
     - Sets shutdown signal
     - Waits for tasks to complete (with timeout)
     - Clears task handles
     - Updates state to recording=false
   - `get_settings()`: Returns current configuration settings
     - Locks config mutex and clones current config
   - `save_settings()`: Validates and persists configuration to disk
     - Validates input (file paths exist, thread count in range)
     - Writes to config file in app data directory
     - Updates in-memory config
     - Returns error if validation fails
   - `load_models()`: Reloads models with new configuration
     - Reads current config
     - Loads encoder and decoder models with ORT settings
     - Loads vocabulary file
     - Updates model and vocab in state
     - Returns error if loading fails

2. **Audio Capture Module** (using cpal)
   - Runs in tokio::task::spawn_blocking (cpal requires blocking context)
   - Initializes default input device using `cpal::default_host()`
   - Configures StreamConfig for 16kHz mono audio
   - Builds input stream with callback using `build_input_stream()`
   - Handles multiple sample formats (F32, I16, U16) with generic function
   - Converts multi-channel to mono by averaging
   - Pushes samples to rtrb::Producer ring buffer
   - Calculates RMS audio levels for visualization
   - Keeps stream alive until shutdown signal using async sleep

3. **Processing Module**
   - Runs in tokio::task (async)
   - Uses StreamingParakeetTDT.process_loop() from FrameProcessor trait
   - Consumes audio from rtrb::Consumer ring buffer
   - Produces tokens to rtrb::Producer ring buffer
   - Automatically detects abandonment when recording thread exits
   - Processes remaining buffered audio before shutting down
   - No explicit mark_finished() calls needed

4. **Output Module**
   - Runs in tokio::task (async)
   - Consumes tokens from rtrb::Consumer ring buffer in async loop
   - Converts tokens to text using token_to_text() function
   - Filters special tokens (starting with < and ending with >)
   - Handles word-initial marker (▁) for proper spacing
   - Emits Tauri "transcription" events with text using app_handle.emit()
   - Emits Tauri "audio-level" events for visualization (throttled to ~30 FPS)
   - Uses tokio::time::sleep for yielding between iterations
   - Continues until shutdown AND ring buffer is empty

5. **State Management**
   - Global state using Arc<Mutex<AppState>>
   - Tracks recording status
   - Manages thread handles for cleanup

6. **Configuration Module**
   - Loads settings from JSON file on startup using serde_json
   - Provides default values via Default trait if no config exists
   - Validates settings:
     - File paths exist and are readable
     - Thread count in range 1-16
   - Persists settings to disk using serde_json::to_string_pretty()
   - Uses `app_handle.path().app_data_dir()` for config file location
   - Config file name: `config.json`
   - Creates app data directory if it doesn't exist

## Components and Interfaces

### Tauri Commands

```rust
#[tauri::command]
async fn start_recording(
    app_handle: tauri::AppHandle,
    state: tauri::State<'_, AppState>
) -> Result<(), String>

#[tauri::command]
async fn stop_recording(
    state: tauri::State<'_, AppState>
) -> Result<(), String>

#[tauri::command]
async fn get_settings(
    state: tauri::State<'_, AppState>
) -> Result<AppConfig, String>

#[tauri::command]
async fn save_settings(
    app_handle: tauri::AppHandle,
    state: tauri::State<'_, AppState>,
    config: AppConfig
) -> Result<(), String>

#[tauri::command]
async fn load_models(
    state: tauri::State<'_, AppState>
) -> Result<(), String>
```

### Tauri Events

Events are emitted using `app_handle.emit()` and listened to in frontend using `listen()` from `@tauri-apps/api/event`.

```rust
// Rust side - emitting events
app_handle.emit("transcription", text)?;
app_handle.emit("audio-level", level)?;
app_handle.emit("error", error_message)?;
```

```typescript
// TypeScript side - event payloads
interface TranscriptionEvent {
  payload: string; // The transcribed text
}

interface AudioLevelEvent {
  payload: number; // 0.0 to 1.0
}

interface ErrorEvent {
  payload: string; // Error message
}
```

### Application State

The application uses Tauri's managed state pattern with interior mutability for thread-safe access:

```rust
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::task::JoinHandle;

// Main application state managed by Tauri
struct AppState {
    // Recording status flag
    recording: Arc<AtomicBool>,

    // Task handles for cleanup
    recording_task: Arc<Mutex<Option<JoinHandle<Result<()>>>>>,
    processing_task: Arc<Mutex<Option<JoinHandle<Result<()>>>>>,
    output_task: Arc<Mutex<Option<JoinHandle<Result<()>>>>>,

    // Shutdown signal for graceful termination
    shutdown: Arc<AtomicBool>,

    // Loaded model and vocabulary (lazy loaded)
    model: Arc<Mutex<Option<ParakeetTDTModel>>>,
    vocab: Arc<Mutex<Option<Vocabulary>>>,

    // User configuration
    config: Arc<Mutex<AppConfig>>,
}

impl AppState {
    fn new() -> Self {
        Self {
            recording: Arc::new(AtomicBool::new(false)),
            recording_task: Arc::new(Mutex::new(None)),
            processing_task: Arc::new(Mutex::new(None)),
            output_task: Arc::new(Mutex::new(None)),
            shutdown: Arc::new(AtomicBool::new(false)),
            model: Arc::new(Mutex::new(None)),
            vocab: Arc::new(Mutex::new(None)),
            config: Arc::new(Mutex::new(AppConfig::default())),
        }
    }

    fn is_recording(&self) -> bool {
        self.recording.load(Ordering::Relaxed)
    }

    fn set_recording(&self, value: bool) {
        self.recording.store(value, Ordering::Relaxed);
    }
}
```

**State Management Pattern:**
- Use `Arc<AtomicBool>` for simple boolean flags (lock-free)
- Use `Arc<Mutex<T>>` for complex data that needs mutation
- State is registered with Tauri using `.manage(AppState::new())` in main.rs
- Commands access state via `tauri::State<'_, AppState>` parameter
- All Arc clones are cheap (reference counting only)

### Configuration Data Model

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
struct AppConfig {
    encoder_model_path: String,
    decoder_model_path: String,
    vocab_path: String,
    ort_num_threads: u32,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            encoder_model_path: "models/encoder-model.int8.onnx".to_string(),
            decoder_model_path: "models/decoder_joint-model.int8.onnx".to_string(),
            vocab_path: "models/vocab.txt".to_string(),
            ort_num_threads: 4,
        }
    }
}
```

### Audio Configuration

```rust
const SAMPLE_RATE: u32 = 16000;
const CHANNELS: u16 = 1;
const AUDIO_RING_BUFFER_SIZE: usize = 16000 * 10; // 10 seconds
const TOKEN_RING_BUFFER_SIZE: usize = 1024;
```

### Streaming Configuration

```rust
const LEFT_CONTEXT: f32 = 1.0;
const CHUNK_SIZE: f32 = 0.25;
const RIGHT_CONTEXT: f32 = 0.25;
```

### Token to Text Conversion

```rust
/// Convert a single token to text with proper spacing
/// Reused from parakeet-mic implementation
fn token_to_text(token_text: &str, is_first_token: bool) -> Option<String> {
    // Skip special tokens like <unk>, <blank>, etc.
    if token_text.starts_with('<') && token_text.ends_with('>') {
        return None;
    }

    // Handle SentencePiece word-initial marker (▁)
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
```

## Data Models

### Token Result (from parakeet crate)

```rust
pub struct TokenResult {
    pub token_id: usize,
    pub confidence: f32,
    pub timestamp: usize,
    pub text: Option<String>,
}
```

### Frontend State (Svelte)

```typescript
// Using Svelte 5 runes for reactivity
type RecordingState = 'idle' | 'recording' | 'error';

// Main page state
let state = $state<RecordingState>('idle');
let transcript = $state<string>('');
let errorMessage = $state<string | null>(null);
let audioLevel = $state<number>(0);

// Settings page state
interface AppConfig {
  encoder_model_path: string;
  decoder_model_path: string;
  vocab_path: string;
  ort_num_threads: number;
}

let config = $state<AppConfig>({
  encoder_model_path: '',
  decoder_model_path: '',
  vocab_path: '',
  ort_num_threads: 4
});
```

## Error Handling

### Backend Error Handling

1. **Model Loading Errors**
   - Catch errors during model initialization
   - Return descriptive error messages to frontend
   - Prevent recording from starting if models not loaded

2. **Audio Capture Errors**
   - Handle device not found errors
   - Handle permission denied errors
   - Handle stream configuration errors
   - Emit error events to frontend

3. **Processing Errors**
   - Catch errors in processing thread
   - Log errors to console
   - Emit error events to frontend
   - Gracefully shut down threads

4. **Thread Coordination Errors**
   - Handle ring buffer overflow gracefully
   - Detect thread panics and clean up
   - Ensure proper shutdown on errors

### Frontend Error Handling

1. **Command Errors**
   - Display error messages in status indicator
   - Reset to idle state on errors
   - Log errors to console

2. **Event Handling Errors**
   - Gracefully handle malformed events
   - Continue operation on non-critical errors

3. **Settings Validation Errors**
   - Display validation errors inline on settings page
   - Prevent saving invalid configurations
   - Show specific error messages for each field

## Testing Strategy

### Unit Tests

1. **Token to Text Conversion**
   - Test proper spacing with word-initial marker (▁)
   - Test filtering of special tokens (<unk>, <blank>, etc.)
   - Test first token vs subsequent tokens

2. **Audio Level Calculation**
   - Test RMS calculation
   - Test normalization to 0.0-1.0 range

3. **Configuration Validation**
   - Test thread count range validation (1-16)
   - Test file path validation
   - Test default configuration generation
   - Test configuration serialization/deserialization

### Integration Tests

1. **Audio Capture**
   - Test cpal device initialization
   - Test stream configuration
   - Test audio data flow to ring buffer

2. **End-to-End Flow**
   - Test start_recording command
   - Test audio capture → processing → output flow
   - Test stop_recording command
   - Test proper cleanup

### Manual Testing

1. **UI Testing**
   - Verify transcript display formatting
   - Verify audio visualizer updates
   - Verify button states
   - Verify status indicators
   - Verify settings page navigation and form behavior

2. **Audio Quality Testing**
   - Test with different microphones
   - Test with background noise
   - Test transcription accuracy

3. **Configuration Testing**
   - Test saving and loading settings
   - Test with different model paths
   - Test with different thread counts
   - Verify settings persist across app restarts
   - Test invalid input handling

## State Lifecycle Management

### Application Startup

1. **Initialize State**
   ```rust
   fn main() {
       tauri::Builder::default()
           .manage(AppState::new())
           .setup(|app| {
               // Load configuration from disk
               let app_handle = app.handle();
               tauri::async_runtime::spawn(async move {
                   if let Err(e) = load_config_on_startup(&app_handle).await {
                       eprintln!("Failed to load config: {}", e);
                   }
               });
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
   ```

2. **Load Configuration**
   - Attempt to load config.json from app data directory
   - If file doesn't exist, use default configuration
   - Validate loaded configuration
   - Store in AppState

3. **Lazy Model Loading**
   - Models are NOT loaded on startup (large memory footprint)
   - Models are loaded when user clicks "Start" for the first time
   - Or when user explicitly calls "load_models" after changing settings
   - Loading status communicated via events to frontend

### Recording Session Lifecycle

1. **Start Recording**
   - Check: not already recording
   - Check: models are loaded (load if needed)
   - Reset shutdown signal
   - Create ring buffers (audio and token)
   - Spawn recording task → processing task → output task
   - Store task handles in state
   - Set recording flag to true
   - Emit "recording-started" event

2. **During Recording**
   - Recording task: captures audio, pushes to ring buffer
   - Processing task: consumes audio, produces tokens
   - Output task: consumes tokens, emits events to frontend
   - All tasks check shutdown signal periodically

3. **Stop Recording**
   - Set shutdown signal to true
   - Recording task exits (drops ring buffer producer)
   - Processing task detects abandonment, processes remaining audio
   - Output task processes remaining tokens, then exits
   - Wait for all tasks to complete (with 5 second timeout)
   - Clear task handles from state
   - Set recording flag to false
   - Emit "recording-stopped" event

### Configuration Changes

1. **User Modifies Settings**
   - Frontend calls `save_settings` command
   - Backend validates new configuration
   - If valid: persist to disk, update in-memory config
   - If invalid: return error to frontend

2. **Applying New Settings**
   - Model path changes require calling `load_models`
   - Cannot change settings while recording (return error)
   - Frontend should disable settings page during recording

### Cleanup on Exit

1. **Window Close Event**
   - Tauri automatically handles cleanup
   - If recording is active, stop_recording should be called
   - Tasks will be aborted when app exits
   - No explicit cleanup needed (Rust Drop trait handles it)

2. **Graceful Shutdown**
   - Optionally implement a cleanup handler:
   ```rust
   .on_window_event(|window, event| {
       if let tauri::WindowEvent::CloseRequested { .. } = event {
           let state = window.state::<AppState>();
           if state.is_recording() {
               // Stop recording before closing
               tauri::async_runtime::block_on(async {
                   let _ = stop_recording_internal(state).await;
               });
           }
       }
   })
   ```

## Implementation Notes

### Reusing Existing Code

The implementation will leverage existing code from:

1. **parakeet-mic** for:
   - cpal audio capture patterns (adapted to tokio::task::spawn_blocking)
   - Token to text conversion logic with `token_to_text()` function
   - Ring buffer coordination using rtrb
   - Automatic abandonment detection patterns
   - Audio stream building with generic sample format handling

2. **parakeet-ws** for:
   - Async task architecture patterns
   - Event emission patterns (adapted for Tauri events)
   - Output formatting
   - tokio::spawn usage for concurrent tasks

3. **parakeet crate** for:
   - StreamingParakeetTDT engine
   - Model loading with ExecutionConfig
   - Vocabulary handling
   - ContextConfig for streaming parameters

4. **frame-processor crate** for:
   - FrameProcessor trait implementation
   - `process_loop()` with automatic abandonment detection

### Threading Architecture

The application uses a three-task async architecture with tokio:

1. **Recording Task** (tokio::task):
   - Spawns blocking task for cpal audio stream
   - Runs cpal stream in blocking context using `tokio::task::spawn_blocking`
   - Pushes samples to audio ring buffer (rtrb::Producer)
   - Exits on shutdown signal
   - When dropped, processing task detects abandonment

2. **Processing Task** (tokio::task):
   - Runs StreamingParakeetTDT.process_loop()
   - Consumes audio from ring buffer
   - Produces tokens to token ring buffer
   - Automatically detects abandonment and processes remaining data

3. **Output Task** (tokio::task):
   - Consumes tokens from ring buffer in async loop
   - Converts tokens to text using vocabulary
   - Emits Tauri events to frontend
   - Uses `tokio::time::sleep` for yielding
   - Exits on shutdown signal

### Tauri Integration Patterns

Based on Tauri v2 best practices:

1. **Commands**: Use `#[tauri::command]` with async functions
   ```rust
   #[tauri::command]
   async fn my_command(
       app_handle: tauri::AppHandle,
       state: tauri::State<'_, AppState>
   ) -> Result<(), String> {
       // Access state immutably
       let is_recording = state.is_recording();

       // Access mutable state with lock
       let mut config = state.config.lock().unwrap();

       Ok(())
   }
   ```

2. **Events**: Use `app_handle.emit()` to send events to frontend
   ```rust
   // Emit to all windows
   app_handle.emit("event-name", payload)?;

   // Emit to specific window
   if let Some(window) = app_handle.get_webview_window("main") {
       window.emit("event-name", payload)?;
   }
   ```

3. **State Management**:
   - Register state in `main.rs` using `.manage(AppState::new())`
   - Access in commands via `tauri::State<'_, AppState>`
   - Use `Arc` for sharing between threads
   - Prefer `AtomicBool` over `Mutex<bool>` for simple flags

4. **Frontend API**: Use `@tauri-apps/api/core` for `invoke()` and `@tauri-apps/api/event` for `listen()`
   ```typescript
   import { invoke } from '@tauri-apps/api/core';
   import { listen } from '@tauri-apps/api/event';

   // Invoke command
   await invoke('start_recording');

   // Listen to events
   const unlisten = await listen('transcription', (event) => {
       console.log(event.payload);
   });
   ```

5. **Config Storage**: Use `app_handle.path().app_data_dir()` for config file location
   ```rust
   let app_data_dir = app_handle.path().app_data_dir()
       .map_err(|e| format!("Failed to get app data dir: {}", e))?;
   let config_path = app_data_dir.join("config.json");
   ```

6. **Error Handling**: Return `Result<T, String>` from commands for automatic error propagation to frontend
   ```rust
   #[tauri::command]
   async fn my_command() -> Result<String, String> {
       some_operation()
           .map_err(|e| format!("Operation failed: {}", e))
   }
   ```

### Simplifications

To keep the implementation simple:

1. **Fixed Audio Parameters**: Use hardcoded values for sample rate, context sizes (not configurable in settings)
2. **No File Output**: Focus only on real-time display
3. **No Audio Recording**: Don't save audio to disk
4. **Simple Settings**: Only configure model paths and ORT thread count, not streaming parameters

### Platform Considerations

1. **cpal** provides cross-platform audio capture for Linux, macOS, and Windows
2. **Tauri** handles platform-specific windowing and packaging
3. Model files can be bundled with the application or user can specify custom paths via settings
4. Configuration file stored in platform-specific app data directory (Tauri's `app_data_dir()`)

### Performance Considerations

1. **Ring Buffer Sizes**: Large enough to handle processing delays without overflow
2. **Event Throttling**: Audio level events throttled to ~30 FPS for visualization
3. **Thread Priority**: Consider setting higher priority for audio capture thread
4. **Memory Usage**: Monitor ring buffer memory usage with large context sizes

## UI Design

### Layout

The UI will match the parakeet-ws HTML interface:

1. **Full-page transcript area**
   - White background (#ffffff)
   - Serif font (Georgia, 20px)
   - Line height 1.8
   - Padding: 80px 120px
   - Max width: 800px centered
   - Auto-scroll to bottom

2. **Floating control panel** (bottom-right)
   - White background with shadow
   - Border radius: 16px
   - Width: 320px
   - Padding: 20px
   - Contains:
     - Title: "🎤 Parakeet"
     - Subtitle: "Real-time transcription"
     - Status indicator
     - Audio visualizer (50px height)
     - Start/Stop buttons

3. **Status Indicator**
   - Ready: Green background (#efe), green text (#3c3)
   - Recording: Yellow background (#fef3cd), dark yellow text (#856404)
   - Error: Red background (#fee), red text (#c33)
   - Recording shows pulsing red dot indicator

4. **Audio Visualizer**
   - Canvas element (50px height)
   - Light gray background (#f7fafc)
   - Blue waveform (#667eea)
   - Updates at 30 FPS during recording

5. **Buttons**
   - Start: Purple (#667eea), white text
   - Stop: Red (#f56565), white text
   - Settings: Gray (#95a5a6), white text (gear icon)
   - Rounded corners (8px)
   - Hover effects with shadow
   - Disabled state (50% opacity)

6. **Settings Page Layout**
   - Clean form layout with labeled inputs
   - File path inputs with browse button (future enhancement)
   - Number input with min/max constraints for thread count
   - Save button (purple) and Cancel button (gray)
   - Back navigation to main page
   - Validation error messages displayed inline
   - Same color scheme as main page

### Color Scheme

- Primary: #667eea (purple)
- Danger: #f56565 (red)
- Success: #3c3 (green)
- Warning: #856404 (dark yellow)
- Background: #f5f5f0 (light beige)
- Paper: #ffffff (white)
- Text: #2c3e50 (dark blue-gray)
- Muted: #95a5a6 (gray)
