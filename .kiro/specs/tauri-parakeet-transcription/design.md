# Design Document

## Overview

This design document outlines the architecture for a Tauri desktop application that provides real-time speech transcription using the Parakeet TDT model. The application consists of a Rust backend handling audio capture via cpal and transcription processing, and a Svelte frontend providing a simple, clean user interface matching the existing parakeet-ws HTML interface.

The design follows a multi-threaded architecture with:
- Main Tauri thread handling commands and events
- Audio capture thread using cpal for microphone input
- Processing thread running the StreamingParakeetTDT engine
- Output thread consuming tokens and emitting events to the frontend

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Svelte Frontend                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Transcript  │  │  Visualizer  │  │   Controls   │      │
│  │   Display    │  │   Canvas     │  │   (Buttons)  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
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
│                                    Emit "transcription" event│
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

#### Backend (Rust)

1. **Tauri Commands**
   - `start_recording()`: Initializes audio capture and processing
   - `stop_recording()`: Stops all threads and cleans up resources

2. **Audio Capture Module** (using cpal)
   - Initializes default input device
   - Configures stream for 16kHz mono audio
   - Captures audio samples and pushes to ring buffer
   - Calculates audio levels for visualization

3. **Processing Module**
   - Runs StreamingParakeetTDT in dedicated thread
   - Consumes audio from ring buffer
   - Produces tokens to token ring buffer
   - Uses existing FrameProcessor trait

4. **Output Module**
   - Consumes tokens from ring buffer
   - Converts tokens to text using vocabulary
   - Emits Tauri events with transcription text
   - Calculates and emits audio level data

5. **State Management**
   - Global state using Arc<Mutex<AppState>>
   - Tracks recording status
   - Manages thread handles for cleanup

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
```

### Tauri Events

```typescript
// Event emitted when new transcription text is available
interface TranscriptionEvent {
  text: string;
  timestamp: number;
}

// Event emitted for audio visualization
interface AudioLevelEvent {
  level: number; // 0.0 to 1.0
}

// Event emitted on errors
interface ErrorEvent {
  message: string;
}
```

### Application State

```rust
struct AppState {
    recording: Arc<Mutex<bool>>,
    audio_thread: Arc<Mutex<Option<JoinHandle<()>>>>,
    processing_thread: Arc<Mutex<Option<JoinHandle<()>>>>,
    output_thread: Arc<Mutex<Option<JoinHandle<()>>>>,
    shutdown: Arc<AtomicBool>,
    model: Arc<Mutex<Option<ParakeetTDTModel>>>,
    vocab: Arc<Mutex<Option<Vocabulary>>>,
}
```

### Audio Configuration

```rust
const SAMPLE_RATE: u32 = 16000;
const CHANNELS: u16 = 1;
const BUFFER_SIZE: usize = 4096;
const AUDIO_RING_BUFFER_SIZE: usize = 16000 * 10; // 10 seconds
const TOKEN_RING_BUFFER_SIZE: usize = 1024;
```

### Streaming Configuration

```rust
const LEFT_CONTEXT: f32 = 1.0;
const CHUNK_SIZE: f32 = 0.25;
const RIGHT_CONTEXT: f32 = 0.25;
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
type RecordingState = 'idle' | 'recording' | 'error';

interface AppState {
  state: RecordingState;
  transcript: string;
  errorMessage: string | null;
  audioLevel: number;
}
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

## Testing Strategy

### Unit Tests

1. **Token to Text Conversion**
   - Test proper spacing with word-initial marker (▁)
   - Test filtering of special tokens (<unk>, <blank>, etc.)
   - Test first token vs subsequent tokens

2. **Audio Level Calculation**
   - Test RMS calculation
   - Test normalization to 0.0-1.0 range

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

2. **Audio Quality Testing**
   - Test with different microphones
   - Test with background noise
   - Test transcription accuracy

## Implementation Notes

### Reusing Existing Code

The implementation will leverage existing code from:

1. **parakeet-ws** for:
   - Token to text conversion logic
   - Ring buffer architecture
   - Thread coordination patterns
   - Output formatting

2. **parakeet crate** for:
   - StreamingParakeetTDT engine
   - Model loading
   - Vocabulary handling

3. **frame-processor crate** for:
   - FrameProcessor trait implementation

### Simplifications

To keep the implementation simple:

1. **No Configuration UI**: Use hardcoded values for sample rate, context sizes
2. **No File Output**: Focus only on real-time display
3. **No Audio Recording**: Don't save audio to disk
4. **Single Model Path**: Assume models are in `models/` directory relative to executable
5. **No Settings Panel**: Remove WebSocket URL configuration (not needed)

### Platform Considerations

1. **cpal** provides cross-platform audio capture for Linux, macOS, and Windows
2. **Tauri** handles platform-specific windowing and packaging
3. Model files must be bundled with the application or placed in a known location

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
   - Rounded corners (8px)
   - Hover effects with shadow
   - Disabled state (50% opacity)

### Color Scheme

- Primary: #667eea (purple)
- Danger: #f56565 (red)
- Success: #3c3 (green)
- Warning: #856404 (dark yellow)
- Background: #f5f5f0 (light beige)
- Paper: #ffffff (white)
- Text: #2c3e50 (dark blue-gray)
- Muted: #95a5a6 (gray)
