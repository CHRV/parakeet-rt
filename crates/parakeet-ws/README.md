# Parakeet WebSocket Transcription

Real-time speech transcription using WebSocket audio streaming with the Parakeet TDT model.

## Features

- Real-time audio streaming via WebSocket
- Browser-based audio capture with HTML interface
- Streaming inference with configurable latency
- Cross-platform support (Linux, macOS, Windows)
- Command-line interface with customizable parameters

## Usage

### Basic Usage

1. Start the WebSocket server:
```bash
cargo run --bin parakeet-ws
```

2. Open `index.html` in your web browser

3. Click "Start Recording" to begin streaming audio

### Custom Configuration

```bash
cargo run --bin parakeet-ws -- \
  --models /path/to/models \
  --sample-rate 16000 \
  --left-context 1.0 \
  --chunk-size 0.25 \
  --right-context 0.25 \
  --address 127.0.0.1:8080
```

### Save Output to File

```bash
# Save as text file
cargo run --bin parakeet-ws -- --output transcription.txt

# Save as JSON
cargo run --bin parakeet-ws -- --output results.json --format json

# Save as CSV
cargo run --bin parakeet-ws -- --output data.csv --format csv

# Append to existing file
cargo run --bin parakeet-ws -- --output transcription.txt --append
```

### Save Audio Recording

```bash
# Save audio to WAV file
cargo run --bin parakeet-ws -- --save-audio recording.wav

# Save both transcription and audio
cargo run --bin parakeet-ws -- --output transcription.txt --save-audio recording.wav

# Complete example with all options
cargo run --bin parakeet-ws -- \
  --output results.json \
  --format json \
  --save-audio session.wav \
  --sample-rate 16000 \
  --address 0.0.0.0:8080
```

## Command Line Options

- `--models, -m`: Path to the models directory (default: "models")
- `--sample-rate, -s`: Sample rate for audio capture (default: 16000)
- `--left-context`: Left context duration in seconds (default: 1.0)
- `--chunk-size`: Chunk size in seconds (default: 0.25)
- `--right-context`: Right context duration in seconds (default: 0.25)
- `--address, -a`: WebSocket server address (default: "127.0.0.1:8080")
- `--output, -o`: Output file to save transcription results
- `--format`: Output format - txt, json, or csv (default: txt)
- `--append`: Append to output file instead of overwriting
- `--save-audio`: Save received audio to WAV file

## Requirements

- ONNX models in the specified models directory
- Modern web browser with WebRTC support
- Rust 1.70+ with Cargo

## Output Formats

### Text Format (default)
```
[1.250s] Token 42 (conf: 0.95)
[1.375s] Token 17 (conf: 0.87)
```

### JSON Format
```json
[
  {
    "timestamp": 20000,
    "token_id": 42,
    "confidence": 0.95,
    "time_seconds": 1.25,
    "session_time": 5.2
  }
]
```

### CSV Format
```csv
timestamp,token_id,confidence,time_seconds
20000,42,0.95,1.25
22000,17,0.87,1.375
```

## Audio Recording Format

The application saves audio in WAV format with the following specifications:
- **Format**: WAV (RIFF)
- **Channels**: 1 (Mono)
- **Sample Rate**: Configurable (default: 16000 Hz)
- **Bit Depth**: 32-bit floating point
- **Encoding**: IEEE Float

This format ensures high quality audio capture suitable for:
- Speech analysis and research
- Model training data collection
- Audio quality debugging
- Playback in standard audio applications

## Architecture

The application uses a WebSocket-based architecture with automatic shutdown coordination:

```
Browser (WebRTC) → WebSocket → [audio ring buffer] → Processing Thread → [token ring buffer] → Output Thread
```

### Components

1. **WebSocket Handler**: Receives real-time audio from browser via WebSocket
2. **Processing Thread**: Processes audio chunks using the FrameProcessor trait
3. **Output Thread**: Consumes tokens and writes transcription results
4. **Ring Buffers**: Lock-free communication between threads

### Automatic Shutdown Coordination

The application leverages ring buffer abandonment detection for graceful shutdown:

- **When Ctrl+C is pressed or client disconnects**: All threads receive shutdown signal
- **WebSocket handler exits**: Audio producer is dropped, processing thread detects abandonment
- **Processing thread detects upstream abandonment**: Processes remaining buffered audio, then drops token producer
- **Output thread detects downstream abandonment**: Drains remaining tokens and exits
- **No explicit coordination needed**: Ring buffer abandonment detection handles cleanup automatically

This design eliminates the need for complex shutdown signaling and ensures all buffered data is processed before exit.

### Features

1. **WebSocket Audio Streaming**: Receives real-time audio from browser via WebSocket
2. **Browser Interface**: User-friendly HTML page with audio visualization
3. **Streaming Engine**: Processes audio chunks with configurable context
4. **Token Detection**: Outputs detected speech tokens with timestamps
5. **Real-time Processing**: Low-latency inference pipeline
6. **Output Storage**: Saves results in multiple formats (TXT, JSON, CSV)
7. **Audio Recording**: Saves received audio to WAV files for analysis

## Performance

- Theoretical latency: chunk_size + right_context (default: 500ms)
- Memory usage: Optimized ring buffers for audio and token streams
- CPU usage: Depends on model complexity and chunk size

## Troubleshooting

### WebSocket Connection Failed
- Ensure the server is running before opening the HTML page
- Check that the WebSocket URL in the browser matches the server address
- If using a different address, update the URL in the HTML page

### Microphone Access Denied
- Grant microphone permissions in your browser
- Check browser settings for microphone access
- Try using HTTPS if accessing from a remote machine

### Models Not Found
- Ensure ONNX models are in the correct directory
- Use `--models` flag to specify custom model path

### High Latency
- Reduce `--chunk-size` and `--right-context` for lower latency
- Note that smaller chunks may reduce transcription quality

### Output File Issues
- Check file permissions if unable to write output
- Use `--append` flag to add to existing files instead of overwriting
- JSON format writes all tokens at the end for valid JSON structure

### Audio Recording Issues
- Audio files are saved in 32-bit float WAV format
- Large audio files may consume significant disk space
- Audio recording continues until the client disconnects

### Browser Compatibility
- Use a modern browser with WebRTC support (Chrome, Firefox, Edge, Safari)
- Ensure JavaScript is enabled
- Check browser console for error messages