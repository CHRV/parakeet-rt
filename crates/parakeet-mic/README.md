# Parakeet Microphone Transcription

Real-time speech transcription using microphone input with the Parakeet TDT model.

## Features

- Real-time audio capture from microphone using CPAL
- Streaming inference with configurable latency
- Cross-platform audio support (Linux, macOS, Windows)
- Command-line interface with customizable parameters

## Usage

### Basic Usage

```bash
cargo run --bin parakeet-mic
```

### List Available Audio Devices

```bash
cargo run --bin parakeet-mic -- --list-devices
```

### Custom Configuration

```bash
cargo run --bin parakeet-mic -- \
  --models /path/to/models \
  --sample-rate 16000 \
  --left-context 1.0 \
  --chunk-size 0.25 \
  --right-context 0.25 \
  --device 0
```

### Save Output to File

```bash
# Save as text file
cargo run --bin parakeet-mic -- --output transcription.txt

# Save as JSON
cargo run --bin parakeet-mic -- --output results.json --format json

# Save as CSV
cargo run --bin parakeet-mic -- --output data.csv --format csv

# Append to existing file
cargo run --bin parakeet-mic -- --output transcription.txt --append
```

### Save Audio Recording

```bash
# Save audio to WAV file
cargo run --bin parakeet-mic -- --save-audio recording.wav

# Save both transcription and audio
cargo run --bin parakeet-mic -- --output transcription.txt --save-audio recording.wav

# Complete example with all options
cargo run --bin parakeet-mic -- \
  --output results.json \
  --format json \
  --save-audio session.wav \
  --sample-rate 16000 \
  --device 0
```

## Command Line Options

- `--models, -m`: Path to the models directory (default: "models")
- `--sample-rate, -s`: Sample rate for audio capture (default: 16000)
- `--left-context`: Left context duration in seconds (default: 1.0)
- `--chunk-size`: Chunk size in seconds (default: 0.25)
- `--right-context`: Right context duration in seconds (default: 0.25)
- `--list-devices`: List available audio input devices
- `--device, -d`: Device index to use (see --list-devices)
- `--output, -o`: Output file to save transcription results
- `--format`: Output format - txt, json, or csv (default: txt)
- `--append`: Append to output file instead of overwriting
- `--save-audio`: Save received audio to WAV file

## Requirements

- ONNX models in the specified models directory
- Audio input device (microphone)
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

The application consists of:

1. **Audio Capture**: Uses CPAL to capture real-time audio from microphone
2. **Streaming Engine**: Processes audio chunks with configurable context
3. **Token Detection**: Outputs detected speech tokens with timestamps
4. **Real-time Processing**: Low-latency inference pipeline
5. **Output Storage**: Saves results in multiple formats (TXT, JSON, CSV)
6. **Audio Recording**: Saves received audio to WAV files for analysis

## Performance

- Theoretical latency: chunk_size + right_context (default: 500ms)
- Memory usage: Optimized ring buffers for audio and token streams
- CPU usage: Depends on model complexity and chunk size

## Troubleshooting

### No Audio Devices Found
- Check that your microphone is connected and recognized by the system
- Try running with `--list-devices` to see available devices

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
- Audio recording continues until the application is stopped