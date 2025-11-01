# Audio Saving Demo

This example demonstrates how to use the audio saving functionality of parakeet-mic.

## Basic Audio Recording

Save audio to a WAV file while performing real-time transcription:

```bash
cargo run --bin parakeet-mic -- --save-audio recording.wav
```

## Combined Output

Save both transcription results and audio:

```bash
# Save transcription as JSON and audio as WAV
cargo run --bin parakeet-mic -- \
  --output transcription.json \
  --format json \
  --save-audio session.wav

# Save transcription as CSV and audio as WAV
cargo run --bin parakeet-mic -- \
  --output data.csv \
  --format csv \
  --save-audio recording.wav
```

## Custom Configuration

Use custom audio settings with recording:

```bash
cargo run --bin parakeet-mic -- \
  --sample-rate 22050 \
  --save-audio high_quality.wav \
  --output results.txt \
  --left-context 1.5 \
  --chunk-size 0.2
```

## Output Files

After running the application, you'll have:

1. **Audio File** (`recording.wav`):
   - Format: WAV (32-bit float)
   - Channels: Mono
   - Sample Rate: As specified (default 16000 Hz)
   - Contains the complete audio session

2. **Transcription File** (optional):
   - Contains detected tokens with timestamps
   - Format depends on `--format` option
   - Synchronized with the audio file

## Use Cases

- **Dataset Creation**: Record audio with synchronized transcription labels
- **Quality Analysis**: Compare audio quality with transcription accuracy
- **Debugging**: Analyze audio segments where transcription failed
- **Research**: Create paired audio-text datasets for model training
- **Archival**: Keep complete records of transcription sessions

## File Size Considerations

Audio files can become large:
- 16 kHz, 32-bit float: ~64 KB per second
- 22 kHz, 32-bit float: ~88 KB per second
- 10-minute session at 16 kHz: ~38 MB

Monitor disk space for long recording sessions.