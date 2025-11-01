# Parakeet TDT Audio Transcription Example

This example demonstrates how to use the Parakeet TDT model to transcribe audio files to text.

## Prerequisites

1. **Model files**: Ensure you have the required model files in the `models/` directory:
   - `encoder-model.int8.onnx` (or `encoder.onnx`)
   - `decoder_joint-model.int8.onnx` (or `decoder_joint.onnx`)
   - `nemo128.onnx` (preprocessor)
   - `vocab.txt` (vocabulary file)

2. **Audio files**: The example can process WAV files. Test files are available in:
   - `crates/vad/tests/audio/sample_1.wav`
   - `crates/vad/tests/audio/birds.wav`
   - `crates/vad/tests/audio/rooster.wav`

## Running the Example

### Basic Usage (uses default test file)
```bash
cargo run --example transcribe_audio --features audio
```

### Specify a custom audio file
```bash
cargo run --example transcribe_audio --features audio -- path/to/your/audio.wav
```

### Example with test files
```bash
# Transcribe sample_1.wav
cargo run --example transcribe_audio --features audio -- crates/vad/tests/audio/sample_1.wav

# Transcribe birds.wav
cargo run --example transcribe_audio --features audio -- crates/vad/tests/audio/birds.wav
```

## What the Example Does

1. **Loads Audio**: Reads WAV files and converts them to the required format (16kHz mono)
2. **Loads Model**: Initializes the Parakeet TDT model with preprocessor, encoder, and decoder
3. **Loads Vocabulary**: Loads the vocabulary file for token-to-text conversion
4. **Runs Inference**: Processes the audio through the model pipeline
5. **Decodes Results**: Converts token IDs to readable text with timestamps
6. **Displays Output**: Shows the full transcription and detailed token information

## Output Format

The example provides:
- **Full transcription text**: Complete transcribed text
- **Timed tokens**: Individual tokens with start/end timestamps
- **Debug information**: Raw token IDs and frame indices for troubleshooting

## Audio Requirements

- **Format**: WAV files (other formats not supported in this example)
- **Sample Rate**: Any (automatically resampled to 16kHz)
- **Channels**: Mono or stereo (stereo converted to mono)
- **Duration**: No specific limits, but longer files will take more time

## Troubleshooting

- **Model files not found**: Ensure all ONNX files are in the `models/` directory
- **Vocabulary not found**: Check that `vocab.txt` exists in `models/`
- **Audio file errors**: Verify the audio file is a valid WAV file
- **No transcription**: The model might not detect speech in the audio

## Performance Notes

- First run may be slower due to model loading
- Processing time depends on audio length and hardware
- The example uses CPU inference by default