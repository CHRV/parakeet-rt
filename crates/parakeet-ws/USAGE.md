# Quick Start Guide

## Step 1: Start the Server

```bash
cargo run --bin parakeet-ws
```

You should see output like:
```
=== Parakeet WebSocket Transcription Server ===
Loading TDT model from models...
✓ Model loaded successfully
✓ Vocabulary loaded (1024 tokens)
Streaming configuration:
  • Left context: 1.0s
  • Chunk size: 250ms
  • Right context: 250ms
  • Total latency: 500ms
✓ WebSocket server listening on ws://127.0.0.1:8080

🌐 Waiting for client connection...
Open the HTML page in your browser to start transcription
```

## Step 2: Open the HTML Page

Open `crates/parakeet-ws/index.html` in your web browser. You can:
- Double-click the file
- Or drag it into your browser
- Or use: `file:///path/to/crates/parakeet-ws/index.html`

## Step 3: Start Recording

1. Click the "Start Recording" button
2. Grant microphone permissions when prompted
3. Start speaking
4. Watch the transcription appear in the server console

## Step 4: Stop Recording

Click the "Stop Recording" button when done.

## Advanced Options

### Custom Server Address

To bind to a different address (e.g., to allow remote connections):

```bash
cargo run --bin parakeet-ws -- --address 0.0.0.0:8080
```

Then update the WebSocket URL in the HTML page to match your server's IP.

### Save Transcription to File

```bash
cargo run --bin parakeet-ws -- --output transcription.txt
```

### Save Audio Recording

```bash
cargo run --bin parakeet-ws -- --save-audio recording.wav
```

### All Options Combined

```bash
cargo run --bin parakeet-ws -- \
  --address 0.0.0.0:8080 \
  --output transcription.json \
  --format json \
  --save-audio recording.wav \
  --sample-rate 16000
```

## Troubleshooting

### "Connection Failed" in Browser
- Make sure the server is running first
- Check that the WebSocket URL matches the server address
- Check firewall settings if using a remote connection

### "Microphone Access Denied"
- Click the browser's address bar and grant microphone permissions
- Check browser settings for site permissions
- Try a different browser if issues persist

### No Transcription Output
- Ensure the models directory exists and contains the required ONNX files
- Check the server console for error messages
- Verify your microphone is working (test in other applications)

### Poor Transcription Quality
- Speak clearly and at a moderate pace
- Reduce background noise
- Ensure your microphone is positioned correctly
- Check audio levels in your system settings
