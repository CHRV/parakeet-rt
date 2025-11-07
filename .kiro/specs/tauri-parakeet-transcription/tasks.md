# Implementation Plan

- [ ] 1. Set up Rust backend dependencies and project structure
  - Add required dependencies to `parakeet-mic-tauri/src-tauri/Cargo.toml`: cpal, parakeet, frame-processor, rtrb, tokio, anyhow
  - Configure workspace dependencies to reference local crates
  - _Requirements: 6.2, 6.3, 7.1, 7.2_

- [ ] 2. Implement core backend state management and initialization
  - [ ] 2.1 Create AppState struct with recording status, thread handles, and model storage
    - Define AppState with Arc<Mutex<>> for thread-safe access
    - Include fields for model, vocabulary, threads, and shutdown flag
    - _Requirements: 7.1, 7.2_

  - [ ] 2.2 Implement model and vocabulary loading on application startup
    - Load ParakeetTDTModel from models directory
    - Load Vocabulary from vocab.txt file
    - Handle and return errors if loading fails
    - _Requirements: 7.1, 7.2, 7.3, 7.4_

- [ ] 3. Implement audio capture using cpal
  - [ ] 3.1 Create audio capture module with cpal device initialization
    - Get default input device
    - Configure stream for 16kHz mono audio
    - Set up error handling for device not found and permission errors
    - _Requirements: 1.1, 6.2_

  - [ ] 3.2 Implement audio stream callback to push samples to ring buffer
    - Create audio ring buffer (rtrb)
    - Push audio samples from cpal callback to ring buffer
    - Calculate RMS audio levels for visualization
    - _Requirements: 1.1, 6.5_

- [ ] 4. Implement processing thread with StreamingParakeetTDT
  - [ ] 4.1 Create processing thread that consumes audio from ring buffer
    - Initialize StreamingParakeetTDT with context configuration
    - Set up audio producer and token consumer ring buffers
    - Implement processing loop using FrameProcessor trait
    - _Requirements: 6.3_

  - [ ] 4.2 Configure streaming parameters (left context, chunk size, right context)
    - Use constants: LEFT_CONTEXT=1.0s, CHUNK_SIZE=0.25s, RIGHT_CONTEXT=0.25s
    - Create ContextConfig with sample rate 16000
    - _Requirements: 6.3_

- [ ] 5. Implement output thread for token processing
  - [ ] 5.1 Create output thread that consumes tokens from ring buffer
    - Poll token ring buffer in loop
    - Convert tokens to text using vocabulary
    - Filter special tokens (tokens starting with < and ending with >)
    - _Requirements: 2.1, 2.2, 6.4_

  - [ ] 5.2 Implement token-to-text conversion with proper spacing
    - Handle word-initial marker (▁) for spacing
    - Add space before words with ▁ prefix (except first token)
    - Detect sentence-ending punctuation for line breaks
    - _Requirements: 2.3, 2.4_

  - [ ] 5.3 Emit Tauri events with transcription text and audio levels
    - Emit "transcription" event with text payload
    - Emit "audio-level" event with level value (0.0-1.0)
    - Throttle audio level events to ~30 FPS
    - _Requirements: 2.1, 3.2, 6.4, 6.5_

- [ ] 6. Implement Tauri commands for recording control
  - [ ] 6.1 Implement start_recording command
    - Check if already recording and return error if true
    - Spawn audio capture thread with cpal
    - Spawn processing thread with StreamingParakeetTDT
    - Spawn output thread for token consumption
    - Update AppState to mark recording as active
    - _Requirements: 1.1, 1.2, 6.1, 7.4_

  - [ ] 6.2 Implement stop_recording command
    - Set shutdown flag to signal threads to stop
    - Wait for threads to complete (join handles)
    - Clean up resources (drop ring buffers, stop cpal stream)
    - Update AppState to mark recording as inactive
    - _Requirements: 1.3, 1.4, 6.1_

  - [ ] 6.3 Implement error handling and error event emission
    - Catch errors in commands and return Result<(), String>
    - Emit "error" event to frontend with error message
    - Handle model loading errors, audio capture errors, and thread errors
    - _Requirements: 4.3, 6.6, 7.3_

- [ ] 7. Create Svelte frontend page component
  - [ ] 7.1 Create main page component with layout structure
    - Create full-page transcript display area with paper styling
    - Create floating control panel in bottom-right corner
    - Set up component state for recording status, transcript, and audio level
    - _Requirements: 5.1, 5.2_

  - [ ] 7.2 Implement transcript display with auto-scroll
    - Bind transcript text to display area
    - Apply serif font styling (Georgia, 20px, line-height 1.8)
    - Implement auto-scroll to bottom when new text arrives
    - Handle line breaks after sentence-ending punctuation
    - _Requirements: 2.2, 2.4, 2.5, 5.3_

  - [ ] 7.3 Implement audio visualizer canvas
    - Create canvas element with 50px height
    - Draw waveform based on audio level data
    - Update visualization at 30 FPS during recording
    - Show empty state when not recording
    - _Requirements: 3.1, 3.2, 3.3_

  - [ ] 7.4 Implement control buttons (Start/Stop)
    - Create Start button that calls start_recording command
    - Create Stop button that calls stop_recording command
    - Disable Start button while recording, enable Stop button
    - Disable Stop button while not recording, enable Start button
    - Apply button styling with hover effects
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 5.4_

  - [ ] 7.5 Implement status indicator
    - Display "Ready" status when idle
    - Display "Recording" status with pulsing red dot when recording
    - Display error message when error occurs
    - Apply color-coded styling (green for ready, yellow for recording, red for error)
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 5.4_

  - [ ] 7.6 Set up Tauri event listeners
    - Listen for "transcription" event and append text to transcript
    - Listen for "audio-level" event and update visualizer
    - Listen for "error" event and display error status
    - _Requirements: 2.1, 3.2, 4.3_

- [ ] 8. Apply UI styling to match parakeet-ws interface
  - [ ] 8.1 Style transcript display area
    - Apply white background, serif font, proper padding and margins
    - Set max-width 800px and center content
    - Add "Waiting for transcription..." placeholder text
    - _Requirements: 5.2, 5.3_

  - [ ] 8.2 Style floating control panel
    - Apply white background with shadow and rounded corners
    - Position in bottom-right corner (30px from edges)
    - Set width to 320px with 20px padding
    - _Requirements: 5.2, 5.4_

  - [ ] 8.3 Apply color scheme and button styles
    - Use purple (#667eea) for Start button, red (#f56565) for Stop button
    - Add hover effects with shadows and transform
    - Style status indicator with color-coded backgrounds
    - Apply disabled state styling (50% opacity)
    - _Requirements: 5.5_

- [ ] 9. Wire up frontend and backend integration
  - [ ] 9.1 Import Tauri API in Svelte component
    - Import invoke function for calling commands
    - Import listen function for event handling
    - _Requirements: 6.1, 6.4_

  - [ ] 9.2 Connect button click handlers to Tauri commands
    - Call start_recording command on Start button click
    - Call stop_recording command on Stop button click
    - Handle command errors and update UI state
    - _Requirements: 1.1, 1.3_

  - [ ] 9.3 Set up event listeners for transcription and audio data
    - Listen for transcription events and update transcript state
    - Listen for audio-level events and update visualizer
    - Listen for error events and update error state
    - _Requirements: 2.1, 3.2, 4.3_

- [ ] 10. Test and validate the application
  - [ ] 10.1 Test basic recording flow
    - Start recording and verify audio capture begins
    - Speak into microphone and verify transcription appears
    - Stop recording and verify cleanup occurs
    - _Requirements: 1.1, 1.3, 2.1_

  - [ ] 10.2 Test UI behavior and styling
    - Verify button states change correctly
    - Verify status indicator updates correctly
    - Verify audio visualizer displays waveform
    - Verify transcript auto-scrolls
    - _Requirements: 1.2, 1.4, 2.5, 3.1, 4.1, 4.2, 4.4_

  - [ ] 10.3 Test error handling
    - Test behavior when models directory is missing
    - Test behavior when microphone permission is denied
    - Verify error messages display correctly
    - _Requirements: 4.3, 7.3_
