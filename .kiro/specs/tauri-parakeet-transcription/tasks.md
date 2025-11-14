# Implementation Plan

- [x] 1. Set up Rust backend dependencies and project structure
  - Add required dependencies to `parakeet-mic-tauri/src-tauri/Cargo.toml`: cpal, parakeet, frame-processor, rtrb, tokio (with full features), anyhow, serde, serde_json
  - Configure workspace dependencies to reference local crates (parakeet, frame-processor)
  - Enable tokio runtime in Tauri with multi-thread feature
  - _Requirements: 6.2, 6.3, 7.1, 7.2, 8.7_

- [x] 2. Implement core backend state management and initialization
  - [x] 2.1 Create AppState struct with recording status, task handles, and model storage
    - Define AppState with Arc<Mutex<>> for thread-safe access
    - Include fields for model, vocabulary, tokio task handles, shutdown flag, and config
    - Use tokio::task::JoinHandle for all async tasks
    - _Requirements: 7.1, 7.2, 8.7, 8.8_

  - [x] 2.2 Create AppConfig struct with serde serialization
    - Define AppConfig with encoder_model_path, decoder_model_path, vocab_path, ort_num_threads
    - Implement Default trait with default model paths
    - Add serde Serialize and Deserialize derives
    - _Requirements: 8.2, 8.3, 8.4, 8.5, 8.9_

  - [x] 2.3 Implement configuration loading and saving functions
    - Create load_config() function that reads from app data directory
    - Create save_config() function that writes to app data directory using serde_json
    - Use app_handle.path().app_data_dir() for config file location
    - Create directory if it doesn't exist
    - _Requirements: 8.7, 8.8, 8.9_

  - [x] 2.4 Implement model and vocabulary loading on application startup
    - Load ParakeetTDTModel using paths from AppConfig
    - Load Vocabulary from vocab_path in AppConfig
    - Handle and return errors if loading fails
    - Load config on startup, use defaults if config doesn't exist
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 8.8, 8.9_

- [x] 3. Implement audio capture using cpal with async patterns
  - [x] 3.1 Create async audio capture function with cpal device initialization
    - Use tokio::task::spawn_blocking for cpal operations
    - Get default input device using cpal::default_host()
    - Configure StreamConfig for 16kHz mono audio
    - Set up error handling for device not found and permission errors
    - _Requirements: 1.1, 6.2_

  - [x] 3.2 Implement audio stream with generic sample format handling
    - Create build_audio_stream() generic function for F32, I16, U16 formats
    - Build input stream with callback using device.build_input_stream()
    - Convert multi-channel to mono by averaging channels
    - Push audio samples from cpal callback to rtrb ring buffer
    - Calculate RMS audio levels for visualization
    - _Requirements: 1.1, 6.5_

  - [x] 3.3 Implement recording task that keeps stream alive
    - Spawn tokio task that runs cpal stream
    - Use tokio::time::sleep in loop to keep stream alive
    - Exit loop on shutdown signal
    - Drop stream to stop recording
    - _Requirements: 1.1, 1.3_

- [x] 4. Implement processing task with StreamingParakeetTDT
  - [x] 4.1 Create async processing task that consumes audio from ring buffer
    - Spawn tokio::task for processing
    - Initialize StreamingParakeetTDT with context configuration
    - Call process_loop() which handles automatic abandonment detection
    - Return Result from task for error handling
    - _Requirements: 6.3_

  - [x] 4.2 Configure streaming parameters (left context, chunk size, right context)
    - Use constants: LEFT_CONTEXT=1.0s, CHUNK_SIZE=0.25s, RIGHT_CONTEXT=0.25s
    - Create ContextConfig with sample rate 16000
    - Create StreamingParakeetTDT using new_with_vocab()
    - _Requirements: 6.3_

- [x] 5. Implement output task for token processing
  - [x] 5.1 Create token_to_text() helper function
    - Filter special tokens (tokens starting with < and ending with >)
    - Handle word-initial marker (▁) for spacing
    - Add space before words with ▁ prefix (except first token)
    - Return Option<String> for valid text
    - _Requirements: 2.1, 2.2, 2.3_

  - [x] 5.2 Create async output task that consumes tokens from ring buffer
    - Spawn tokio::task for output processing
    - Poll token ring buffer in async loop using tokio::time::sleep for yielding
    - Convert tokens to text using token_to_text() function
    - Track text buffer for sentence detection
    - Continue until shutdown AND ring buffer is empty
    - _Requirements: 2.1, 2.2, 6.4_

  - [x] 5.3 Emit Tauri events with transcription text and audio levels
    - Use app_handle.emit() to emit "transcription" event with text payload
    - Emit "audio-level" event with level value (0.0-1.0)
    - Throttle audio level events to ~30 FPS using timestamp tracking
    - Handle event emission errors gracefully
    - _Requirements: 2.1, 3.2, 6.4, 6.5_

- [x] 6. Implement Tauri commands for recording control and settings
  - [x] 6.1 Implement start_recording command
    - Mark as async function with #[tauri::command]
    - Check if already recording and return error if true
    - Create rtrb ring buffers for audio and tokens
    - Spawn audio capture task with cpal using tokio::task::spawn
    - Spawn processing task with StreamingParakeetTDT
    - Spawn output task for token consumption
    - Store task handles in AppState
    - Update AppState to mark recording as active
    - _Requirements: 1.1, 1.2, 6.1, 7.4_

  - [x] 6.2 Implement stop_recording command
    - Mark as async function with #[tauri::command]
    - Set shutdown flag to signal tasks to stop
    - Await all task handles to complete
    - Handle task results and log errors
    - Clean up resources (task handles dropped automatically)
    - Update AppState to mark recording as inactive
    - _Requirements: 1.3, 1.4, 6.1_

  - [x] 6.3 Implement get_settings command
    - Mark as async function with #[tauri::command]
    - Return current AppConfig from AppState
    - Return Result<AppConfig, String> for error handling
    - _Requirements: 8.1_

  - [x] 6.4 Implement save_settings command
    - Mark as async function with #[tauri::command]
    - Accept AppConfig parameter from frontend
    - Validate settings (file paths exist, thread count in range 1-16)
    - Save config to disk using save_config() function
    - Update AppState with new config
    - Return Result<(), String> for error handling
    - _Requirements: 8.6, 8.7_

  - [x] 6.5 Implement load_models command
    - Mark as async function with #[tauri::command]
    - Check if currently recording and return error if true
    - Load models using current config paths
    - Update AppState with new model and vocabulary
    - Return Result<(), String> for error handling
    - _Requirements: 7.1, 7.2, 7.3_

  - [x] 6.6 Implement error handling and error event emission
    - Catch errors in commands and return Result<(), String>
    - Emit "error" event to frontend with error message using app_handle.emit()
    - Handle model loading errors, audio capture errors, and task errors
    - _Requirements: 4.3, 6.6, 7.3_

- [x] 7. Create Svelte frontend main page component
  - [x] 7.1 Create main page component with layout structure
    - Create full-page transcript display area with paper styling
    - Create floating control panel in bottom-right corner
    - Set up component state using Svelte 5 $state runes for recording status, transcript, and audio level
    - Add Settings button to navigate to settings page
    - _Requirements: 5.1, 5.2, 8.1_

  - [x] 7.2 Implement transcript display with auto-scroll
    - Bind transcript text to display area
    - Apply serif font styling (Georgia, 20px, line-height 1.8)
    - Implement auto-scroll to bottom when new text arrives using $effect
    - Handle line breaks after sentence-ending punctuation
    - _Requirements: 2.2, 2.4, 2.5, 5.3_

  - [x] 7.3 Implement audio visualizer canvas
    - Create canvas element with 50px height
    - Draw waveform based on audio level data
    - Update visualization using $effect when audioLevel changes
    - Show empty state when not recording
    - _Requirements: 3.1, 3.2, 3.3_

  - [x] 7.4 Implement control buttons (Start/Stop/Settings)
    - Create Start button that calls start_recording command using invoke()
    - Create Stop button that calls stop_recording command
    - Create Settings button that navigates to /settings page
    - Disable Start button while recording, enable Stop button
    - Disable Stop button while not recording, enable Start button
    - Apply button styling with hover effects
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 5.4, 8.1_

  - [x] 7.5 Implement status indicator
    - Display "Ready" status when idle
    - Display "Recording" status with pulsing red dot when recording
    - Display error message when error occurs
    - Apply color-coded styling (green for ready, yellow for recording, red for error)
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 5.4_

  - [x] 7.6 Set up Tauri event listeners
    - Import listen from @tauri-apps/api/event
    - Listen for "transcription" event and append text to transcript
    - Listen for "audio-level" event and update visualizer
    - Listen for "error" event and display error status
    - Set up listeners in onMount lifecycle
    - _Requirements: 2.1, 3.2, 4.3_

- [x] 8. Apply UI styling to match parakeet-ws interface
  - [x] 8.1 Style transcript display area
    - Apply white background, serif font, proper padding and margins
    - Set max-width 800px and center content
    - Add "Waiting for transcription..." placeholder text
    - _Requirements: 5.2, 5.3_

  - [x] 8.2 Style floating control panel
    - Apply white background with shadow and rounded corners
    - Position in bottom-right corner (30px from edges)
    - Set width to 320px with 20px padding
    - _Requirements: 5.2, 5.4_

  - [x] 8.3 Apply color scheme and button styles
    - Use purple (#667eea) for Start button, red (#f56565) for Stop button
    - Add hover effects with shadows and transform
    - Style status indicator with color-coded backgrounds
    - Apply disabled state styling (50% opacity)
    - _Requirements: 5.5_

- [x] 9. Create Svelte settings page component
  - [x] 9.1 Create settings page route at settings/+page.svelte
    - Create form layout with labeled inputs
    - Set up component state using Svelte 5 $state for config fields
    - Add navigation back to main page
    - _Requirements: 8.1_

  - [x] 9.2 Implement settings form inputs
    - Create text input for encoder_model_path
    - Create text input for decoder_model_path
    - Create text input for vocab_path
    - Create number input for ort_num_threads with min=1 max=16
    - Bind inputs to state variables
    - _Requirements: 8.2, 8.3, 8.4, 8.5_

  - [x] 9.3 Implement settings loading on page mount
    - Call get_settings command using invoke() in onMount
    - Update state with loaded config
    - Handle errors and display error message
    - _Requirements: 8.8_

  - [x] 9.4 Implement settings validation and saving
    - Create Save button that calls save_settings command
    - Validate inputs before saving (paths not empty, thread count in range)
    - Display validation errors inline
    - Show success message after successful save
    - Handle command errors and display error message
    - _Requirements: 8.6, 8.7_

  - [x] 9.5 Implement Cancel button
    - Create Cancel button that navigates back to main page
    - Discard unsaved changes
    - _Requirements: 8.1_

- [x] 10. Wire up frontend and backend integration
  - [x] 10.1 Import Tauri API in Svelte components
    - Import invoke from @tauri-apps/api/core for calling commands
    - Import listen from @tauri-apps/api/event for event handling
    - _Requirements: 6.1, 6.4_

  - [x] 10.2 Connect button click handlers to Tauri commands
    - Call start_recording command on Start button click
    - Call stop_recording command on Stop button click
    - Handle command errors and update UI state
    - _Requirements: 1.1, 1.3_

  - [x] 10.3 Set up event listeners for transcription and audio data
    - Listen for transcription events and update transcript state
    - Listen for audio-level events and update visualizer
    - Listen for error events and update error state
    - Clean up listeners on component unmount
    - _Requirements: 2.1, 3.2, 4.3_

- [ ] 11. Test and validate the application
  - [ ]* 11.1 Test basic recording flow
    - Start recording and verify audio capture begins
    - Speak into microphone and verify transcription appears
    - Stop recording and verify cleanup occurs
    - _Requirements: 1.1, 1.3, 2.1_

  - [ ]* 11.2 Test UI behavior and styling
    - Verify button states change correctly
    - Verify status indicator updates correctly
    - Verify audio visualizer displays waveform
    - Verify transcript auto-scrolls
    - _Requirements: 1.2, 1.4, 2.5, 3.1, 4.1, 4.2, 4.4_

  - [ ]* 11.3 Test settings page functionality
    - Navigate to settings page and verify form loads
    - Modify settings and save
    - Verify settings persist across app restarts
    - Test validation for invalid inputs
    - _Requirements: 8.1, 8.6, 8.7, 8.8_

  - [ ]* 11.4 Test error handling
    - Test behavior when models directory is missing
    - Test behavior when microphone permission is denied
    - Verify error messages display correctly
    - Test invalid settings (non-existent paths, invalid thread count)
    - _Requirements: 4.3, 7.3, 8.6_
