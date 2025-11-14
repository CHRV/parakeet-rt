# Requirements Document

## Introduction

This feature implements a Tauri desktop application with a Svelte frontend that provides real-time speech transcription using the Parakeet TDT model. The application replicates the functionality of the existing parakeet-ws WebSocket implementation but as a native desktop application with integrated audio capture and transcription processing.

## Glossary

- **Tauri App**: The desktop application framework combining Rust backend with Svelte frontend
- **Parakeet TDT**: The speech recognition model used for transcription
- **Streaming Engine**: The component that processes audio chunks in real-time using the Parakeet model
- **Audio Capture**: The system component that captures microphone input using the cpal library
- **cpal**: Cross-platform audio library used for microphone input capture
- **Transcript Display**: The UI component showing transcribed text in real-time
- **Audio Visualizer**: The UI component displaying audio waveform during recording
- **Settings Page**: The UI component for configuring transcription parameters
- **ORT**: ONNX Runtime, the inference engine used by the Parakeet model
- **Model Path**: The file system location of the Parakeet model files

## Requirements

### Requirement 1

**User Story:** As a user, I want to start and stop audio recording with a single button click, so that I can control when transcription occurs

#### Acceptance Criteria

1. WHEN the user clicks the "Start" button, THE Tauri App SHALL initialize the Audio Capture using the cpal library and begin streaming audio data to the Streaming Engine
2. WHILE recording is active, THE Tauri App SHALL set the "Start" button to disabled state and set the "Stop" button to enabled state
3. WHEN the user clicks the "Stop" button, THE Tauri App SHALL terminate the Audio Capture and stop the Streaming Engine
4. WHEN recording stops, THE Tauri App SHALL set the "Start" button to enabled state and set the "Stop" button to disabled state

### Requirement 2

**User Story:** As a user, I want to see transcribed text appear in real-time as I speak, so that I can verify the transcription is working correctly

#### Acceptance Criteria

1. WHEN the Streaming Engine produces a token, THE Tauri App SHALL convert the token to text using the vocabulary file
2. WHEN transcription text is generated, THE Transcript Display SHALL append the text to the existing transcript content
3. THE Transcript Display SHALL format transcription text with proper spacing between words
4. WHEN sentence-ending punctuation is detected in the text, THE Transcript Display SHALL insert a line break after the punctuation
5. WHEN new text is appended to the transcript, THE Transcript Display SHALL scroll the view to show the most recent text

### Requirement 3

**User Story:** As a user, I want to see a visual representation of my audio input, so that I can confirm my microphone is working

#### Acceptance Criteria

1. WHILE recording is active, THE Audio Visualizer SHALL display a waveform representation of the audio input
2. WHILE recording is active, THE Audio Visualizer SHALL update the waveform display at a rate of 30 frames per second or greater
3. WHEN recording is not active, THE Audio Visualizer SHALL display an empty state

### Requirement 4

**User Story:** As a user, I want to see the current status of the application, so that I know whether it's ready, recording, or encountering issues

#### Acceptance Criteria

1. WHEN the application starts, THE Tauri App SHALL display a "Ready" status message
2. WHEN recording begins, THE Tauri App SHALL display a "Recording" status message with a visual indicator
3. IF an error occurs, THEN THE Tauri App SHALL display an error status message with a description of the error condition
4. WHEN recording stops, THE Tauri App SHALL display a "Ready" status message

### Requirement 5

**User Story:** As a user, I want the application to use the same simple, clean interface as the WebSocket version, so that I have a familiar and distraction-free experience

#### Acceptance Criteria

1. THE Tauri App SHALL display a full-page transcript area with a paper-like background color
2. THE Tauri App SHALL display a floating control panel positioned in the bottom-right corner of the window
3. THE Transcript Display SHALL render text using a serif font at 20 pixel size with 1.8 line height spacing
4. THE Tauri App SHALL display within the control panel the application title, status indicator, Audio Visualizer, and control buttons
5. THE Tauri App SHALL apply color values matching the color scheme used in the WebSocket version

### Requirement 6

**User Story:** As a developer, I want the Rust backend to handle all audio processing and transcription, so that the frontend remains simple and focused on UI

#### Acceptance Criteria

1. THE Tauri App SHALL implement Tauri commands for starting recording and stopping recording
2. THE Tauri App SHALL use the cpal library for cross-platform microphone input capture
3. THE Tauri App SHALL use the StreamingParakeetTDT implementation for audio processing
4. WHEN new transcription text is available, THE Tauri App SHALL emit an event to the frontend containing the text
5. WHEN audio level data is calculated, THE Tauri App SHALL emit an event to the frontend containing the audio level value for visualization
6. WHEN an error condition occurs in the Rust backend, THE Tauri App SHALL emit an error event to the frontend containing the error information

### Requirement 7

**User Story:** As a user, I want the application to load the Parakeet models on startup, so that transcription is ready when I need it

#### Acceptance Criteria

1. WHEN the application starts, THE Tauri App SHALL load the Parakeet TDT model files from the models directory
2. WHEN the application starts, THE Tauri App SHALL load the vocabulary file from the models directory
3. IF model loading fails, THEN THE Tauri App SHALL display an error message to the user describing the failure
4. WHEN model files are loaded successfully, THE Tauri App SHALL enable the "Start" button

### Requirement 8

**User Story:** As a user, I want to configure transcription settings like model paths and performance parameters, so that I can customize the application for my system

#### Acceptance Criteria

1. THE Tauri App SHALL provide a Settings Page accessible from the main interface
2. THE Settings Page SHALL provide an input field allowing the user to specify the encoder model file path
3. THE Settings Page SHALL provide an input field allowing the user to specify the decoder model file path
4. THE Settings Page SHALL provide an input field allowing the user to specify the vocabulary file path
5. THE Settings Page SHALL provide an input field allowing the user to configure the ORT thread count with values between 1 and 16 inclusive
6. WHEN the user modifies a setting value, THE Tauri App SHALL validate the input value before accepting the change
7. WHEN the user saves settings, THE Tauri App SHALL write the configuration data to persistent storage on disk
8. WHEN the application starts, THE Tauri App SHALL read saved settings from persistent storage on disk
9. IF no saved settings exist on disk, THEN THE Tauri App SHALL use default values for all configuration parameters
