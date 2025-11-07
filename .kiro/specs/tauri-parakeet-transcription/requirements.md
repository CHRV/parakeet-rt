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

## Requirements

### Requirement 1

**User Story:** As a user, I want to start and stop audio recording with a single button click, so that I can control when transcription occurs

#### Acceptance Criteria

1. WHEN the user clicks the "Start" button, THE Tauri App SHALL initialize the Audio Capture using cpal and begin streaming audio to the Streaming Engine
2. WHILE recording is active, THE Tauri App SHALL disable the "Start" button and enable the "Stop" button
3. WHEN the user clicks the "Stop" button, THE Tauri App SHALL terminate the Audio Capture and stop the Streaming Engine
4. WHEN recording stops, THE Tauri App SHALL re-enable the "Start" button and disable the "Stop" button

### Requirement 2

**User Story:** As a user, I want to see transcribed text appear in real-time as I speak, so that I can verify the transcription is working correctly

#### Acceptance Criteria

1. WHEN the Streaming Engine produces a token, THE Tauri App SHALL convert the token to text using the vocabulary
2. WHEN text is generated, THE Transcript Display SHALL append the text to the existing transcript
3. THE Transcript Display SHALL format text with proper spacing between words
4. WHEN a sentence-ending punctuation is detected, THE Transcript Display SHALL create a line break
5. THE Transcript Display SHALL automatically scroll to show the most recent text

### Requirement 3

**User Story:** As a user, I want to see a visual representation of my audio input, so that I can confirm my microphone is working

#### Acceptance Criteria

1. WHILE recording is active, THE Audio Visualizer SHALL display a waveform representation of the audio input
2. THE Audio Visualizer SHALL update at a minimum rate of 30 frames per second
3. WHEN recording is not active, THE Audio Visualizer SHALL display an empty state

### Requirement 4

**User Story:** As a user, I want to see the current status of the application, so that I know whether it's ready, recording, or encountering issues

#### Acceptance Criteria

1. WHEN the application starts, THE Tauri App SHALL display a "Ready" status
2. WHEN recording begins, THE Tauri App SHALL display a "Recording" status with a visual indicator
3. WHEN an error occurs, THE Tauri App SHALL display an error status with a descriptive message
4. WHEN recording stops, THE Tauri App SHALL return to "Ready" status

### Requirement 5

**User Story:** As a user, I want the application to use the same simple, clean interface as the WebSocket version, so that I have a familiar and distraction-free experience

#### Acceptance Criteria

1. THE Tauri App SHALL display a full-page transcript area with a paper-like background
2. THE Tauri App SHALL display a floating control panel in the bottom-right corner
3. THE Transcript Display SHALL use a serif font at 20px size with 1.8 line height
4. THE control panel SHALL include the application title, status indicator, Audio Visualizer, and control buttons
5. THE Tauri App SHALL use the same color scheme as the WebSocket version

### Requirement 6

**User Story:** As a developer, I want the Rust backend to handle all audio processing and transcription, so that the frontend remains simple and focused on UI

#### Acceptance Criteria

1. THE Tauri App SHALL implement Tauri commands for starting and stopping recording
2. THE Tauri App SHALL use cpal for cross-platform microphone input capture
3. THE Tauri App SHALL use the existing StreamingParakeetTDT implementation for audio processing
4. THE Tauri App SHALL emit events to the frontend when new transcription text is available
5. THE Tauri App SHALL provide audio level data to the frontend for visualization
6. THE Tauri App SHALL handle all error conditions in the Rust backend and communicate them to the frontend

### Requirement 7

**User Story:** As a user, I want the application to load the Parakeet models on startup, so that transcription is ready when I need it

#### Acceptance Criteria

1. WHEN the application starts, THE Tauri App SHALL load the Parakeet TDT model from the models directory
2. WHEN the application starts, THE Tauri App SHALL load the vocabulary file
3. IF model loading fails, THE Tauri App SHALL display an error message to the user
4. WHEN models are loaded successfully, THE Tauri App SHALL enable the "Start" button
