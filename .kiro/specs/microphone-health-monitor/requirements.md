# Requirements Document

## Introduction

This feature adds microphone health monitoring to the parakeet-mic application to detect and alert users when the microphone is not functioning as expected. On macOS, the operating system can silently block microphone access, leaving users unaware that no audio is being captured. This feature will monitor audio input levels and provide debug messages when no sound is detected after a configurable timeout period.

## Glossary

- **Parakeet-Mic**: The real-time speech transcription application that captures audio from a microphone
- **Audio Health Monitor**: A component that tracks audio input levels and detects silence periods
- **Silent Blocking**: When the operating system blocks microphone access without providing explicit error messages
- **Audio Level Threshold**: The minimum audio amplitude considered as "sound" versus silence
- **Silence Timeout**: The duration of continuous silence after which a warning is triggered
- **Recording Thread**: The thread responsible for capturing audio from the microphone using CPAL
- **Audio Sample**: A single floating-point value representing audio amplitude at a point in time

## Requirements

### Requirement 1

**User Story:** As a user running parakeet-mic on macOS, I want to be notified when my microphone is not capturing any audio, so that I can troubleshoot permission issues or hardware problems.

#### Acceptance Criteria

1. WHEN the Recording Thread captures audio samples, THE Audio Health Monitor SHALL track the maximum absolute amplitude of samples within each monitoring window
2. WHEN the maximum amplitude remains below the Audio Level Threshold for the duration of the Silence Timeout, THE Audio Health Monitor SHALL emit a debug warning message to the user
3. WHEN audio above the Audio Level Threshold is detected, THE Audio Health Monitor SHALL reset the silence timer
4. WHERE the user has not configured a custom threshold, THE Audio Health Monitor SHALL use a default Audio Level Threshold of 0.001 for detecting meaningful audio
5. WHERE the user has not configured a custom timeout, THE Audio Health Monitor SHALL use a default Silence Timeout of 10 seconds

### Requirement 2

**User Story:** As a developer debugging audio capture issues, I want to see diagnostic information about audio levels, so that I can understand whether the microphone is working correctly.

#### Acceptance Criteria

1. WHEN the Silence Timeout is reached, THE Audio Health Monitor SHALL display a message indicating that no audio has been detected and suggesting possible causes
2. WHEN the warning message is displayed, THE Audio Health Monitor SHALL include information about checking system microphone permissions
3. WHEN the warning message is displayed, THE Audio Health Monitor SHALL include information about verifying the correct audio device is selected
4. WHEN audio is detected after a warning has been issued, THE Audio Health Monitor SHALL display a confirmation message that audio input has resumed

### Requirement 3

**User Story:** As a user, I want to configure the sensitivity of the microphone health monitoring, so that I can adjust it based on my environment and microphone characteristics.

#### Acceptance Criteria

1. WHERE the user provides a custom silence timeout value, THE Parakeet-Mic SHALL accept a command-line argument to configure the Silence Timeout duration
2. WHERE the user provides a custom audio threshold value, THE Parakeet-Mic SHALL accept a command-line argument to configure the Audio Level Threshold
3. WHEN invalid configuration values are provided, THE Parakeet-Mic SHALL display an error message and use default values
4. WHEN the application starts, THE Parakeet-Mic SHALL display the configured Audio Level Threshold and Silence Timeout values in the startup information
