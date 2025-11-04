use std::time::{Duration, Instant};
use tracing;

/// Status of the audio health monitoring
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthStatus {
    /// Audio is being received normally
    Healthy,
    /// No audio detected, warning should be issued
    SilenceDetected,
    /// Audio resumed after previous silence warning
    AudioResumed,
    /// No change in status
    NoChange,
}

/// Minimum time between status messages to prevent console spam
const MESSAGE_THROTTLE_DURATION: Duration = Duration::from_secs(5);

/// Monitors audio input health by tracking silence periods
pub struct AudioHealthMonitor {
    /// Threshold below which audio is considered silence
    pub threshold: f32,
    /// Duration of silence before triggering a warning (in seconds)
    pub timeout_secs: f32,
    /// Sample rate for time calculations
    sample_rate: u32,
    /// Counter for samples below threshold
    silence_sample_count: usize,
    /// Total samples needed to trigger timeout
    timeout_sample_count: usize,
    /// Whether a warning has been issued
    warning_issued: bool,
    /// Last time a status message was printed
    last_status_time: Instant,
}

impl AudioHealthMonitor {
    /// Create a new health monitor with specified parameters
    pub fn new(threshold: f32, timeout_secs: f32, sample_rate: u32) -> Self {
        let timeout_sample_count = (timeout_secs * sample_rate as f32) as usize;

        tracing::debug!(
            threshold = threshold,
            timeout_secs = timeout_secs,
            sample_rate = sample_rate,
            "Health monitor initialized"
        );

        Self {
            threshold,
            timeout_secs,
            sample_rate,
            silence_sample_count: 0,
            timeout_sample_count,
            warning_issued: false,
            last_status_time: Instant::now(),
        }
    }

    /// Process a single audio sample and check health status
    /// Returns the current health status based on the sample
    pub fn process_sample(&mut self, sample: f32) -> HealthStatus {
        let amplitude = sample.abs();

        if amplitude > self.threshold {
            // Audio detected - reset silence counter
            self.silence_sample_count = 0;

            // If we previously issued a warning, report that audio has resumed
            if self.warning_issued {
                self.warning_issued = false;
                tracing::info!("Audio input resumed");
                return HealthStatus::AudioResumed;
            }

            return HealthStatus::Healthy;
        } else {
            // Sample is below threshold - increment silence counter
            self.silence_sample_count += 1;

            // Check if we've reached the timeout threshold
            if self.silence_sample_count >= self.timeout_sample_count && !self.warning_issued {
                self.warning_issued = true;
                tracing::warn!(
                    silence_duration_secs = self.timeout_secs,
                    threshold = self.threshold,
                    "Silence detected - no audio input"
                );
                return HealthStatus::SilenceDetected;
            }

            return HealthStatus::NoChange;
        }
    }

    /// Check if enough time has passed since the last status message
    /// Returns true if a message should be displayed
    pub fn should_display_message(&mut self) -> bool {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_status_time);

        if elapsed >= MESSAGE_THROTTLE_DURATION {
            self.last_status_time = now;
            true
        } else {
            false
        }
    }

    /// Get default threshold value
    pub const fn default_threshold() -> f32 {
        0.001
    }

    /// Get default timeout value
    pub const fn default_timeout() -> f32 {
        10.0
    }
}

/// Format a silence warning message with troubleshooting tips
/// Includes platform-specific guidance for macOS
pub fn format_silence_warning(timeout_secs: f32, threshold: f32) -> String {
    let mut message = String::new();

    message.push_str(&format!(
        "\n⚠ WARNING: No audio detected for {:.1} seconds\n",
        timeout_secs
    ));
    message.push_str("  Possible causes:\n");
    message.push_str("  • Microphone permissions may be blocked by the operating system\n");
    message.push_str("  • Wrong audio device selected (use --list-devices to see options)\n");
    message.push_str("  • Microphone hardware not connected or muted\n");
    message.push_str(&format!(
        "  • Ambient noise level below detection threshold (--audio-threshold {})\n",
        threshold
    ));

    // Platform-specific guidance for macOS
    if cfg!(target_os = "macos") {
        message.push_str(
            "\n  Tip: On macOS, check System Settings → Privacy & Security → Microphone\n",
        );
    }

    message
}

/// Format an audio resume confirmation message
pub fn format_audio_resumed() -> String {
    "✓ Audio input detected - microphone is working\n".to_string()
}
