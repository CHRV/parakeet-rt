pub mod silero;
pub mod streaming_vad;
pub mod utils;

// Re-export the main streaming VAD components for easier access
pub use streaming_vad::{SpeechSegment, StreamingVad, VadEvent};
