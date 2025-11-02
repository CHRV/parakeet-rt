pub mod model;
pub mod streaming;
pub mod utils;

// Re-export the main streaming VAD components for easier access
pub use streaming::{SpeechSegment, StreamingVad, VadEvent};
