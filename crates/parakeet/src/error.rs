use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error)]
pub enum Error {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("ONNX Runtime error: {0}")]
    Ort(#[from] ort::Error),

    #[error("Audio processing error: {0}")]
    Audio(String),

    #[error("Model error: {0}")]
    Model(String),

    #[error("Tokenizer error: {0}")]
    Tokenizer(String),

    #[error("Config error: {0}")]
    Config(String),
}

impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self {
        Error::Config(e.to_string())
    }
}

#[cfg(any(feature = "audio", test))]
impl From<hound::Error> for Error {
    fn from(e: hound::Error) -> Self {
        Error::Audio(e.to_string())
    }
}
