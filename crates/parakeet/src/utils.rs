#[cfg(test)]
use crate::error::{Error, Result};
#[cfg(test)]
use hound::{WavReader, WavSpec};

#[cfg(test)]
use std::path::Path;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessorConfig {
    pub feature_size: usize,
    pub hop_length: usize,
    pub n_fft: usize,
    pub padding_side: String,
    pub padding_value: f32,
    pub preemphasis: f32,
    pub return_attention_mask: bool,
    pub sampling_rate: usize,
    pub win_length: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub architectures: Vec<String>,
    pub vocab_size: usize,
    pub pad_token_id: usize,
}

impl Default for PreprocessorConfig {
    fn default() -> Self {
        Self {
            feature_size: 128,
            hop_length: 160,
            n_fft: 512,
            padding_side: "right".to_string(),
            padding_value: 0.0,
            preemphasis: 0.97,
            return_attention_mask: true,
            sampling_rate: 16000,
            win_length: 400,
        }
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            architectures: vec!["ParakeetForCTC".to_string()],
            vocab_size: 1025,
            pad_token_id: 1024,
        }
    }
}

#[cfg(test)]
pub fn load_audio<P: AsRef<Path>>(path: P) -> Result<(Vec<f32>, WavSpec)> {
    let mut reader = WavReader::open(path)?;
    let spec = reader.spec();

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| Error::Audio(format!("Failed to read float samples: {e}")))?,
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .map(|s| s.map(|s| s as f32 / 32768.0))
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| Error::Audio(format!("Failed to read int samples: {e}")))?,
    };

    Ok((samples, spec))
}
