use anyhow::Result;
use hound::{WavSpec, WavWriter};
use parakeet::streaming::TokenResult;
use serde::{Deserialize, Serialize};
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};
use tracing;

#[derive(Clone, Debug)]
pub enum OutputFormat {
    Txt,
    Json,
    Csv,
}

impl std::str::FromStr for OutputFormat {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "txt" => Ok(OutputFormat::Txt),
            "json" => Ok(OutputFormat::Json),
            "csv" => Ok(OutputFormat::Csv),
            _ => Err(anyhow::anyhow!(
                "Unsupported output format: {}. Use txt, json, or csv",
                s
            )),
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct TokenOutput {
    pub timestamp: usize,
    pub token_id: i32,
    pub text: Option<String>,
    pub time_seconds: f32,
    pub session_time: f32,
}

pub struct OutputWriter {
    writer: Option<Arc<Mutex<BufWriter<std::fs::File>>>>,
    format: OutputFormat,
    start_time: SystemTime,
    json_tokens: Arc<Mutex<Vec<TokenOutput>>>,
}

impl OutputWriter {
    pub fn new(output_path: Option<String>, format: OutputFormat, append: bool) -> Result<Self> {
        let writer = if let Some(ref path) = output_path {
            tracing::debug!(
                path = %path,
                format = ?format,
                append = append,
                "Output writer initialized"
            );

            let file = OpenOptions::new()
                .create(true)
                .write(true)
                .append(append)
                .truncate(!append)
                .open(&path)?;

            let mut buf_writer = BufWriter::new(file);

            // Write headers for CSV format
            if matches!(format, OutputFormat::Csv) && !append {
                writeln!(buf_writer, "timestamp,token_id,text,time_seconds")?;
            }

            Some(Arc::new(Mutex::new(buf_writer)))
        } else {
            None
        };

        Ok(Self {
            writer,
            format,
            start_time: SystemTime::now(),
            json_tokens: Arc::new(Mutex::new(Vec::new())),
        })
    }

    pub fn write_tokens(&self, tokens: &[TokenResult]) -> Result<()> {
        if let Some(writer) = &self.writer {
            tracing::trace!(token_count = tokens.len(), "Writing tokens to file");
            for token in tokens {
                let time_seconds = token.timestamp as f32 / 16000.0;
                let session_time = self
                    .start_time
                    .elapsed()
                    .unwrap_or(Duration::ZERO)
                    .as_secs_f32();

                match self.format {
                    OutputFormat::Txt => {
                        let mut writer = writer.lock().unwrap();
                        if let Some(ref text) = token.text {
                            writeln!(writer, "[{:.3}s] {}", time_seconds, text)?;
                        } else {
                            writeln!(writer, "[{:.3}s] Token {}", time_seconds, token.token_id)?;
                        }
                        writer.flush()?;
                    }
                    OutputFormat::Json => {
                        // Store tokens for batch writing at the end
                        let token_output = TokenOutput {
                            timestamp: token.timestamp,
                            token_id: token.token_id,
                            text: token.text.clone(),
                            time_seconds,
                            session_time,
                        };
                        self.json_tokens.lock().unwrap().push(token_output);
                    }
                    OutputFormat::Csv => {
                        let mut writer = writer.lock().unwrap();
                        let text_field = token.text.as_deref().unwrap_or("");
                        writeln!(
                            writer,
                            "{},{},\"{}\",{:.3}",
                            token.timestamp, token.token_id, text_field, time_seconds
                        )?;
                        writer.flush()?;
                    }
                }
            }
        }

        Ok(())
    }

    pub fn finalize(&self) -> Result<()> {
        if let Some(writer) = &self.writer {
            tracing::debug!("Finalizing output file");
            let mut writer = writer.lock().unwrap();

            // Write JSON data for JSON format
            if matches!(self.format, OutputFormat::Json) {
                let tokens = self.json_tokens.lock().unwrap();
                let json_output = serde_json::to_string_pretty(&*tokens)?;
                write!(writer, "{}", json_output)?;
            }

            writer.flush()?;
            tracing::info!("Output file finalized successfully");
        }

        Ok(())
    }
}

#[derive(Clone)]
pub struct AudioWriter {
    writer: Option<Arc<Mutex<WavWriter<std::io::BufWriter<std::fs::File>>>>>,
}

impl AudioWriter {
    pub fn new(audio_path: Option<String>, sample_rate: u32) -> Result<Self> {
        let writer = if let Some(ref path) = audio_path {
            tracing::debug!(
                path = %path,
                sample_rate = sample_rate,
                "Audio writer initialized"
            );

            let spec = WavSpec {
                channels: 1,
                sample_rate,
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            };

            let writer = WavWriter::create(&path, spec)?;
            Some(Arc::new(Mutex::new(writer)))
        } else {
            None
        };

        Ok(Self { writer })
    }

    pub fn write_sample(&self, sample: f32) -> Result<()> {
        if let Some(writer) = &self.writer {
            let mut writer = writer.lock().unwrap();
            writer.write_sample(sample)?;
        }
        Ok(())
    }

    pub fn finalize(self) -> Result<()> {
        if let Some(writer) = self.writer {
            tracing::debug!("Finalizing audio file");
            let writer = Arc::try_unwrap(writer)
                .map_err(|_| anyhow::anyhow!("Failed to unwrap Arc for audio writer"))?
                .into_inner()
                .unwrap();
            writer.finalize()?;
            tracing::info!("Audio file finalized successfully");
        }
        Ok(())
    }
}
