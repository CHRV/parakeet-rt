use crate::error::Result;
use crate::vocab::Vocabulary;
use tracing;

// Token with its timestamp information
// start and end are in seconds
#[derive(Debug, Clone)]
pub struct TimedToken {
    pub text: String,
    pub start: f32,
    pub end: f32,
}

#[derive(Debug, Clone)]
pub struct TranscriptionResult {
    pub text: String,
    pub tokens: Vec<TimedToken>,
}
/// TDT greedy decoder for Parakeet TDT models
#[derive(Debug)]
pub struct ParakeetTDTDecoder {
    vocab: Vocabulary,
}

impl ParakeetTDTDecoder {
    /// Load decoder from vocab file
    pub fn from_vocab(vocab: Vocabulary) -> Self {
        Self { vocab }
    }

    /// Decode tokens with timestamps
    /// For TDT models, greedy decoding is done in the model, here we just convert to text
    #[tracing::instrument(skip(self, tokens, frame_indices, _durations))]
    pub fn decode_with_timestamps(
        &self,
        tokens: &[usize],
        frame_indices: &[usize],
        _durations: &[usize],
        hop_length: usize,
        sample_rate: usize,
    ) -> Result<TranscriptionResult> {
        tracing::debug!(
            token_count = tokens.len(),
            "Decoding tokens with timestamps"
        );
        let mut result_tokens = Vec::new();
        let mut full_text = String::new();
        // TDT encoder does 8x subsampling
        let encoder_stride = 8;

        for (i, &token_id) in tokens.iter().enumerate() {
            if let Some(token_text) = self.vocab.decode_token(token_id as i32) {
                let frame = frame_indices[i];
                let start = (frame * encoder_stride * hop_length) as f32 / sample_rate as f32;
                let end = if i + 1 < frame_indices.len() {
                    (frame_indices[i + 1] * encoder_stride * hop_length) as f32 / sample_rate as f32
                } else {
                    start + 0.01
                };

                // Handle SentencePiece format (▁ prefix for word start)
                let display_text = token_text.replace('▁', " ");

                // Skip special tokens
                if !(token_text.starts_with('<')
                    && token_text.ends_with('>')
                    && token_text != "<unk>")
                {
                    full_text.push_str(&display_text);

                    result_tokens.push(TimedToken {
                        text: display_text,
                        start,
                        end,
                    });
                }
            }
        }

        let final_text = full_text.trim().to_string();
        tracing::debug!(transcription_length = final_text.len(), "Decoding complete");

        Ok(TranscriptionResult {
            text: final_text,
            tokens: result_tokens,
        })
    }
}
