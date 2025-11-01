use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use crate::error::{Error, Result};

/// Vocabulary for decoding token IDs to text
pub struct Vocabulary {
    /// Map from token ID to token text
    id_to_token: HashMap<i32, String>,
    /// Map from token text to token ID
    token_to_id: HashMap<String, i32>,
}

impl Vocabulary {
    /// Load vocabulary from a vocab.txt file
    pub fn from_file<P: AsRef<Path>>(vocab_path: P) -> Result<Self> {
        let file = File::open(&vocab_path)
            .map_err(|e| Error::Config(format!("Failed to open vocab file {}: {}", vocab_path.as_ref().display(), e)))?;

        let reader = BufReader::new(file);
        let mut id_to_token = HashMap::new();
        let mut token_to_id = HashMap::new();

        for line in reader.lines() {
            let line = line.map_err(|e| Error::Config(format!("Failed to read vocab line: {}", e)))?;
            let line = line.trim();

            if line.is_empty() {
                continue;
            }

            // Parse line format: "token_text token_id"
            let parts: Vec<&str> = line.rsplitn(2, ' ').collect();
            if parts.len() != 2 {
                continue; // Skip malformed lines
            }

            let token_text = parts[1].to_string();
            let token_id = parts[0].parse::<i32>()
                .map_err(|e| Error::Config(format!("Invalid token ID '{}': {}", parts[0], e)))?;

            id_to_token.insert(token_id, token_text.clone());
            token_to_id.insert(token_text, token_id);
        }

        Ok(Self {
            id_to_token,
            token_to_id,
        })
    }

    /// Decode a token ID to its text representation
    pub fn decode_token(&self, token_id: i32) -> Option<&str> {
        self.id_to_token.get(&token_id).map(|s| s.as_str())
    }

    /// Encode a token text to its ID
    pub fn encode_token(&self, token_text: &str) -> Option<i32> {
        self.token_to_id.get(token_text).copied()
    }

    /// Decode a sequence of token IDs to text
    pub fn decode_tokens(&self, token_ids: &[i32]) -> String {
        let mut result = String::new();

        for &token_id in token_ids {
            if let Some(token_text) = self.decode_token(token_id) {
                // Handle special tokens and formatting
                if token_text.starts_with("▁") {
                    // SentencePiece space prefix - replace with actual space
                    if !result.is_empty() {
                        result.push(' ');
                    }
                    result.push_str(&token_text[3..]); // Skip the ▁ character (3 bytes in UTF-8)
                } else if token_text.starts_with('<') && token_text.ends_with('>') {
                    // Special tokens like <|endoftext|> - skip them for clean output
                    continue;
                } else {
                    // Regular token - append directly
                    result.push_str(token_text);
                }
            }
        }

        result
    }

    /// Get the vocabulary size
    pub fn size(&self) -> usize {
        self.id_to_token.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_vocab_loading() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "hello 100").unwrap();
        writeln!(temp_file, "▁world 101").unwrap();
        writeln!(temp_file, "<|endoftext|> 102").unwrap();

        let vocab = Vocabulary::from_file(temp_file.path()).unwrap();

        assert_eq!(vocab.decode_token(100), Some("hello"));
        assert_eq!(vocab.decode_token(101), Some("▁world"));
        assert_eq!(vocab.decode_token(999), None);

        assert_eq!(vocab.encode_token("hello"), Some(100));
        assert_eq!(vocab.encode_token("▁world"), Some(101));
        assert_eq!(vocab.encode_token("nonexistent"), None);
    }

    #[test]
    fn test_token_decoding() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "hello 100").unwrap();
        writeln!(temp_file, "▁world 101").unwrap();
        writeln!(temp_file, "! 102").unwrap();
        writeln!(temp_file, "<|endoftext|> 103").unwrap();

        let vocab = Vocabulary::from_file(temp_file.path()).unwrap();

        let tokens = vec![100, 101, 102, 103];
        let decoded = vocab.decode_tokens(&tokens);

        // Should be "hello world!" (special tokens skipped, ▁ converted to space)
        assert_eq!(decoded, "hello world!");
    }
}