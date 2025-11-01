/// Vocabulary parser for the "vocab.txt" format used by TDT models.
///
/// This module provides a small, robust parser that maps integer token IDs to
/// their textual token representation. It is intentionally simple and tuned to
/// the text-file format produced/consumed by the TDT toolchain: each line in
/// the vocabulary file is expected to contain a token and its numeric ID,
/// separated by the first space on the line (i.e. "<token> <id>"). Lines that
/// do not contain exactly two parts are ignored.
///
/// Derived from: https://github.com/altunenes/parakeet-rs/blob/master/src/vocab.rs
///
/// Behavior and guarantees:
/// - The parser reads the file line-by-line using a buffered reader and returns
///   a Result containing the loaded Vocabulary or an Error::Config on failure.
/// - For each valid line the token string is inserted at the given numeric ID.
///   The internal storage (Vec<String>) is resized as needed to accommodate the
///   largest ID encountered; missing slots are filled with empty strings.
/// - The code recognizes the blank token if the token string equals "<blk>" or
///   "<blank>" and stores its ID in the vocabulary's blank-id field.
/// - If no blank token is explicitly found during parsing and the vocabulary is
///   non-empty, the blank-id defaults to the last token index (i.e. id_to_token.len() - 1).
/// - The parser maps common IO and parse errors to Error::Config with an
///   explanatory message (file open/read failures, invalid numeric IDs, ...).
///
/// Struct: Vocabulary
/// - id_to_token: Vec<String>
///     Public vector mapping token IDs (indices) to token strings. Accessing a
///     token by ID should be done through the provided helper method rather than
///     indexing directly when possible.
/// - _blank_id: usize
///     Internal index of the blank token (kept private/underscored to indicate
///     internal use). Stored for models or decoding routines that need a blank.
///
/// Method: from_file<P: AsRef<Path>>(path: P) -> Result<Self>
/// - Loads and parses a vocabulary file located at `path`.
/// - Accepts any path-like type that implements AsRef<Path>.
/// - Returns Ok(Vocabulary) on success, or Err(Error::Config) with a human
///   readable message on failure (IO or parse error).
///
/// Method: id_to_text(&self, id: usize) -> Option<&str>
/// - Safe helper to get the token string for a given numeric ID.
/// - Returns Some(&str) if the ID is in range, otherwise None.
///
/// Notes and edge cases:
/// - Duplicate IDs in the file will overwrite previous entries for that ID;
///   the last occurrence wins.
/// - Lines that do not split into exactly two components by the first space
///   are silently skipped (no panic). Malformed numeric IDs cause an error.
/// - The choice to default the blank ID to the last token if none was found is
///   a pragmatic fallback and mirrors behavior in common TDT toolchains.
use crate::error::{Error, Result};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Vocabulary parser for vocab.txt format used by TDT models
#[derive(Debug, Clone)]
pub struct Vocabulary {
    pub id_to_token: Vec<String>,
    pub _blank_id: usize,
}

impl Vocabulary {
    /// Load vocabulary from vocab.txt file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path.as_ref())
            .map_err(|e| Error::Config(format!("Failed to open vocab file: {}", e)))?;

        let reader = BufReader::new(file);
        let mut id_to_token = Vec::new();
        let mut blank_id = 0;

        for line in reader.lines() {
            let line =
                line.map_err(|e| Error::Config(format!("Failed to read vocab file: {}", e)))?;

            let parts: Vec<&str> = line.splitn(2, ' ').collect();
            if parts.len() == 2 {
                let token = parts[0].to_string();
                let id: usize = parts[1]
                    .parse()
                    .map_err(|e| Error::Config(format!("Invalid token ID in vocab: {}", e)))?;

                if id >= id_to_token.len() {
                    id_to_token.resize(id + 1, String::new());
                }
                id_to_token[id] = token.clone();

                // Track blank token
                if token == "<blk>" || token == "<blank>" {
                    blank_id = id;
                }
            }
        }

        // Default to last token if no blank found
        if blank_id == 0 && !id_to_token.is_empty() {
            blank_id = id_to_token.len() - 1;
        }

        Ok(Self {
            id_to_token,
            _blank_id: blank_id,
        })
    }

    /// Get token by ID
    pub fn id_to_text(&self, id: usize) -> Option<&str> {
        self.id_to_token.get(id).map(|s| s.as_str())
    }
}
