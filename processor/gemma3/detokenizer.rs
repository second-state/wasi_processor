use crate::gemma3::processor::Gemma3Processor;

/// Detokenizer for converting token sequences back to text
/// Handles special Unicode space characters and provides trimming options
pub struct Detokenizer {
    trim_space: bool,
    unflushed: String,
    text: String,
}

impl Detokenizer {
    /// Create a new Detokenizer instance
    ///
    /// # Arguments
    /// * `_processor` - Reference to the Gemma3Processor (currently unused but kept for API consistency)
    /// * `trim_space` - Whether to trim leading spaces from the output
    pub fn new(_processor: &Gemma3Processor, trim_space: bool) -> Self {
        Detokenizer {
            trim_space,
            unflushed: String::new(),
            text: String::new(),
        }
    }

    /// Add a single token to the detokenizer
    ///
    /// # Arguments
    /// * `token` - Token ID to add
    /// * `processor` - Reference to the processor for token decoding
    pub fn add_token(&mut self, token: usize, processor: &Gemma3Processor) {
        // Use tokenizer's decode method to convert token ID to string
        let token_str = processor.tokenizer.decode(&[token as u32]);

        // Check if token starts with Unicode space (\u2581)
        if token_str.starts_with("\u{2581}") {
            let replaced = self.unflushed.replace("\u{2581}", " ");

            if !self.text.is_empty() || !self.trim_space {
                self.text.push_str(&replaced);
            } else {
                // If trim_space is true and text is empty, remove leading space
                self.text = Self::remove_leading_space(&replaced);
            }

            self.unflushed = token_str;
        } else {
            self.unflushed.push_str(&token_str);
        }
    }

    /// Get the final decoded text
    ///
    /// # Returns
    /// * `String` - The complete decoded text
    pub fn get_text(&self) -> String {
        let mut result = self.text.clone();
        let final_part = self.unflushed.replace("\u{2581}", " ");

        if !self.text.is_empty() || !self.trim_space {
            result.push_str(&final_part);
        } else {
            result = Self::remove_leading_space(&final_part);
        }

        result
    }

    /// Remove leading space from a string
    ///
    /// # Arguments
    /// * `s` - Input string
    ///
    /// # Returns
    /// * `String` - String with leading space removed
    fn remove_leading_space(s: &str) -> String {
        if s.starts_with(' ') {
            s[1..].to_string()
        } else {
            s.to_string()
        }
    }

    /// Decode a sequence of tokens
    ///
    /// # Arguments
    /// * `tokens` - Slice of token IDs to decode
    /// * `processor` - Reference to the processor for token decoding
    pub fn decode_tokens(&mut self, tokens: &[usize], processor: &Gemma3Processor) {
        for &token in tokens {
            self.add_token(token, processor);
        }
    }
}

/// Convenience function to decode tokens directly without creating a Detokenizer instance
///
/// # Arguments
/// * `tokens` - Slice of token IDs to decode
/// * `processor` - Reference to the processor for token decoding
/// * `trim_space` - Whether to trim leading spaces
///
/// # Returns
/// * `String` - Decoded text
pub fn decode(tokens: &[usize], processor: &Gemma3Processor, trim_space: bool) -> String {
    // For simple cases, directly use tokenizer's decode method
    let token_ids: Vec<u32> = tokens.iter().map(|&t| t as u32).collect();
    let raw_text = processor.tokenizer.decode(&token_ids);

    // Handle space trimming
    if trim_space && raw_text.starts_with(' ') {
        raw_text[1..].to_string()
    } else {
        raw_text
    }
}
