#![allow(dead_code)]
use tokenizers::{Tokenizer as HFTokenizer, EncodeInput};
use serde::Deserialize;
use std::fs;
use std::collections::HashMap;

/// Gemma3 Tokenizer for handling text tokenization with image support
#[derive(Debug, Clone)]
pub struct Gemma3Tokenizer {
    tokenizer: HFTokenizer,
    add_bos_token: bool,
    add_eos_token: bool,
    bos_token_id: u32,
    eos_token_id: u32,
    pub pad_token_id: u32,
    unk_token_id: u32,
    pub boi_token: String,     // Beginning of image marker
    pub eoi_token: String,     // End of image marker
    pub image_token: String,   // Image token
    pub image_token_id: u32,   // Image token ID
}

/// Configuration structure for tokenizer settings
#[derive(Deserialize)]
struct TokenizerConfig {
    add_bos_token: Option<bool>,
    add_eos_token: Option<bool>,
    bos_token_id: Option<u32>,
    eos_token_id: Option<u32>,
    pad_token_id: Option<u32>,
    unk_token_id: Option<u32>,
}

impl Gemma3Tokenizer {
    /// Load tokenizer from pretrained model directory
    /// 
    /// # Arguments
    /// * `model_dir` - Path to the model directory
    /// 
    /// # Returns
    /// * `Self` - Configured Gemma3Tokenizer instance
    pub fn from_pretrained(model_dir: &str) -> Self {
        let tokenizer_path = format!("{}/tokenizer.json", model_dir);
        let config_path = format!("{}/tokenizer_config.json", model_dir);

        let tokenizer = HFTokenizer::from_file(&tokenizer_path)
            .expect("Failed to load tokenizer.json");

        let config_str = fs::read_to_string(&config_path).expect("Failed to read tokenizer_config.json");
        let config: TokenizerConfig = serde_json::from_str(&config_str).expect("Failed to parse tokenizer_config.json");

        // Read image_token_id from config.json
        let model_config_path = format!("{}/config.json", model_dir);
        let model_config_str = fs::read_to_string(&model_config_path).expect("Failed to read config.json");
        let model_config: serde_json::Value = serde_json::from_str(&model_config_str).expect("Failed to parse config.json");
        let image_token_id = model_config["image_token_index"].as_u64().unwrap_or(262144) as u32;

        // Set image-related tokens
        let boi_token = "<start_of_image>".to_string();
        let eoi_token = "<end_of_image>".to_string();
        let image_token = "<image>".to_string();
        
        Gemma3Tokenizer {
            tokenizer,
            add_bos_token: config.add_bos_token.unwrap_or(true),
            add_eos_token: config.add_eos_token.unwrap_or(false),
            bos_token_id: config.bos_token_id.unwrap_or(2),
            eos_token_id: config.eos_token_id.unwrap_or(1),
            pad_token_id: config.pad_token_id.unwrap_or(0),
            unk_token_id: config.unk_token_id.unwrap_or(3),
            boi_token,
            eoi_token,
            image_token,
            image_token_id,
        }
    }

    /// Tokenize text with optional BOS/EOS tokens
    /// 
    /// # Arguments
    /// * `text` - Input text to tokenize
    /// 
    /// # Returns
    /// * `Vec<u32>` - Token sequence
    pub fn tokenize(&self, text: &str) -> Vec<u32> {
        let mut ids = vec![];
        
        if self.add_bos_token {
            ids.push(self.bos_token_id);
        }
        
        let encoding = self.tokenizer.encode(EncodeInput::Single(text.into()), false).unwrap();
        ids.extend(encoding.get_ids().iter().copied());
        
        if self.add_eos_token {
            ids.push(self.eos_token_id);
        }
        
        ids
    }

    /// Decode token sequence back to text
    /// 
    /// # Arguments
    /// * `ids` - Token sequence to decode
    /// 
    /// # Returns
    /// * `String` - Decoded text
    pub fn decode(&self, ids: &[u32]) -> String {
        self.tokenizer.decode(ids, true).unwrap()
    }

    /// Batch decode multiple token sequences
    /// 
    /// # Arguments
    /// * `batch_ids` - Batch of token sequences
    /// 
    /// # Returns
    /// * `Vec<String>` - Batch of decoded text strings
    pub fn batch_decode(&self, batch_ids: &[Vec<u32>]) -> Vec<String> {
        batch_ids.iter().map(|ids| self.decode(ids)).collect()
    }
    
    /// Get vocabulary size
    /// 
    /// # Returns
    /// * `usize` - Size of the vocabulary
    pub fn get_vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(false)
    }
    
    /// Get vocabulary mapping
    /// 
    /// # Returns
    /// * `HashMap<String, usize>` - Mapping from tokens to IDs
    pub fn get_vocab(&self) -> HashMap<String, usize> {
        let vocab = self.tokenizer.get_vocab(false);
        vocab.into_iter().map(|(token, id)| (token, id as usize)).collect()
    }
    
    /// Encode text without adding BOS/EOS tokens
    /// 
    /// # Arguments
    /// * `text` - Input text to encode
    /// 
    /// # Returns
    /// * `Vec<u32>` - Raw token sequence without special tokens
    pub fn encode_raw(&self, text: &str) -> Vec<u32> {
        let encoding = self.tokenizer.encode(EncodeInput::Single(text.into()), false).unwrap();
        encoding.get_ids().iter().copied().collect()
    }
}

/// Trait defining the core tokenizer interface
pub trait Tokenizer {
    /// Tokenize input text
    fn tokenize(&self, text: &str) -> Vec<u32>;
    
    /// Decode token sequence to text
    fn decode(&self, ids: &[u32]) -> String;
    
    /// Batch decode multiple token sequences
    fn batch_decode(&self, batch_ids: &[Vec<u32>]) -> Vec<String>;
}

impl Tokenizer for Gemma3Tokenizer {
    fn tokenize(&self, text: &str) -> Vec<u32> {
        self.tokenize(text)
    }
    
    fn decode(&self, ids: &[u32]) -> String {
        self.decode(ids)
    }
    
    fn batch_decode(&self, batch_ids: &[Vec<u32>]) -> Vec<String> {
        self.batch_decode(batch_ids)
    }
} 