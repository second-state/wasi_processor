#![allow(dead_code)]
use crate::gemma3::tokenizer::Gemma3Tokenizer;
use crate::gemma3::image_processor::Gemma3ImageProcessor;

/// Gemma3 Processor for handling both text and image inputs
#[derive(Debug, Clone)]
pub struct Gemma3Processor {
    pub tokenizer: Gemma3Tokenizer,
    pub image_processor: Gemma3ImageProcessor,
    pub chat_template: Option<String>,
    pub image_seq_length: usize,
    pub image_token_id: u32,
    pub boi_token: String,     // <start_of_image>
    pub eoi_token: String,     // <end_of_image>
    pub image_token: String,   // <image>
}

/// Batch feature structure containing all processed inputs
#[derive(Debug)]
pub struct BatchFeature {
    pub input_ids: Vec<Vec<u32>>,          // Keep u32 type, convert to i64 at output
    pub attention_mask: Vec<Vec<u32>>,     // Keep u32 type, convert to i64 at output
    pub token_type_ids: Vec<Vec<u32>>,     // Keep u32 type, convert to i64 at output
    pub pixel_values: Option<Vec<f32>>,
    pub num_crops: Option<Vec<usize>>,
}

impl Gemma3Processor {
    /// Load processor from pretrained model directory
    /// 
    /// # Arguments
    /// * `model_dir` - Path to the model directory
    /// 
    /// # Returns
    /// * `Self` - Configured Gemma3Processor instance
    pub fn from_pretrained(model_dir: &str) -> Self {
        let tokenizer = Gemma3Tokenizer::from_pretrained(model_dir);
        let image_processor = Gemma3ImageProcessor::from_pretrained(model_dir);
        
        let image_seq_length = image_processor.image_seq_length;
        let image_token_id = tokenizer.image_token_id;
        let boi_token = tokenizer.boi_token.clone();
        let eoi_token = tokenizer.eoi_token.clone();
        let image_token = tokenizer.image_token.clone();
        
        Gemma3Processor { 
            tokenizer, 
            image_processor,
            chat_template: None,
            image_seq_length,
            image_token_id,
            boi_token,
            eoi_token,
            image_token,
        }
    }

    /// Process text, converting image markers to image token IDs
    /// 
    /// # Arguments
    /// * `text` - Input text to process
    /// 
    /// # Returns
    /// * `Vec<u32>` - Processed token sequence
    pub fn process_text(&self, text: &str) -> Vec<u32> {
        // First perform normal tokenization
        let tokens = self.tokenizer.tokenize(text);
        
        // If text contains image markers, we need special handling
        if text.contains(&self.boi_token) {
            self.process_text_with_image_tokens(&tokens)
        } else {
            tokens
        }
    }
    
    /// Process text containing image tokens
    /// 
    /// # Arguments
    /// * `tokens` - Original token sequence
    /// 
    /// # Returns
    /// * `Vec<u32>` - Processed token sequence with image tokens
    fn process_text_with_image_tokens(&self, tokens: &[u32]) -> Vec<u32> {
        // Create a new token sequence
        let mut new_tokens = Vec::new();
        let mut i = 0;
        
        // Use tokenizer's encode_raw method to get raw encoding
        let boi_tokens = self.tokenizer.encode_raw(&self.boi_token);
        
        while i < tokens.len() {
            // Check if current position matches <start_of_image> token sequence  
            // Note: Since we use encode_raw, boi_tokens should only contain [255999]
            let mut found_boi = false;
            if i + boi_tokens.len() <= tokens.len() {
                let mut matches = true;
                for (j, &expected_token) in boi_tokens.iter().enumerate() {
                    if tokens[i + j] != expected_token {
                        matches = false;
                        break;
                    }
                }
                if matches {
                    found_boi = true;
                }
            }
            
            if found_boi {
                // First keep <start_of_image> token (255999)
                new_tokens.push(tokens[i]);
                
                // Then add image token sequence
                for _ in 0..self.image_seq_length {
                    new_tokens.push(self.image_token_id);
                }
                i += boi_tokens.len(); // Skip processed tokens
            } else {
                // Not an image marker, copy directly
                new_tokens.push(tokens[i]);
                i += 1;
            }
        }
        
        new_tokens
    }
    
    /// Process text and optional image
    /// 
    /// # Arguments
    /// * `text` - Optional text input
    /// * `image` - Optional image bytes
    /// 
    /// # Returns
    /// * `BatchFeature` - Processed features ready for model input
    pub fn process(&self, text: Option<&str>, image: Option<&[u8]>) -> BatchFeature {
        match (text, image) {
            (Some(text), Some(img)) => {
                self.process_text_and_image(text, img)
            },
            (Some(text), None) => {
                self.process_text_only(text)
            },
            (None, Some(img)) => {
                self.process_image_only(img)
            },
            _ => {
                // No input, return empty result
                BatchFeature {
                    input_ids: vec![vec![]],
                    attention_mask: vec![vec![]],
                    token_type_ids: vec![vec![]],
                    pixel_values: None,
                    num_crops: None,
                }
            },
        }
    }
    
    /// Process both text and image
    /// 
    /// # Arguments
    /// * `text` - Text input
    /// * `img` - Image bytes
    /// 
    /// # Returns
    /// * `BatchFeature` - Processed features
    fn process_text_and_image(&self, text: &str, img: &[u8]) -> BatchFeature {
        // Process text and image simultaneously
        let input_ids = self.process_text(text);
        
        // Process image to get pixel values
        let (pixel_values, num_crops) = self.image_processor.preprocess(img);
        
        // Create attention mask - all 1s since no padding
        let attention_mask = vec![1; input_ids.len()];
        
        // Create token type IDs (0 for text, 1 for image)
        let token_type_ids = self.create_token_type_ids(&input_ids);
        
        BatchFeature {
            input_ids: vec![input_ids],
            attention_mask: vec![attention_mask],
            token_type_ids: vec![token_type_ids],
            pixel_values: Some(pixel_values),
            num_crops: Some(num_crops),
        }
    }
    
    /// Process text only (no image)
    /// 
    /// # Arguments
    /// * `text` - Text input
    /// 
    /// # Returns
    /// * `BatchFeature` - Processed features
    fn process_text_only(&self, text: &str) -> BatchFeature {
        let input_ids = self.process_text(text);
        
        // Create attention mask - all 1s since no padding
        let attention_mask = vec![1; input_ids.len()];
        
        // Create token type IDs - all 0s for text
        let token_type_ids = self.create_token_type_ids(&input_ids);
        
        BatchFeature {
            input_ids: vec![input_ids],
            attention_mask: vec![attention_mask],
            token_type_ids: vec![token_type_ids],
            pixel_values: None,
            num_crops: None,
        }
    }
    
    /// Process image only (no text)
    /// 
    /// # Arguments
    /// * `img` - Image bytes
    /// 
    /// # Returns
    /// * `BatchFeature` - Processed features
    fn process_image_only(&self, img: &[u8]) -> BatchFeature {
        // Process image only, create input containing image markers
        let image_token_sequence = vec![self.image_token_id; self.image_seq_length];
        let (pixel_values, num_crops) = self.image_processor.preprocess(img);
        
        // Since all are image markers, token type IDs are all 1s
        let token_type_ids = vec![1; self.image_seq_length];
        let attention_mask = vec![1; self.image_seq_length];
        
        BatchFeature {
            input_ids: vec![image_token_sequence],
            attention_mask: vec![attention_mask],
            token_type_ids: vec![token_type_ids],
            pixel_values: Some(pixel_values),
            num_crops: Some(num_crops),
        }
    }
    
    /// Create token type IDs based on input IDs
    /// 
    /// # Arguments
    /// * `input_ids` - Input token sequence
    /// 
    /// # Returns
    /// * `Vec<u32>` - Token type IDs (0 for text, 1 for image)
    fn create_token_type_ids(&self, input_ids: &[u32]) -> Vec<u32> {
        let mut token_type_ids = vec![0; input_ids.len()];
        for (i, &id) in input_ids.iter().enumerate() {
            if id == self.image_token_id {
                token_type_ids[i] = 1;
            }
        }
        token_type_ids
    }
    
    /// Batch process multiple texts and images
    /// 
    /// # Arguments
    /// * `texts` - Vector of text inputs
    /// * `images` - Optional vector of image bytes
    /// 
    /// # Returns
    /// * `BatchFeature` - Batched processed features
    pub fn process_batch(&self, texts: Vec<&str>, images: Option<Vec<&[u8]>>) -> BatchFeature {
        let batch_size = texts.len();
        let mut all_input_ids = Vec::with_capacity(batch_size);
        let mut all_attention_masks = Vec::with_capacity(batch_size);
        let mut all_token_type_ids = Vec::with_capacity(batch_size);
        let mut all_pixel_values = Vec::new();
        let mut all_num_crops = Vec::new();
        
        // Process each text and possible image
        match images {
            Some(imgs) if imgs.len() == batch_size => {
                // Text and image counts are equal, process one-to-one
                for (text, img) in texts.iter().zip(imgs.iter()) {
                    let result = self.process(Some(text), Some(img));
                    self.collect_batch_results(&result, &mut all_input_ids, &mut all_attention_masks, 
                                             &mut all_token_type_ids, &mut all_pixel_values, &mut all_num_crops);
                }
            },
            _ => {
                // Process text only
                for text in texts {
                    let result = self.process(Some(text), None);
                    self.collect_batch_results(&result, &mut all_input_ids, &mut all_attention_masks, 
                                             &mut all_token_type_ids, &mut all_pixel_values, &mut all_num_crops);
                }
            }
        }
        
        BatchFeature {
            input_ids: all_input_ids,
            attention_mask: all_attention_masks,
            token_type_ids: all_token_type_ids,
            pixel_values: if all_pixel_values.is_empty() { None } else { Some(all_pixel_values) },
            num_crops: if all_num_crops.is_empty() { None } else { Some(all_num_crops) },
        }
    }
    
    /// Collect results from individual processing into batch collections
    /// 
    /// # Arguments
    /// * `result` - Individual processing result
    /// * `all_input_ids` - Collection of all input IDs
    /// * `all_attention_masks` - Collection of all attention masks
    /// * `all_token_type_ids` - Collection of all token type IDs
    /// * `all_pixel_values` - Collection of all pixel values
    /// * `all_num_crops` - Collection of all crop numbers
    fn collect_batch_results(
        &self,
        result: &BatchFeature,
        all_input_ids: &mut Vec<Vec<u32>>,
        all_attention_masks: &mut Vec<Vec<u32>>,
        all_token_type_ids: &mut Vec<Vec<u32>>,
        all_pixel_values: &mut Vec<f32>,
        all_num_crops: &mut Vec<usize>,
    ) {
        all_input_ids.extend(result.input_ids.clone());
        all_attention_masks.extend(result.attention_mask.clone());
        all_token_type_ids.extend(result.token_type_ids.clone());
        
        if let Some(ref pv) = result.pixel_values {
            all_pixel_values.extend(pv);
        }
        
        if let Some(ref nc) = result.num_crops {
            all_num_crops.extend(nc);
        }
    }

    /// Batch decode token sequences to text
    /// 
    /// # Arguments
    /// * `batch_ids` - Batch of token sequences
    /// 
    /// # Returns
    /// * `Vec<String>` - Decoded text strings
    pub fn batch_decode(&self, batch_ids: &[Vec<u32>]) -> Vec<String> {
        batch_ids.iter().map(|ids| self.decode(ids)).collect()
    }

    /// Decode single token sequence to text
    /// 
    /// # Arguments
    /// * `ids` - Token sequence
    /// 
    /// # Returns
    /// * `String` - Decoded text
    pub fn decode(&self, ids: &[u32]) -> String {
        self.tokenizer.decode(ids)
    }
}

/// Trait defining the core processor interface
pub trait Processor {
    /// Process text and optional image
    fn process(&self, text: Option<&str>, image: Option<&[u8]>) -> BatchFeature;
    
    /// Batch decode token sequences
    fn batch_decode(&self, batch_ids: &[Vec<u32>]) -> Vec<String>;
    
    /// Decode single token sequence
    fn decode(&self, ids: &[u32]) -> String;
}

impl Processor for Gemma3Processor {
    fn process(&self, text: Option<&str>, image: Option<&[u8]>) -> BatchFeature {
        self.process(text, image)
    }
    
    fn batch_decode(&self, batch_ids: &[Vec<u32>]) -> Vec<String> {
        self.batch_decode(batch_ids)
    }
    
    fn decode(&self, ids: &[u32]) -> String {
        self.decode(ids)
    }
} 