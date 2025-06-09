use crate::gemma3::processor::Gemma3Processor;
use serde::Deserialize;
use std::fs;

/// Enum representing different processor types that can be automatically loaded
pub enum AutoProcessorType {
    Gemma3(Gemma3Processor),
    NotImplemented(String),
}

/// Auto processor that can load the appropriate processor based on model configuration
pub struct AutoProcessor;

/// Configuration structure for reading model config files
#[derive(Deserialize)]
struct Config {
    model_type: Option<String>,
    text_config: Option<TextConfig>,
}

/// Text configuration nested in the main config
#[derive(Deserialize)]
struct TextConfig {
    model_type: Option<String>,
}

/// Extract model type from config.json file
/// 
/// # Arguments
/// * `model_dir` - Path to the model directory containing config.json
/// 
/// # Returns
/// * `Option<String>` - Model type if found, None otherwise
fn get_model_type_from_config(model_dir: &str) -> Option<String> {
    let config_path = format!("{}/config.json", model_dir);
    let config_str = fs::read_to_string(&config_path).ok()?;
    let config: Config = serde_json::from_str(&config_str).ok()?;
    
    if let Some(mt) = config.model_type {
        Some(mt)
    } else if let Some(tc) = config.text_config {
        tc.model_type
    } else {
        None
    }
}

impl AutoProcessor {
    /// Load processor from pretrained model directory
    /// 
    /// # Arguments
    /// * `model_dir` - Path to the model directory
    /// 
    /// # Returns
    /// * `Result<AutoProcessorType, String>` - Loaded processor or error message
    pub fn from_pretrained(model_dir: &str) -> Result<AutoProcessorType, String> {
        let model_type = get_model_type_from_config(model_dir)
            .ok_or_else(|| format!("Unable to get model_type from {}/config.json", model_dir))?;
            
        match model_type.as_str() {
            "gemma3" | "gemma3_text" => {
                Ok(AutoProcessorType::Gemma3(Gemma3Processor::from_pretrained(model_dir)))
            },
            _ => Err(format!("AutoProcessor for model_type '{}' is not implemented", model_type)),
        }
    }
} 