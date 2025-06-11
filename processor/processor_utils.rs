use crate::gemma3::processor::Gemma3Processor;
use crate::tensor::NDTensorF32;
use std::collections::HashMap;
use std::fs;

/// Prepare inputs for the model, supporting both text-only and image+text processing
///
/// # Arguments
/// * `processor` - Mutable reference to the Gemma3Processor
/// * `image_paths` - Array of image file paths (can be empty for text-only)
/// * `prompt` - Text prompt for the model
/// * `_image_token_index` - Image token index (currently unused but kept for API consistency)
/// * `resize_shape` - Optional image resize dimensions
pub fn prepare_inputs(
    processor: &mut Gemma3Processor,
    image_paths: &[&str],
    prompt: &str,
    _image_token_index: u32,
    resize_shape: Option<(u32, u32)>,
) -> HashMap<String, NDTensorF32> {
    // Read multiple images
    let mut images_bytes = Vec::new();

    // If no images provided, return text-only results
    if image_paths.is_empty() {
        return process_text_only(processor, prompt);
    }

    // Read all images
    for image_path in image_paths {
        let image_bytes =
            fs::read(image_path).expect(&format!("Cannot read image: {}", image_path));
        images_bytes.push(image_bytes);
    }

    // Set resize shape if provided
    if let Some(shape) = resize_shape {
        processor.image_processor.size = shape;
    }

    // Process images and text
    process_images_and_text(processor, &images_bytes, prompt, resize_shape)
}

/// Process text-only input (no images)
fn process_text_only(
    processor: &mut Gemma3Processor,
    prompt: &str,
) -> HashMap<String, NDTensorF32> {
    let batch_feature = processor.process(Some(prompt), None);
    let mut model_inputs = HashMap::new();

    // Add input_ids (2D tensor [1, seq_len])
    if !batch_feature.input_ids.is_empty() {
        let input_ids = &batch_feature.input_ids[0];
        let shape = vec![1u32, input_ids.len() as u32];
        let data: Vec<f32> = input_ids.iter().map(|&id| id as f32).collect();
        model_inputs.insert("input_ids".to_string(), NDTensorF32::new(data, shape));
    }

    // Add attention_mask (2D tensor [1, seq_len])
    if !batch_feature.attention_mask.is_empty() {
        let attention_mask = &batch_feature.attention_mask[0];
        let shape = vec![1u32, attention_mask.len() as u32];
        let data: Vec<f32> = attention_mask.iter().map(|&mask| mask as f32).collect();
        model_inputs.insert("mask".to_string(), NDTensorF32::new(data, shape));
    }

    // Add token_type_ids (2D tensor [1, seq_len])
    if !batch_feature.token_type_ids.is_empty() {
        let token_type_ids = &batch_feature.token_type_ids[0];
        let shape = vec![1u32, token_type_ids.len() as u32];
        let data: Vec<f32> = token_type_ids.iter().map(|&id| id as f32).collect();
        model_inputs.insert("token_type_ids".to_string(), NDTensorF32::new(data, shape));
    }

    model_inputs
}

/// Process both images and text
fn process_images_and_text(
    processor: &mut Gemma3Processor,
    images_bytes: &[Vec<u8>],
    prompt: &str,
    resize_shape: Option<(u32, u32)>,
) -> HashMap<String, NDTensorF32> {
    let mut all_pixel_values = Vec::new();
    let mut all_num_crops = Vec::new();
    let mut final_input_ids = Vec::new();
    let mut final_attention_mask = Vec::new();
    let mut final_token_type_ids = Vec::new();

    // Process the first image with text
    if !images_bytes.is_empty() {
        let batch_feature = processor.process(Some(prompt), Some(&images_bytes[0]));

        // Collect results from the first image
        if let Some(pixel_values) = batch_feature.pixel_values {
            all_pixel_values.extend(pixel_values);
        }
        if let Some(num_crops) = batch_feature.num_crops {
            all_num_crops.extend(num_crops);
        }

        if !batch_feature.input_ids.is_empty() {
            final_input_ids = batch_feature.input_ids[0].clone();
        }
        if !batch_feature.attention_mask.is_empty() {
            final_attention_mask = batch_feature.attention_mask[0].clone();
        }
        if !batch_feature.token_type_ids.is_empty() {
            final_token_type_ids = batch_feature.token_type_ids[0].clone();
        }

        // Process remaining images (image-only, no text)
        for image_bytes in &images_bytes[1..] {
            let batch_feature = processor.process(None, Some(image_bytes));

            if let Some(pixel_values) = batch_feature.pixel_values {
                all_pixel_values.extend(pixel_values);
            }
            if let Some(num_crops) = batch_feature.num_crops {
                all_num_crops.extend(num_crops);
            }
        }
    }

    // Build model inputs (similar to test.py's model_inputs, but supporting multi-dimensional arrays)
    let mut model_inputs = HashMap::new();

    // Add pixel_values (adjust dimensions based on number of images)
    if !all_pixel_values.is_empty() {
        // Use provided resize_shape or default size
        let (height, width) = resize_shape.unwrap_or((896, 896));
        let channels = 3u32;
        let num_images = images_bytes.len() as u32;

        // Multi-image shape: [num_images, channels, height, width]
        let shape = vec![num_images, channels, height, width];
        model_inputs.insert(
            "pixel_values".to_string(),
            NDTensorF32::new(all_pixel_values, shape),
        );
    }

    // Add input_ids (2D tensor [1, seq_len])
    if !final_input_ids.is_empty() {
        let shape = vec![1u32, final_input_ids.len() as u32];
        let data: Vec<f32> = final_input_ids.iter().map(|&id| id as f32).collect();
        model_inputs.insert("input_ids".to_string(), NDTensorF32::new(data, shape));
    }

    // Add attention_mask (2D tensor [1, seq_len])
    if !final_attention_mask.is_empty() {
        let shape = vec![1u32, final_attention_mask.len() as u32];
        let data: Vec<f32> = final_attention_mask
            .iter()
            .map(|&mask| mask as f32)
            .collect();
        model_inputs.insert("mask".to_string(), NDTensorF32::new(data, shape));
    }

    // Add token_type_ids (2D tensor [1, seq_len])
    if !final_token_type_ids.is_empty() {
        let shape = vec![1u32, final_token_type_ids.len() as u32];
        let data: Vec<f32> = final_token_type_ids.iter().map(|&id| id as f32).collect();
        model_inputs.insert("token_type_ids".to_string(), NDTensorF32::new(data, shape));
    }

    // Add num_crops information
    if !all_num_crops.is_empty() {
        let shape = vec![all_num_crops.len() as u32];
        let data: Vec<f32> = all_num_crops.iter().map(|&crop| crop as f32).collect();
        model_inputs.insert("num_crops".to_string(), NDTensorF32::new(data, shape));
    }

    model_inputs
}
