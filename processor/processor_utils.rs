use crate::gemma3::processor::Gemma3Processor;
use crate::tensor::*;
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
) -> HashMap<String, AnyNDTensor> {
    let mut model_inputs = HashMap::new();

    // Check if we have images to process
    let has_images = !image_paths.is_empty();
    let mut input_ids = Vec::new();
    let mut attention_mask = Vec::new();
    let mut token_type_ids = Vec::new();

    if has_images {
        // Set resize shape if provided
        if let Some(shape) = resize_shape {
            processor.image_processor.size = shape;
        }

        // Read all images
        let mut images_bytes = Vec::new();
        for image_path in image_paths {
            let image_bytes =
                fs::read(image_path).expect(&format!("Cannot read image: {}", image_path));
            images_bytes.push(image_bytes);
        }

        // Process images and collect pixel values
        let mut all_pixel_values = Vec::new();
        let mut all_num_crops = Vec::new();

        // Process the first image with text
        let batch_feature = processor.process(Some(prompt), Some(&images_bytes[0]));

        // Collect results from the first image
        if let Some(pixel_values) = batch_feature.pixel_values {
            all_pixel_values.extend(pixel_values);
        }
        if let Some(num_crops) = batch_feature.num_crops {
            all_num_crops.extend(num_crops);
        }

        if !batch_feature.input_ids.is_empty() {
            input_ids = batch_feature.input_ids[0].clone();
        }
        if !batch_feature.attention_mask.is_empty() {
            attention_mask = batch_feature.attention_mask[0].clone();
        }
        if !batch_feature.token_type_ids.is_empty() {
            token_type_ids = batch_feature.token_type_ids[0].clone();
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

        // Add pixel_values to model inputs
        if !all_pixel_values.is_empty() {
            // Use provided resize_shape or default size
            let (height, width) = resize_shape.unwrap_or((896, 896));
            let channels = 3u32;
            let num_images = images_bytes.len() as u32;

            // Multi-image shape: [num_images, channels, height, width]
            let shape = vec![num_images, channels, height, width];
            model_inputs.insert(
                "pixel_values".to_string(),
                AnyNDTensor::F32(NDTensorF32::new(all_pixel_values, shape)),
            );
        }

        // Add num_crops information
        if !all_num_crops.is_empty() {
            let shape = vec![all_num_crops.len() as u32];
            let data: Vec<f32> = all_num_crops.iter().map(|&crop| crop as f32).collect();
            model_inputs.insert(
                "num_crops".to_string(),
                AnyNDTensor::F32(NDTensorF32::new(data, shape)),
            );
        }
    } else {
        // Process text-only (no images)
        let batch_feature = processor.process(Some(prompt), None);

        if !batch_feature.input_ids.is_empty() {
            input_ids = batch_feature.input_ids[0].clone();
        }
        if !batch_feature.attention_mask.is_empty() {
            attention_mask = batch_feature.attention_mask[0].clone();
        }
        if !batch_feature.token_type_ids.is_empty() {
            token_type_ids = batch_feature.token_type_ids[0].clone();
        }
    }

    // Add text-related tensors (unified processing)
    if !input_ids.is_empty() {
        let shape = vec![1u32, input_ids.len() as u32];
        let data: Vec<i64> = input_ids.iter().map(|&id| id as i64).collect();
        model_inputs.insert(
            "input_ids".to_string(),
            AnyNDTensor::I64(NDTensorI64::new(data, shape)),
        );
    }

    // Add attention_mask (2D tensor [1, seq_len])
    if !attention_mask.is_empty() {
        let shape = vec![1u32, attention_mask.len() as u32];
        let data: Vec<i64> = attention_mask.iter().map(|&mask| mask as i64).collect();
        model_inputs.insert(
            "mask".to_string(),
            AnyNDTensor::I64(NDTensorI64::new(data, shape)),
        );
    }

    // Add token_type_ids (2D tensor [1, seq_len])
    if !token_type_ids.is_empty() {
        let shape = vec![1u32, token_type_ids.len() as u32];
        let data: Vec<i64> = token_type_ids.iter().map(|&id| id as i64).collect();
        model_inputs.insert(
            "token_type_ids".to_string(),
            AnyNDTensor::I64(NDTensorI64::new(data, shape)),
        );
    }

    model_inputs
}
