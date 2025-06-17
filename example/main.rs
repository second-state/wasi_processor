mod mlx_output;
mod mlx_wrapper;

#[cfg(not(target_family = "wasm"))]
use mlx_rs::Array;
use rust_processor::auto::processing_auto::AutoProcessor;
use rust_processor::gemma3::detokenizer::{decode, Detokenizer};
use rust_processor::{prepare_inputs, tensor::*};
use std::env;

// Use local MLX functions
use mlx_output::{add_prefix_to_tensor_names, save_tensors_as_npy};
use mlx_wrapper::ArrayExt;

fn main() {
    let args: Vec<String> = env::args().collect();
    let model_dir = if args.len() > 1 {
        &args[1]
    } else {
        "gemma-3-4b-it-4bit"
    };

    #[cfg(target_family = "wasm")]
    {
        println!("WASM target detected - MLX functionality is limited");
        return;
    }

    #[cfg(not(target_family = "wasm"))]
    {
        match AutoProcessor::from_pretrained(model_dir) {
            Ok(processor_type) => match processor_type {
                rust_processor::auto::processing_auto::AutoProcessorType::Gemma3(mut processor) => {
                    run_single_image_test(&mut processor);
                    run_multiple_images_test(&mut processor);
                    run_text_only_test(&mut processor);
                    run_token_decoding_test(&mut processor);
                }
                _ => println!("The loaded processor is not a Gemma3Processor!"),
            },
            Err(e) => println!("AutoProcessor loading failed: {}", e),
        }
    }
}

#[cfg(not(target_family = "wasm"))]
/// Test single image processing (comparison with Python)
fn run_single_image_test(processor: &mut rust_processor::gemma3::processor::Gemma3Processor) {
    println!("\n===== Single Image Test (Comparison with Python) =====");

    let prompt = "<bos><start_of_turn>user\
                                                Describe this image.<start_of_image><end_of_turn>\
                                                <start_of_turn>model";
    let image_path = "wasmedge-runtime-logo.png";

    let model_inputs = prepare_inputs(
        processor,
        &[image_path], // Use single image array
        prompt,
        262144,
        Some((896, 896)), // Use 896x896 as image size
    );

    // Print model inputs, similar to Python output
    print_model_inputs(
        "Single Image Model Inputs (similar to Python MLX output)",
        &model_inputs,
    );

    // Save tensors as .npy files using local MLX output
    println!("\n===== Saving Single Image Tensors as .npy Files =====");
    if let Err(e) = save_tensors_as_npy(&model_inputs) {
        println!("Failed to save .npy files: {:?}", e);
    } else {
        println!("Successfully saved all tensors as .npy files");
    }
}

#[cfg(not(target_family = "wasm"))]
/// Test multiple images processing
fn run_multiple_images_test(processor: &mut rust_processor::gemma3::processor::Gemma3Processor) {
    println!("\n===== Multiple Images Test =====");

    // Test multiple images (assuming multiple identical images for testing)
    let image_paths = ["cat.jpg", "dog.jpg"];

    let multi_prompt = "<bos><start_of_turn>user\
                                                 Describe these images.<start_of_image><start_of_image><end_of_turn>\
                                                 <start_of_turn>model";

    let multi_model_inputs = prepare_inputs(
        processor,
        &image_paths,
        multi_prompt,
        262144,
        Some((896, 896)),
    );

    print_model_inputs("Multiple Images Model Inputs", &multi_model_inputs);

    println!("\n===== Saving Multiple Images Tensors as .npy Files =====");
    let multi_tensors_with_prefix = add_prefix_to_tensor_names(multi_model_inputs, "multi");

    if let Err(e) = save_tensors_as_npy(&multi_tensors_with_prefix) {
        println!("Failed to save multiple images .npy files: {:?}", e);
    } else {
        println!("Successfully saved multiple images tensors as .npy files");
    }
}

#[cfg(not(target_family = "wasm"))]
/// Test text-only processing (no images)
fn run_text_only_test(processor: &mut rust_processor::gemma3::processor::Gemma3Processor) {
    println!("\n===== Text-Only Test (No Images) =====");

    let text_only_prompt = "<bos><start_of_turn>user\
                                                     Hello, how are you?<end_of_turn>\
                                                     <start_of_turn>model";

    let text_model_inputs = prepare_inputs(
        processor,
        &[], // Empty image array
        text_only_prompt,
        262144,
        Some((896, 896)),
    );

    print_model_inputs("Text-Only Model Inputs", &text_model_inputs);
}

#[cfg(not(target_family = "wasm"))]
/// Test token decoding from Answer.npy
fn run_token_decoding_test(processor: &rust_processor::gemma3::processor::Gemma3Processor) {
    println!("\n===== Decoding Tokens from Answer.npy =====");

    if let Ok(array) = Array::load("Answer.npy") {
        // Use ArrayExt to get tokens
        match array.to_vec::<i32>() {
            Ok(values) => {
                let tokens = values.into_iter().map(|x| x as usize).collect::<Vec<_>>();

                println!(
                    "Tokens loaded from Answer.npy: {:?}...",
                    &tokens[..10.min(tokens.len())]
                );

                // Use Detokenizer to decode tokens
                let decoded_text = decode(&tokens, processor, true);
                println!("Decoded text: {}", decoded_text);

                // Manual decoding using Detokenizer
                let mut detokenizer = Detokenizer::new(processor, true);
                detokenizer.decode_tokens(&tokens, processor);
                let manual_text = detokenizer.get_text();
                println!("Manually decoded text: {}", manual_text);
            }
            Err(e) => {
                println!("Cannot extract data from array: {}", e);

                // Try other types
                if let Ok(values) = array.to_vec::<i64>() {
                    let tokens = values.into_iter().map(|x| x as usize).collect::<Vec<_>>();

                    println!(
                        "Tokens loaded from Answer.npy (i64): {:?}...",
                        &tokens[..10.min(tokens.len())]
                    );

                    // Use Detokenizer to decode tokens
                    let decoded_text = decode(&tokens, processor, true);
                    println!("Decoded text: {}", decoded_text);
                } else {
                    println!("Cannot read data as i64 type either");
                }
            }
        }
    } else {
        println!("Cannot load Answer.npy file");
    }
}

#[cfg(not(target_family = "wasm"))]
/// Helper function to print model inputs in a formatted way
fn print_model_inputs(title: &str, model_inputs: &std::collections::HashMap<String, AnyNDTensor>) {
    println!("{}:", title);
    println!("{{");
    for (key, tensor) in model_inputs {
        let shape = tensor.shape();
        let shape_str = shape
            .iter()
            .map(|&d| d.to_string())
            .collect::<Vec<_>>()
            .join(", ");

        println!(
            "  '{}': array(shape=[{}], rtype={:?}, values={:?}...),",
            key,
            shape_str,
            tensor.rtype(),
            tensor.first_n(5)
        );
    }
    println!("}}");
}
