#[cfg(not(target_family = "wasm"))]
use mlx_rs::Array;
#[cfg(not(target_family = "wasm"))]
use mlx_sys::mlx_save;
use rust_processor::NDTensorF32;
use std::collections::HashMap;
#[cfg(not(target_family = "wasm"))]
use std::ffi::CString;

/// Save a collection of NDTensors as .npy files using MLX
///
/// # Arguments
/// * `tensors` - HashMap containing tensor name and NDTensor pairs
///
/// # Returns
/// * `Result<(), Box<dyn std::error::Error>>` - Success or error
pub fn save_tensors_as_npy(
    tensors: &HashMap<String, NDTensorF32>,
) -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(target_family = "wasm")]
    {
        println!("WASM target - saving tensors is not supported");
        return Err("Tensor saving not supported in WASM".into());
    }

    #[cfg(not(target_family = "wasm"))]
    {
        for (key, tensor) in tensors {
            save_tensor_as_npy(key, tensor)?;
        }
        Ok(())
    }
}

#[cfg(not(target_family = "wasm"))]
/// Save a single NDTensor as a .npy file using MLX
///
/// # Arguments
/// * `name` - Name of the tensor (used as filename)
/// * `tensor` - The NDTensor to save
fn save_tensor_as_npy(name: &str, tensor: &NDTensorF32) -> Result<(), Box<dyn std::error::Error>> {
    // Create MLX Array with appropriate shape
    let shape: Vec<i32> = tensor.shape.iter().map(|&x| x as i32).collect();

    if name.contains("pixel_values") {
        // pixel_values keep f32 type
        let mlx_array = Array::from_slice(&tensor.data, shape.as_slice());
        save_mlx_array(name, &mlx_array)?;
    } else if name.contains("input_ids")
        || name.contains("mask")
        || name.contains("attention_mask")
        || name.contains("token_type_ids")
        || name.contains("num_crops")
    {
        // Other types convert to i32
        let data: Vec<i32> = tensor.data.iter().map(|&x| x as i32).collect();
        let mlx_array = Array::from_slice(&data, shape.as_slice());
        save_mlx_array(name, &mlx_array)?;
    }

    Ok(())
}

#[cfg(not(target_family = "wasm"))]
/// Save an MLX Array to a .npy file
///
/// # Arguments
/// * `name` - Base filename (without extension)
/// * `array` - The MLX Array to save
fn save_mlx_array(name: &str, array: &Array) -> Result<(), Box<dyn std::error::Error>> {
    let file_path = format!("{}.npy", name);
    println!("Saving file: {}", file_path);

    let c_filename = CString::new(file_path)?;
    unsafe {
        mlx_save(c_filename.as_ptr(), array.as_ptr());
    }

    Ok(())
}

/// Utility function to create a HashMap with prefixed tensor names
///
/// # Arguments
/// * `tensors` - Original tensor HashMap
/// * `prefix` - Prefix to add to all tensor names
///
/// # Returns
/// * `HashMap<String, NDTensor>` - New HashMap with prefixed names
pub fn add_prefix_to_tensor_names(
    tensors: HashMap<String, NDTensorF32>,
    prefix: &str,
) -> HashMap<String, NDTensorF32> {
    tensors
        .into_iter()
        .map(|(k, v)| (format!("{}_{}", prefix, k), v))
        .collect()
}
