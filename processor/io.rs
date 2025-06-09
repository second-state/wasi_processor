use crate::tensor::NDTensor;
use std::collections::HashMap;

/// Utility function to create a HashMap with prefixed tensor names
/// 
/// # Arguments
/// * `tensors` - Original tensor HashMap
/// * `prefix` - Prefix to add to all tensor names
/// 
/// # Returns
/// * `HashMap<String, NDTensor>` - New HashMap with prefixed names
pub fn add_prefix_to_tensor_names(
    tensors: HashMap<String, NDTensor>,
    prefix: &str,
) -> HashMap<String, NDTensor> {
    tensors
        .into_iter()
        .map(|(k, v)| (format!("{}_{}", prefix, k), v))
        .collect()
} 