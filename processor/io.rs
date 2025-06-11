use crate::tensor::{NDTensor, TensorData};
use std::collections::HashMap;

/// Utility function to create a HashMap with prefixed tensor names
///
/// # Arguments
/// * `tensors` - Original tensor HashMap
/// * `prefix` - Prefix to add to all tensor names
///
/// # Returns
/// * `HashMap<String, NDTensor<T>>` - New HashMap with prefixed names
pub fn add_prefix_to_tensor_names<T: TensorData>(
    tensors: HashMap<String, NDTensor<T>>,
    prefix: &str,
) -> HashMap<String, NDTensor<T>> {
    tensors
        .into_iter()
        .map(|(k, v)| (format!("{}_{}", prefix, k), v))
        .collect()
}
