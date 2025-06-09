/// Multi-dimensional tensor type that stores data as a flat vector with shape information
#[derive(Debug, Clone)]
pub struct NDTensor {
    /// Flattened data storage
    pub data: Vec<f32>,
    /// Dimension information
    pub shape: Vec<usize>,
}

impl NDTensor {
    /// Create a new NDTensor with the given data and shape
    /// 
    /// # Arguments
    /// * `data` - Flattened data vector
    /// * `shape` - Shape of the tensor
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        // TODO: Add validation to ensure data.len() matches the product of shape dimensions
        NDTensor { data, shape }
    }

    /// Get the shape information
    pub fn shape(&self) -> &Vec<usize> {
        &self.shape
    }

    /// Get the first N values from the tensor
    pub fn first_n(&self, n: usize) -> Vec<f32> {
        self.data.iter().take(n).cloned().collect()
    }

    /// Get the total number of elements in the tensor
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if the tensor is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Validate that the data length matches the shape
    pub fn is_valid(&self) -> bool {
        let expected_len: usize = self.shape.iter().product();
        self.data.len() == expected_len
    }
} 