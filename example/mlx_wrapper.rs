#[cfg(not(target_family = "wasm"))]
use mlx_rs::{Array, Dtype};
#[cfg(not(target_family = "wasm"))]
use mlx_sys::{mlx_array, mlx_load, mlx_default_cpu_stream_new};
#[cfg(not(target_family = "wasm"))]
use std::ffi::CString;
#[cfg(not(target_family = "wasm"))]
use std::ptr;
use std::convert::TryFrom;

/// Extension trait for MLX Array to provide additional functionality
pub trait ArrayExt {
    /// Load an array from a file path
    fn load(file_path: &str) -> Result<Self, Box<dyn std::error::Error>> where Self: Sized;
    
    /// Convert array to a vector of the specified type
    fn to_vec<T: TryFrom<i32> + Copy>(&self) -> Result<Vec<T>, Box<dyn std::error::Error>>;
}

#[cfg(not(target_family = "wasm"))]
impl ArrayExt for Array {
    fn load(file_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let c_file_path = CString::new(file_path)?;
        
        unsafe {
            let mut array = mlx_array { ctx: ptr::null_mut() };
            let default_stream = mlx_default_cpu_stream_new();
            let result = mlx_load(&mut array, c_file_path.as_ptr(), default_stream);
            
            if result != 0 {
                return Err(format!("Failed to load array from file: {}", file_path).into());
            }
            
            Ok(Array::from_ptr(array))
        }
    }
    
    fn to_vec<T: TryFrom<i32> + Copy>(&self) -> Result<Vec<T>, Box<dyn std::error::Error>> {
        // Check data type
        let dtype = self.dtype();
        println!("Array data type: {:?}", dtype);
        
        let shape_info = self.shape();
        let total_elements = shape_info.iter().product::<i32>() as usize;
        println!("Array shape: [{:#?}], total elements: {}", shape_info, total_elements);
        
        let result = self.flatten(None, None);
        if let Err(e) = result {
            return Err(format!("Failed to flatten array: {:?}", e).into());
        }
        
        if dtype == Dtype::Int32 {
            let mut vec = Vec::with_capacity(total_elements);
            for &x in result.unwrap().as_slice::<i32>() {
                match T::try_from(x) {
                    Ok(val) => vec.push(val),
                    Err(_) => return Err(format!("Element {} cannot be converted to target type", x).into()),
                }
            }
            println!("Successfully read {} elements", vec.len());
            Ok(vec)
        } else {
            return Err(format!("Unsupported dtype: {:?}", dtype).into());
        }
    }
}

// WASM placeholder implementation
#[cfg(target_family = "wasm")]
pub struct DummyArray;

#[cfg(target_family = "wasm")]
impl ArrayExt for DummyArray {
    fn load(_file_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        Err("MLX functionality not available in WASM".into())
    }
    
    fn to_vec<T: TryFrom<i32> + Copy>(&self) -> Result<Vec<T>, Box<dyn std::error::Error>> {
        Err("MLX functionality not available in WASM".into())
    }
} 