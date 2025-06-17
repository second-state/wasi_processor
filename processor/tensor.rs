/// Data type enumeration for NDTensor
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u8)]
pub enum RType {
    F16 = 0,
    F32 = 1,
    F64 = 2,
    U8 = 3,
    I32 = 4,
    I64 = 5,
}

/// Simple f16 representation using u16 (IEEE 754 half-precision)
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct F16(pub u16);

impl F16 {
    /// Convert f32 to f16
    pub fn from_f32(value: f32) -> Self {
        // Simple conversion from f32 to f16 (IEEE 754)
        let bits = value.to_bits();
        let sign = (bits >> 31) & 0x1;
        let exponent = ((bits >> 23) & 0xff) as i32;
        let mantissa = bits & 0x7fffff;

        // Handle special cases
        if exponent == 0xff {
            // Infinity or NaN
            let f16_mantissa = if mantissa == 0 { 0 } else { 0x200 };
            return F16(((sign << 15) | (0x1f << 10) | f16_mantissa) as u16);
        }

        if exponent == 0 && mantissa == 0 {
            // Zero
            return F16((sign << 15) as u16);
        }

        // Convert exponent
        let f16_exp = exponent - 127 + 15;

        if f16_exp <= 0 {
            // Underflow to zero or denormal
            return F16((sign << 15) as u16);
        }

        if f16_exp >= 31 {
            // Overflow to infinity
            return F16(((sign << 15) | (0x1f << 10)) as u16);
        }

        // Normal number
        let f16_mantissa = (mantissa >> 13) & 0x3ff;
        F16(((sign << 15) | ((f16_exp as u32) << 10) | f16_mantissa) as u16)
    }

    /// Convert f16 to f32
    pub fn to_f32(self) -> f32 {
        let bits = self.0 as u32;
        let sign = (bits >> 15) & 0x1;
        let exponent = ((bits >> 10) & 0x1f) as i32;
        let mantissa = bits & 0x3ff;

        if exponent == 0x1f {
            // Infinity or NaN
            let f32_mantissa = if mantissa == 0 { 0 } else { mantissa << 13 };
            return f32::from_bits((sign << 31) | (0xff << 23) | f32_mantissa);
        }

        if exponent == 0 && mantissa == 0 {
            // Zero
            return f32::from_bits((sign << 31) as u32);
        }

        if exponent == 0 {
            // Denormal - not implemented, return zero
            return f32::from_bits((sign << 31) as u32);
        }

        // Normal number
        let f32_exp = (exponent - 15 + 127) as u32;
        let f32_mantissa = mantissa << 13;
        f32::from_bits((sign << 31) | (f32_exp << 23) | f32_mantissa)
    }
}

impl RType {
    /// Get the size in bytes for each data type
    pub const fn size_of(&self) -> usize {
        match self {
            RType::F16 => 2, // f16 is 2 bytes
            RType::F32 => 4, // f32 is 4 bytes
            RType::F64 => 8, // f64 is 8 bytes
            RType::U8 => 1,  // u8 is 1 byte
            RType::I32 => 4, // i32 is 4 bytes
            RType::I64 => 8, // i64 is 8 bytes
        }
    }

    /// Convert from u8 value
    pub const fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(RType::F16),
            1 => Some(RType::F32),
            2 => Some(RType::F64),
            3 => Some(RType::U8),
            4 => Some(RType::I32),
            5 => Some(RType::I64),
            _ => None,
        }
    }
}

/// Trait for types that can be used in NDTensor
pub trait TensorData: Clone + Copy + Default + 'static {
    const RTYPE: RType;
    fn to_bytes_vec(self) -> Vec<u8>;
    fn from_bytes_slice(bytes: &[u8]) -> Self;
}

// Macro to implement TensorData for primitive types
macro_rules! impl_tensor_data {
    ($type:ty, $rtype:expr, $size:expr) => {
        impl TensorData for $type {
            const RTYPE: RType = $rtype;

            fn to_bytes_vec(self) -> Vec<u8> {
                self.to_le_bytes().to_vec()
            }

            fn from_bytes_slice(bytes: &[u8]) -> Self {
                <$type>::from_le_bytes(bytes[..$size].try_into().unwrap())
            }
        }
    };
}

// Implement TensorData for all primitive types
impl_tensor_data!(f32, RType::F32, 4);
impl_tensor_data!(f64, RType::F64, 8);
impl_tensor_data!(i32, RType::I32, 4);
impl_tensor_data!(i64, RType::I64, 8);

// Special implementations for F16 and u8
impl TensorData for F16 {
    const RTYPE: RType = RType::F16;

    fn to_bytes_vec(self) -> Vec<u8> {
        self.0.to_le_bytes().to_vec()
    }

    fn from_bytes_slice(bytes: &[u8]) -> Self {
        F16(u16::from_le_bytes(bytes[..2].try_into().unwrap()))
    }
}

impl TensorData for u8 {
    const RTYPE: RType = RType::U8;

    fn to_bytes_vec(self) -> Vec<u8> {
        vec![self]
    }

    fn from_bytes_slice(bytes: &[u8]) -> Self {
        bytes[0]
    }
}

/// Multi-dimensional tensor type that stores data as a flat vector with shape information
#[derive(Debug, Clone)]
pub struct NDTensor<T: TensorData> {
    /// Flattened data storage
    pub data: Vec<T>,
    /// Dimension information
    pub shape: Vec<u32>,
    /// Data type indicator
    pub rtype: RType,
}

impl<T: TensorData> NDTensor<T> {
    /// Create a new NDTensor with the given data and shape
    ///
    /// # Arguments
    /// * `data` - Flattened data vector
    /// * `shape` - Shape of the tensor
    pub fn new(data: Vec<T>, shape: Vec<u32>) -> Self {
        Self {
            data,
            shape,
            rtype: T::RTYPE,
        }
    }

    /// Get the shape information
    pub fn shape(&self) -> &Vec<u32> {
        &self.shape
    }

    /// Get the first N values from the tensor
    pub fn first_n(&self, n: usize) -> Vec<T> {
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
        let expected_len: u32 = self.shape.iter().product();
        self.data.len() == expected_len as usize
    }

    /// Convert to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut result = Vec::new();

        // Add rtype
        result.push(self.rtype as u8);

        // Add shape
        let dim_buf: Vec<u8> = self
            .shape
            .iter()
            .flat_map(|&dim| dim.to_le_bytes())
            .collect();

        result.extend_from_slice(&(dim_buf.len() as u32).to_le_bytes());
        result.extend_from_slice(&dim_buf);

        // Add data
        let data_buf: Vec<u8> = self
            .data
            .iter()
            .flat_map(|&item| item.to_bytes_vec())
            .collect();

        result.extend_from_slice(&(data_buf.len() as u32).to_le_bytes());
        result.extend_from_slice(&data_buf);

        result
    }

    /// Create from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 9 {
            return Err("Insufficient bytes".to_string());
        }

        let mut offset = 0;

        // Read rtype
        let rtype = RType::from_u8(bytes[offset]).ok_or("Invalid rtype")?;

        if rtype != T::RTYPE {
            return Err(format!(
                "Type mismatch: expected {:?}, got {:?}",
                T::RTYPE,
                rtype
            ));
        }
        offset += 1;

        // Read shape
        let dim_buf_len = u32::from_le_bytes(
            bytes[offset..offset + 4]
                .try_into()
                .map_err(|_| "Invalid dim buf len")?,
        ) as usize;
        offset += 4;

        let shape: Vec<u32> = bytes[offset..offset + dim_buf_len]
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        offset += dim_buf_len;

        // Read data
        let data_buf_len = u32::from_le_bytes(
            bytes[offset..offset + 4]
                .try_into()
                .map_err(|_| "Invalid data buf len")?,
        ) as usize;
        offset += 4;

        let element_size = rtype.size_of();
        let data: Vec<T> = bytes[offset..offset + data_buf_len]
            .chunks_exact(element_size)
            .map(T::from_bytes_slice)
            .collect();

        Ok(Self::new(data, shape))
    }
}

pub type NDTensorF16 = NDTensor<F16>;
pub type NDTensorF32 = NDTensor<f32>;
pub type NDTensorF64 = NDTensor<f64>;
pub type NDTensorU8 = NDTensor<u8>;
pub type NDTensorI32 = NDTensor<i32>;
pub type NDTensorI64 = NDTensor<i64>;

#[derive(Debug, Clone)]
pub enum AnyNDTensor {
    F16(NDTensorF16),
    F32(NDTensorF32),
    F64(NDTensorF64),
    I8(NDTensorU8),
    I32(NDTensorI32),
    I64(NDTensorI64),
}

// Unified macro for AnyNDTensor operations
macro_rules! any_tensor_op {
    ($self:expr, $method:ident $(, $($args:expr),+)?) => {
        match $self {
            AnyNDTensor::F16(t) => t.$method($($($args),+)?),
            AnyNDTensor::F32(t) => t.$method($($($args),+)?),
            AnyNDTensor::F64(t) => t.$method($($($args),+)?),
            AnyNDTensor::I8(t) => t.$method($($($args),+)?),
            AnyNDTensor::I32(t) => t.$method($($($args),+)?),
            AnyNDTensor::I64(t) => t.$method($($($args),+)?),
        }
    };
}

macro_rules! any_tensor_field {
    // For field access returning reference
    ($self:expr, &$field:ident) => {
        match $self {
            AnyNDTensor::F16(t) => &t.$field,
            AnyNDTensor::F32(t) => &t.$field,
            AnyNDTensor::F64(t) => &t.$field,
            AnyNDTensor::I8(t) => &t.$field,
            AnyNDTensor::I32(t) => &t.$field,
            AnyNDTensor::I64(t) => &t.$field,
        }
    };
    // For field access returning value
    ($self:expr, $field:ident) => {
        match $self {
            AnyNDTensor::F16(t) => t.$field,
            AnyNDTensor::F32(t) => t.$field,
            AnyNDTensor::F64(t) => t.$field,
            AnyNDTensor::I8(t) => t.$field,
            AnyNDTensor::I32(t) => t.$field,
            AnyNDTensor::I64(t) => t.$field,
        }
    };
}

impl AnyNDTensor {
    /// Get shape
    pub fn shape(&self) -> &Vec<u32> {
        any_tensor_field!(self, &shape)
    }

    /// Get rtype
    pub fn rtype(&self) -> RType {
        any_tensor_field!(self, rtype)
    }

    /// Get first N values as f64 for uniform interface
    pub fn first_n(&self, n: usize) -> Vec<f64> {
        match self {
            AnyNDTensor::F16(t) => t.data.iter().take(n).map(|x| x.to_f32() as f64).collect(),
            AnyNDTensor::F32(t) => t.data.iter().take(n).map(|&x| x as f64).collect(),
            AnyNDTensor::F64(t) => t.data.iter().take(n).copied().collect(),
            AnyNDTensor::I8(t) => t.data.iter().take(n).map(|&x| x as f64).collect(),
            AnyNDTensor::I32(t) => t.data.iter().take(n).map(|&x| x as f64).collect(),
            AnyNDTensor::I64(t) => t.data.iter().take(n).map(|&x| x as f64).collect(),
        }
    }

    /// Get length
    pub fn len(&self) -> usize {
        any_tensor_op!(self, len)
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        any_tensor_op!(self, is_empty)
    }

    /// Convert to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        any_tensor_op!(self, to_bytes)
    }

    /// Create from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.is_empty() {
            return Err("Empty bytes".to_string());
        }

        match RType::from_u8(bytes[0]).ok_or("Invalid rtype")? {
            RType::F16 => Ok(AnyNDTensor::F16(NDTensorF16::from_bytes(bytes)?)),
            RType::F32 => Ok(AnyNDTensor::F32(NDTensorF32::from_bytes(bytes)?)),
            RType::F64 => Ok(AnyNDTensor::F64(NDTensorF64::from_bytes(bytes)?)),
            RType::U8 => Ok(AnyNDTensor::I8(NDTensorU8::from_bytes(bytes)?)),
            RType::I32 => Ok(AnyNDTensor::I32(NDTensorI32::from_bytes(bytes)?)),
            RType::I64 => Ok(AnyNDTensor::I64(NDTensorI64::from_bytes(bytes)?)),
        }
    }

    /// Check if F32 tensor
    pub fn is_f32(&self) -> bool {
        matches!(self, AnyNDTensor::F32(_))
    }

    /// Convert to F32 tensor if possible
    pub fn to_f32_tensor(&self) -> Option<NDTensorF32> {
        match self {
            AnyNDTensor::F32(tensor) => Some(tensor.clone()),
            _ => None,
        }
    }
}
