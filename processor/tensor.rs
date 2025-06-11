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
    pub fn size_of(&self) -> usize {
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
    pub fn from_u8(value: u8) -> Option<Self> {
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
    fn rtype() -> RType;
    fn to_bytes(&self) -> Vec<u8>;
    fn from_bytes(bytes: &[u8]) -> Self;
}

impl TensorData for F16 {
    fn rtype() -> RType {
        RType::F16
    }
    fn to_bytes(&self) -> Vec<u8> {
        self.0.to_le_bytes().to_vec()
    }
    fn from_bytes(bytes: &[u8]) -> Self {
        F16(u16::from_le_bytes(bytes.try_into().unwrap()))
    }
}

impl TensorData for f32 {
    fn rtype() -> RType {
        RType::F32
    }
    fn to_bytes(&self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }
    fn from_bytes(bytes: &[u8]) -> Self {
        f32::from_le_bytes(bytes.try_into().unwrap())
    }
}

impl TensorData for f64 {
    fn rtype() -> RType {
        RType::F64
    }
    fn to_bytes(&self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }
    fn from_bytes(bytes: &[u8]) -> Self {
        f64::from_le_bytes(bytes.try_into().unwrap())
    }
}

impl TensorData for u8 {
    fn rtype() -> RType {
        RType::U8
    }
    fn to_bytes(&self) -> Vec<u8> {
        vec![*self]
    }
    fn from_bytes(bytes: &[u8]) -> Self {
        bytes[0]
    }
}

impl TensorData for i32 {
    fn rtype() -> RType {
        RType::I32
    }
    fn to_bytes(&self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }
    fn from_bytes(bytes: &[u8]) -> Self {
        i32::from_le_bytes(bytes.try_into().unwrap())
    }
}

impl TensorData for i64 {
    fn rtype() -> RType {
        RType::I64
    }
    fn to_bytes(&self) -> Vec<u8> {
        self.to_le_bytes().to_vec()
    }
    fn from_bytes(bytes: &[u8]) -> Self {
        i64::from_le_bytes(bytes.try_into().unwrap())
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
        let rtype = T::rtype();
        NDTensor { data, shape, rtype }
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

    /// Flatten the tensor to u8 bytes for communication with other programs
    /// Format: | rtype | dim buf len | dim buf | data buf len | data buf |
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut result = Vec::new();

        // 1. Add rtype (1 byte)
        result.push(self.rtype as u8);

        // 2. Prepare dimension buffer (shape as u32 values in little-endian)
        let mut dim_buf = Vec::new();
        for &dim in &self.shape {
            dim_buf.extend_from_slice(&dim.to_le_bytes());
        }

        // 3. Add dim buf len (4 bytes, u32 in little-endian)
        let dim_buf_len = dim_buf.len() as u32;
        result.extend_from_slice(&dim_buf_len.to_le_bytes());

        // 4. Add dim buf
        result.extend_from_slice(&dim_buf);

        // 5. Prepare data buffer
        let mut data_buf = Vec::new();
        for &item in &self.data {
            data_buf.extend_from_slice(&item.to_bytes());
        }

        // 6. Add data buf len (4 bytes, u32 in little-endian)
        let data_buf_len = data_buf.len() as u32;
        result.extend_from_slice(&data_buf_len.to_le_bytes());

        // 7. Add data buf
        result.extend_from_slice(&data_buf);

        result
    }

    /// Create NDTensor from flattened u8 bytes
    /// Format: | rtype | dim buf len | dim buf | data buf len | data buf |
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 9 {
            // At least 1 + 4 + 4 = 9 bytes minimum
            return Err("Insufficient bytes for tensor header".to_string());
        }

        let mut offset = 0;

        // 1. Read rtype (1 byte)
        let rtype_value = bytes[offset];
        let rtype = RType::from_u8(rtype_value)
            .ok_or_else(|| format!("Invalid rtype value: {}", rtype_value))?;

        if rtype != T::rtype() {
            return Err(format!(
                "Type mismatch: expected {:?}, got {:?}",
                T::rtype(),
                rtype
            ));
        }

        offset += 1;

        // 2. Read dim buf len (4 bytes)
        let dim_buf_len = u32::from_le_bytes(
            bytes[offset..offset + 4]
                .try_into()
                .map_err(|_| "Failed to read dim buf len")?,
        ) as usize;
        offset += 4;

        // 3. Read dim buf and parse shape
        if offset + dim_buf_len > bytes.len() {
            return Err("Insufficient bytes for dimension buffer".to_string());
        }

        let mut shape = Vec::new();
        let dim_bytes = &bytes[offset..offset + dim_buf_len];
        for chunk in dim_bytes.chunks_exact(4) {
            let dim = u32::from_le_bytes(chunk.try_into().unwrap());
            shape.push(dim);
        }
        offset += dim_buf_len;

        // 4. Read data buf len (4 bytes)
        if offset + 4 > bytes.len() {
            return Err("Insufficient bytes for data buf len".to_string());
        }

        let data_buf_len = u32::from_le_bytes(
            bytes[offset..offset + 4]
                .try_into()
                .map_err(|_| "Failed to read data buf len")?,
        ) as usize;
        offset += 4;

        // 5. Read data buf and parse data
        if offset + data_buf_len > bytes.len() {
            return Err("Insufficient bytes for data buffer".to_string());
        }

        let data_bytes = &bytes[offset..offset + data_buf_len];
        let element_size = rtype.size_of();
        let mut data = Vec::new();

        for chunk in data_bytes.chunks_exact(element_size) {
            let element = T::from_bytes(chunk);
            data.push(element);
        }

        Ok(NDTensor { data, shape, rtype })
    }
}

// Type aliases for common NDTensor types
pub type NDTensorF16 = NDTensor<F16>;
pub type NDTensorF32 = NDTensor<f32>;
pub type NDTensorF64 = NDTensor<f64>;
pub type NDTensorU8 = NDTensor<u8>;
pub type NDTensorI32 = NDTensor<i32>;
pub type NDTensorI64 = NDTensor<i64>;
