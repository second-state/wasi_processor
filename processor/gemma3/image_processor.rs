#![allow(dead_code)]
use image::ImageReader;
use image::{DynamicImage, imageops};
use std::fs;
use std::io::Cursor;

/// PIL Image resampling methods
#[derive(Debug, Clone)]
pub enum PILImageResampling {
    Bilinear,
    Nearest,
    Lanczos,
    Bicubic,
}

/// Channel dimension ordering for image data
#[derive(Debug, Clone)]
pub enum ChannelDimension {
    First,  // (channels, height, width)
    Last,   // (height, width, channels) 
    None,   // (height, width)
}

/// Gemma3 Image Processor for handling image preprocessing
#[derive(Debug, Clone)]
pub struct Gemma3ImageProcessor {
    pub image_token_index: u32,
    pub image_seq_length: usize,
    pub do_resize: bool,
    pub size: (u32, u32),      // (height, width)
    pub resample: PILImageResampling,
    pub do_rescale: bool,
    pub rescale_factor: f32,
    pub do_normalize: bool,
    pub image_mean: Vec<f32>,
    pub image_std: Vec<f32>,
    pub do_convert_rgb: bool,
    pub do_pan_and_scan: bool,
    pub pan_and_scan_min_crop_size: u32,
    pub pan_and_scan_max_num_crops: u32,
    pub pan_and_scan_min_ratio_to_activate: f32,
}

impl Default for Gemma3ImageProcessor {
    fn default() -> Self {
        Self {
            image_token_index: 262144,
            image_seq_length: 256,
            do_resize: true,
            size: (896, 896),  // Gemma3 default size
            resample: PILImageResampling::Bilinear,
            do_rescale: true,
            rescale_factor: 1.0 / 255.0,
            do_normalize: true,
            image_mean: vec![0.5, 0.5, 0.5],  // Gemma3 normalization parameters
            image_std: vec![0.5, 0.5, 0.5],   // Gemma3 normalization parameters
            do_convert_rgb: true,
            do_pan_and_scan: false,
            pan_and_scan_min_crop_size: 256,
            pan_and_scan_max_num_crops: 4,
            pan_and_scan_min_ratio_to_activate: 1.2,
        }
    }
}

impl Gemma3ImageProcessor {
    /// Load image processor from pretrained model directory
    /// 
    /// # Arguments
    /// * `model_dir` - Path to the model directory
    /// 
    /// # Returns
    /// * `Self` - Configured Gemma3ImageProcessor instance
    pub fn from_pretrained(model_dir: &str) -> Self {
        // Read config.json
        let config_path = format!("{}/config.json", model_dir);
        let config_str = fs::read_to_string(&config_path).expect("Unable to read config.json");
        let config: serde_json::Value = serde_json::from_str(&config_str).expect("Failed to parse config.json");
        let image_token_index = config["image_token_index"].as_u64().unwrap_or(262144) as u32;
        
        // Read processor_config.json
        let proc_path = format!("{}/processor_config.json", model_dir);
        let proc_str = fs::read_to_string(&proc_path).expect("Unable to read processor_config.json");
        let proc_json: serde_json::Value = serde_json::from_str(&proc_str).expect("Failed to parse processor_config.json");
        let image_seq_length = proc_json["image_seq_length"].as_u64().unwrap_or(256) as usize;
        
        // Create and configure processor
        let mut processor = Self::default();
        processor.image_token_index = image_token_index;
        processor.image_seq_length = image_seq_length;
        
        // Read additional image processing parameters if available
        if let Some(obj) = proc_json["image_processor"].as_object() {
            processor.configure_from_json(obj);
        }
        
        processor
    }
    
    /// Configure processor from JSON object
    /// 
    /// # Arguments
    /// * `config` - JSON object containing configuration parameters
    fn configure_from_json(&mut self, config: &serde_json::Map<String, serde_json::Value>) {
        if let Some(value) = config.get("do_resize") {
            self.do_resize = value.as_bool().unwrap_or(true);
        }
        
        if let Some(size) = config.get("size") {
            if let Some(size_obj) = size.as_object() {
                if let (Some(height), Some(width)) = (size_obj.get("height"), size_obj.get("width")) {
                    self.size = (
                        height.as_u64().unwrap_or(224) as u32,
                        width.as_u64().unwrap_or(224) as u32
                    );
                }
            }
        }
        
        if let Some(value) = config.get("do_rescale") {
            self.do_rescale = value.as_bool().unwrap_or(true);
        }
        
        if let Some(value) = config.get("rescale_factor") {
            self.rescale_factor = value.as_f64().unwrap_or(1.0 / 255.0) as f32;
        }
        
        if let Some(value) = config.get("do_normalize") {
            self.do_normalize = value.as_bool().unwrap_or(true);
        }
        
        if let Some(value) = config.get("image_mean") {
            if let Some(array) = value.as_array() {
                self.image_mean = array.iter()
                    .filter_map(|v| v.as_f64().map(|f| f as f32))
                    .collect();
            }
        }
        
        if let Some(value) = config.get("image_std") {
            if let Some(array) = value.as_array() {
                self.image_std = array.iter()
                    .filter_map(|v| v.as_f64().map(|f| f as f32))
                    .collect();
            }
        }
        
        if let Some(value) = config.get("do_convert_rgb") {
            self.do_convert_rgb = value.as_bool().unwrap_or(true);
        }
        
        if let Some(value) = config.get("do_pan_and_scan") {
            self.do_pan_and_scan = value.as_bool().unwrap_or(false);
        }
        
        if let Some(value) = config.get("pan_and_scan_min_crop_size") {
            self.pan_and_scan_min_crop_size = value.as_u64().unwrap_or(256) as u32;
        }
        
        if let Some(value) = config.get("pan_and_scan_max_num_crops") {
            self.pan_and_scan_max_num_crops = value.as_u64().unwrap_or(4) as u32;
        }
        
        if let Some(value) = config.get("pan_and_scan_min_ratio_to_activate") {
            self.pan_and_scan_min_ratio_to_activate = value.as_f64().unwrap_or(1.2) as f32;
        }
        
        // Handle resample parameter
        if let Some(value) = config.get("resample") {
            if let Some(resample_str) = value.as_str() {
                self.resample = match resample_str.to_lowercase().as_str() {
                    "bilinear" => PILImageResampling::Bilinear,
                    "nearest" => PILImageResampling::Nearest,
                    "lanczos" => PILImageResampling::Lanczos,
                    "bicubic" => PILImageResampling::Bicubic,
                    _ => PILImageResampling::Bilinear,
                };
            }
        }
    }
    
    /// Pan and Scan method - consistent with Python version
    /// 
    /// # Arguments
    /// * `image` - Input image to process
    /// 
    /// # Returns
    /// * `Vec<DynamicImage>` - Vector of cropped images
    pub fn pan_and_scan(&self, image: &DynamicImage) -> Vec<DynamicImage> {
        if !self.do_pan_and_scan {
            return vec![];
        }
        
        let (width, height) = (image.width(), image.height());
        let mut crops = Vec::new();
        
        // Check if pan-and-scan should be activated
        if width >= height {
            // Landscape image
            if (width as f32) / (height as f32) < self.pan_and_scan_min_ratio_to_activate {
                return crops;
            }
            
            crops.extend(self.generate_horizontal_crops(image, width, height));
        } else {
            // Portrait image
            if (height as f32) / (width as f32) < self.pan_and_scan_min_ratio_to_activate {
                return crops;
            }
            
            crops.extend(self.generate_vertical_crops(image, width, height));
        }
        
        crops
    }
    
    /// Generate horizontal crops for landscape images
    /// 
    /// # Arguments
    /// * `image` - Input image
    /// * `width` - Image width
    /// * `height` - Image height
    /// 
    /// # Returns
    /// * `Vec<DynamicImage>` - Vector of horizontal crops
    fn generate_horizontal_crops(&self, image: &DynamicImage, width: u32, height: u32) -> Vec<DynamicImage> {
        let mut crops = Vec::new();
        
        // Calculate number of crops - consistent with Python logic
        let mut num_crops_w = ((width as f32) / (height as f32) + 0.5).floor() as u32;
        num_crops_w = num_crops_w.min(width / self.pan_and_scan_min_crop_size);
        num_crops_w = num_crops_w.max(2);
        num_crops_w = num_crops_w.min(self.pan_and_scan_max_num_crops);
        
        let crop_size_w = ((width as f32) / (num_crops_w as f32)).ceil() as u32;
        
        // Check if crop size is sufficient
        if crop_size_w < self.pan_and_scan_min_crop_size {
            return crops;
        }
        
        // Generate crops
        for i in 0..num_crops_w {
            let x = crop_size_w * i;
            let actual_width = crop_size_w.min(width - x);
            crops.push(image.crop_imm(x, 0, actual_width, height));
        }
        
        crops
    }
    
    /// Generate vertical crops for portrait images
    /// 
    /// # Arguments
    /// * `image` - Input image
    /// * `width` - Image width
    /// * `height` - Image height
    /// 
    /// # Returns
    /// * `Vec<DynamicImage>` - Vector of vertical crops
    fn generate_vertical_crops(&self, image: &DynamicImage, width: u32, height: u32) -> Vec<DynamicImage> {
        let mut crops = Vec::new();
        
        // Calculate number of crops - consistent with Python logic
        let mut num_crops_h = ((height as f32) / (width as f32) + 0.5).floor() as u32;
        num_crops_h = num_crops_h.min(height / self.pan_and_scan_min_crop_size);
        num_crops_h = num_crops_h.max(2);
        num_crops_h = num_crops_h.min(self.pan_and_scan_max_num_crops);
        
        let crop_size_h = ((height as f32) / (num_crops_h as f32)).ceil() as u32;
        
        // Check if crop size is sufficient
        if crop_size_h < self.pan_and_scan_min_crop_size {
            return crops;
        }
        
        // Generate crops
        for i in 0..num_crops_h {
            let y = crop_size_h * i;
            let actual_height = crop_size_h.min(height - y);
            crops.push(image.crop_imm(0, y, width, actual_height));
        }
        
        crops
    }

    /// Main preprocessing method - consistent with Python version functionality
    /// 
    /// # Arguments
    /// * `image_bytes` - Raw image bytes
    /// 
    /// # Returns
    /// * `(Vec<f32>, Vec<usize>)` - Processed pixel values and number of crops
    pub fn preprocess(&self, image_bytes: &[u8]) -> (Vec<f32>, Vec<usize>) {
        // 1. Decode image
        let img = ImageReader::new(Cursor::new(image_bytes))
            .with_guessed_format()
            .expect("Unable to determine image format")
            .decode()
            .expect("Unable to decode image");
        
        // 2. Convert to RGB if needed
        let img = if self.do_convert_rgb {
            DynamicImage::ImageRgb8(img.to_rgb8())
        } else {
            img
        };
        
        // 3. Apply pan-and-scan if enabled
        let mut all_images = vec![img.clone()];
        let crops = self.pan_and_scan(&img);
        let num_crops = vec![crops.len()]; // Number of crops per original image
        all_images.extend(crops);
        
        // 4. Process all images
        let mut all_pixel_values = Vec::new();
        
        for image in all_images {
            let pixel_values = self.process_single_image(&image);
            all_pixel_values.extend(pixel_values);
        }
        
        (all_pixel_values, num_crops)
    }
    
    /// Process a single image
    /// 
    /// # Arguments
    /// * `image` - Input image to process
    /// 
    /// # Returns
    /// * `Vec<f32>` - Processed pixel values in channels-first format
    fn process_single_image(&self, image: &DynamicImage) -> Vec<f32> {
        // Resize image
        let processed_img = if self.do_resize {
            let filter = self.get_image_filter();
            image.resize_exact(self.size.1, self.size.0, filter)
        } else {
            image.clone()
        };
        
        // Convert to RGB and extract pixels
        let rgb_img = processed_img.to_rgb8();
        let (width, height) = (rgb_img.width(), rgb_img.height());
        
        // Process pixel values - using channels_first format
        let mut pixel_values = Vec::with_capacity((width * height * 3) as usize);
        
        for c in 0..3 {
            for y in 0..height {
                for x in 0..width {
                    let pixel_value = rgb_img.get_pixel(x, y)[c] as f32;
                    let processed_value = self.process_pixel_value(pixel_value, c);
                    pixel_values.push(processed_value);
                }
            }
        }
        
        pixel_values
    }
    
    /// Get image filter based on resampling method
    /// 
    /// # Returns
    /// * `imageops::FilterType` - Corresponding filter type
    fn get_image_filter(&self) -> imageops::FilterType {
        match self.resample {
            PILImageResampling::Bilinear => imageops::FilterType::Triangle,
            PILImageResampling::Nearest => imageops::FilterType::Nearest,
            PILImageResampling::Lanczos => imageops::FilterType::Lanczos3,
            PILImageResampling::Bicubic => imageops::FilterType::CatmullRom,
        }
    }
    
    /// Process a single pixel value
    /// 
    /// # Arguments
    /// * `pixel_value` - Raw pixel value
    /// * `channel` - Channel index (0, 1, or 2)
    /// 
    /// # Returns
    /// * `f32` - Processed pixel value
    fn process_pixel_value(&self, pixel_value: f32, channel: usize) -> f32 {
        // Rescale
        let mut value = if self.do_rescale {
            pixel_value * self.rescale_factor
        } else {
            pixel_value
        };
        
        // Normalize
        if self.do_normalize && channel < self.image_mean.len() && channel < self.image_std.len() {
            value = (value - self.image_mean[channel]) / self.image_std[channel];
        }
        
        value
    }

    /// Validate processor parameters
    /// 
    /// # Returns
    /// * `Result<(), String>` - Ok if valid, Err with message if invalid
    pub fn validate_parameters(&self) -> Result<(), String> {
        if self.image_mean.len() != 3 {
            return Err("image_mean must contain 3 values".to_string());
        }
        if self.image_std.len() != 3 {
            return Err("image_std must contain 3 values".to_string());
        }
        if self.rescale_factor <= 0.0 {
            return Err("rescale_factor must be greater than 0".to_string());
        }
        if self.size.0 == 0 || self.size.1 == 0 {
            return Err("Image size must be greater than 0".to_string());
        }
        Ok(())
    }

    /// Check if image has already been scaled (utility method)
    /// 
    /// # Arguments
    /// * `sample_pixel` - Sample pixel value to check
    /// 
    /// # Returns
    /// * `bool` - True if image appears to be scaled
    fn _is_scaled_image(&self, sample_pixel: f32) -> bool {
        // Simple check: if pixel value is in 0-1 range, it may already be scaled
        sample_pixel <= 1.0
    }
}

/// Trait for image processors
pub trait ImageProcessor {
    /// Preprocess image bytes
    /// 
    /// # Arguments
    /// * `image` - Raw image bytes
    /// 
    /// # Returns
    /// * `(Vec<f32>, Vec<usize>)` - Processed pixel values and crop information
    fn preprocess(&self, image: &[u8]) -> (Vec<f32>, Vec<usize>);
}
