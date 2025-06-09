# Rust Processor

A transformer processor library written in Rust, featuring support for both native and WebAssembly targets.
This library is written with Cursor.

## Overview

This workspace contains:
- `processor/`: The core processor library (`rust_processor`), pure processing logic
- `example/`: Example applications using the processor library

### Native Compilation
```bash
cd example
git clone https://huggingface.co/mlx-community/gemma-3-4b-it-4bit

cargo build --release
../target/release/main
```

### WebAssembly Compilation
```bash
cargo build -p rust_processor --target wasm32-wasip1 --release
```